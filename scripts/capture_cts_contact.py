#!/usr/bin/env python3
"""Bounded passive contact diagnostics. Never constructs a command publisher.

Keeps raw motor data, source header stamps, receive monotonic stamps and signed
inverse-dynamics estimates separately. A recording never grants CTS admission.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/m20_heightmap'))
from m20_heightmap.contact_estimator import ContactModel, vector

TOPICS = {'/JOINTS_DATA': 'drdds/msg/JointsData', '/IMU': 'sensor_msgs/msg/Imu',
          '/MOTION_INFO': 'drdds/msg/MotionInfo', '/MOTION_STATE': 'drdds/msg/MotionState',
          '/MOTION_STATUS': 'drdds/msg/MotionStatus'}


def json_safe(value):
    """Keep nonfinite raw feedback as tagged IEEE float64 bits in strict JSON."""
    if isinstance(value, float) and not np.isfinite(value):
        return {'nonfinite_float64_be_hex': struct.pack('!d', value).hex()}
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def quaternion_rotation(q):
    x, y, z, w = vector(q, 4, 'IMU quaternion')
    norm = float(np.linalg.norm([x, y, z, w]))
    if abs(norm-1) > .01:
        raise ValueError('Invalid IMU quaternion norm')
    x, y, z, w = np.array([x, y, z, w])/norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def header_stamp(payload):
    stamp = payload['header']['stamp']
    sec, ns = stamp['sec'], stamp['nanosec']
    if type(sec) is not int or type(ns) is not int or sec < 0 or not 0 <= ns < 10**9:
        raise ValueError('Invalid source timestamp')
    return sec*10**9+ns


class Diagnostics:
    def __init__(self, robot, geometry):
        joints = robot['joints']
        if joints['sdk_order'] != geometry['joint_order']:
            raise ValueError('Geometry and SDK joint order differ')
        self.direction = vector(joints['sdk_direction'], 16, 'direction')
        if not np.isin(self.direction, [-1, 1]).all():
            raise ValueError('Invalid joint directions')
        self.offset = np.deg2rad(vector(joints['sdk_initial_offset_deg'], 16, 'offset'))
        self.model = ContactModel(geometry)
        self.previous_joint = self.previous_imu = self.imu = None
        self.offset_initialized = False

    def consume(self, topic, payload, received_ns, *, estimate=True):
        stamp = header_stamp(payload)
        if topic == '/IMU':
            q = payload['orientation']; av = payload['angular_velocity']; ac = payload['linear_acceleration']
            r = quaternion_rotation([q[k] for k in ('x', 'y', 'z', 'w')])
            omega = vector([av[k] for k in ('x', 'y', 'z')], 3, 'omega')
            specific = vector([ac[k] for k in ('x', 'y', 'z')], 3, 'specific force')
            if self.previous_imu:
                dt = (stamp-self.previous_imu[0])*1e-9
                if not .001 <= dt <= .05:
                    self.previous_imu = None; self.imu = None
                    raise ValueError('IMU derivative clock reset/gap')
                alpha = (omega-self.previous_imu[1])/dt
                self.imu = (received_ns, r, specific, omega, alpha)
            self.previous_imu = (stamp, omega)
            return None
        if topic != '/JOINTS_DATA':
            return None
        data = payload['data']['joints_data']
        if len(data) != 16:
            raise ValueError('Joint array shape differs from SDK16')
        for i, joint in enumerate(data):
            name = bytes(joint['name']).split(b'\0')[0].decode('ascii')
            if name != str(i):
                raise ValueError('Raw joint identity/order changed')
        raw_q, raw_dq, raw_tau = (vector([j[key] for j in data], 16, key)
                                  for key in ('position', 'velocity', 'torque'))
        q = raw_q*self.direction+self.offset
        if not self.offset_initialized:
            # Same periodic branch as M20Interface, without its reset command.
            for i in range(16):
                limit = np.deg2rad(140 if i % 4 == 1 else 164)
                if i % 4 in (1, 2):
                    self.offset[i] += 2*np.pi if q[i] < -limit else (-2*np.pi if q[i] > limit else 0)
            self.offset_initialized = True; q = raw_q*self.direction+self.offset
        dq, tau = raw_dq*self.direction, raw_tau*self.direction
        previous = self.previous_joint; self.previous_joint = (stamp, dq)
        if previous is None or self.imu is None:
            return None
        dt = (stamp-previous[0])*1e-9
        if not .001 <= dt <= .05:
            raise ValueError('Joint derivative clock reset/gap')
        if not 0 <= received_ns-self.imu[0] <= 50_000_000:
            raise ValueError('No recent preceding IMU sample')
        if not estimate:
            return None
        _, r, specific, omega, alpha = self.imu
        result = self.model.diagnose(tau, q=q, dq=dq, ddq=(dq-previous[1])/dt,
                                     base_rotation=r, base_specific_force=specific,
                                     omega_body=omega, alpha_body=alpha)
        result.update(joint_source_stamp_ns=stamp, imu_receive_age_ns=received_ns-self.imu[0],
                      effective_sdk_offsets_rad=self.offset.tolist(),
                      acceleration_method='causal backward difference of velocity; unfiltered',
                      alignment='preceding host-receive IMU; capture clock alignment not attested')
        return result


def record(output, seconds, robot, geometry, *, condition='lying', leg=None):
    # Import only inside the real read-only entry; tests/offline use no ROS.
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from rosidl_runtime_py.utilities import get_message
    from rosidl_runtime_py.convert import message_to_ordereddict
    if not 3 <= seconds <= 30:
        raise ValueError('Recording duration must be 3..30 seconds')
    if (condition not in ('lying', 'wheel_unloaded', 'wheel_loaded')
            or (condition == 'lying' and leg is not None)
            or (condition != 'lying' and leg not in ('FL', 'FR', 'HL', 'HR'))):
        raise ValueError('A labelled wheel recording needs exactly one leg; lying has no contact label')
    if Path('/opt/robot/os_type').read_text().strip() != 'CA9B_AOS':
        raise ValueError('Capture must run on AOS, the joint receive clock owner')
    if shutil.disk_usage(output.parent).free < 256*1024**2:
        raise ValueError('Insufficient disk reserve')
    output.mkdir()  # Exclusive new directory: never overwrite a prior capture.
    report = dict(schema='m20_contact_passive_capture.v1', condition=condition, labelled_leg=leg,
                  contact_label_origin='none' if condition=='lying' else 'operator_report; not an independent force calibration',
                  motor_command_publishers_created=0, hardware_validated=False,
                  hardware_serial=Path('/proc/device-tree/serial-number').read_bytes().rstrip(b'\0').decode(),
                  boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                  origin='hardware_recording', counts={t: 0 for t in TOPICS},
                  diagnostic_frames=0, errors={}, topics=TOPICS, clock='AOS CLOCK_MONOTONIC host_receive',
                  raw_nonfinite_encoding='tagged nonfinite_float64_be_hex; never converted to valid zero',
                  delivery_scope='best-effort depth 1; retains delivered callbacks, not a zero-loss DDS source recording',
                  writer_identity_scope='Foxy Python callback has no message info; graph endpoints only, not per-frame GID attestation',
                  caveats=['No contact labels; belly/knee/shank contact cannot be separated from wheel load',
                           'No torque bias/friction calibration; no contact normal observation',
                           'IMU lever arm/frame and source clock alignment not attested by this diagnostic',
                           'Finite differences amplify encoder noise; estimates are not valid CTS feedback'])
    for name, path in [('robot', robot), ('geometry', geometry)]:
        report[name+'_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    diag = Diagnostics(json.loads(robot.read_text()), json.loads(geometry.read_text()))
    rclpy.init()
    node = rclpy.create_node('m20_cts_passive_contact_capture', enable_rosout=False, start_parameter_services=False)
    subscriptions = []; byte_count = 0; last_diagnostic_attempt = 0; exhausted = False
    def endpoints():
        return {topic: sorted([dict(node=p.node_name, namespace=p.node_namespace, gid=list(p.endpoint_gid), type=p.topic_type)
                               for p in node.get_publishers_info_by_topic(topic)], key=lambda p:p['gid']) for topic in (*TOPICS, '/JOINTS_CMD')}
    trace = output/'samples.jsonl'
    try:
        with trace.open('x') as stream:
            def callback(topic, msg):
                nonlocal byte_count, last_diagnostic_attempt, exhausted
                if exhausted:
                    return
                now = time.monotonic_ns()
                payload = message_to_ordereddict(msg)
                item = dict(topic=topic, receive_monotonic_ns=now, data=payload,
                            publisher_gid=None)
                should_estimate = topic == '/JOINTS_DATA' and now-last_diagnostic_attempt >= 50_000_000
                if should_estimate:
                    last_diagnostic_attempt = now
                try:
                    estimate = diag.consume(topic, payload, now, estimate=should_estimate)
                    if estimate is not None:
                        item['diagnostic'] = estimate; report['diagnostic_frames'] += 1
                except (ValueError, KeyError, TypeError, np.linalg.LinAlgError) as exc:
                    reason = str(exc); report['errors'][reason] = report['errors'].get(reason, 0)+1
                    item['diagnostic_error'] = reason
                line = json.dumps(json_safe(item), separators=(',', ':'), allow_nan=False)+'\n'
                size = len(line.encode())
                if byte_count+size > 32*1024**2 or sum(report['counts'].values()) >= 20000:
                    exhausted = True; return
                stream.write(line); byte_count += size; report['counts'][topic] += 1
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
            def handler_for(topic):
                def handler(msg): callback(topic, msg)
                return handler
            for topic, typ in TOPICS.items():
                subscriptions.append(node.create_subscription(get_message(typ), topic, handler_for(topic), qos))
            discovery_end = time.monotonic()+2
            while time.monotonic()<discovery_end:
                rclpy.spin_once(node, timeout_sec=.02)
            report['publishers_before'] = endpoints()
            start = time.monotonic()
            while time.monotonic()-start < seconds and not exhausted:
                rclpy.spin_once(node, timeout_sec=.01)
            report['elapsed_after_discovery_s'] = time.monotonic()-start
            report['publishers_after'] = endpoints()
            report['publisher_set_stable'] = report['publishers_before'] == report['publishers_after']
            report['limits_reached'] = exhausted
    finally:
        node.destroy_node(); rclpy.shutdown()
        report['samples_sha256'] = hashlib.sha256(trace.read_bytes()).hexdigest()
        report['samples_bytes'] = trace.stat().st_size
        (output/'manifest.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=10)
    parser.add_argument('--robot', type=Path, default=ROOT/'deploy/robot.json')
    parser.add_argument('--geometry', type=Path, default=ROOT/'deploy/m20_contact_geometry.json')
    parser.add_argument('--condition', choices=['lying','wheel_unloaded','wheel_loaded'], default='lying')
    parser.add_argument('--leg', choices=['FL','FR','HL','HR'])
    args = parser.parse_args()
    record(args.output, args.seconds, args.robot, args.geometry, condition=args.condition, leg=args.leg)
