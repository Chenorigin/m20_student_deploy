"""Real local ROS nodes; inject bounded scan stalls, never create motor publishers."""
from concurrent.futures import Future
import json
import os
from pathlib import Path
import threading
import time

import numpy as np
import pytest
import rclpy
from rclpy.executors import MultiThreadedExecutor
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, PointCloud2, PointField
from m20_heightmap_msgs.msg import TerrainObservation
from m20_heightmap import ros_nodes as rn
from m20_heightmap import native_inputs as ni
from m20_heightmap.geometry import Pose
from m20_heightmap.contract import GRID


ROOT = Path(__file__).resolve().parents[1]


class PendingWorker:
    """No real scan worker: its compute/IPC already has separate parity tests."""
    def __init__(self, *args, **kwargs):
        self.future = Future()
        self.submissions = 0

    def submit(self, *args):
        self.submissions += 1
        return self.future

    def shutdown(self, wait=True):
        pass


@pytest.fixture
def environment(tmp_path, monkeypatch):
    assert os.environ.get('ROS_LOCALHOST_ONLY') == '1'
    assert os.environ.get('ROS_DOMAIN_ID') == '176'
    cfg = json.loads((ROOT/'src/m20_heightmap/config/v1.json').read_text())
    cfg['calibration_confirmed'] = True
    cfg['lidar_extrinsics'] = {'lidar': {'translation_m': [0, 0, 0], 'quaternion_xyzw': [0, 0, 0, 1]}}
    cfg['raw_input'] = {'protocol': ni.PROTOCOL, 'source': ni.SOURCE}
    for key in cfg['topics']:
        cfg['topics'][key] = '/offline_terrain_fix/'+key
    path = tmp_path/'runtime/config.json'
    path.parent.mkdir()
    cfg['visibility'] = {'site_config': 'site.json'}
    (path.parent/'site.json').write_text((ROOT/'deploy/site.json').read_text())
    path.write_text(json.dumps(cfg))
    monkeypatch.setattr(rn, 'ProcessPoolExecutor', PendingWorker)
    monkeypatch.setattr(rn, 'geometry_hash', lambda *a: 'f'*64)
    monkeypatch.setattr(rn, 'loaded_geometry_commissioned', lambda *a: False)
    rclpy.init(args=['--ros-args', '-p', 'config:='+str(path)])
    yield cfg, path
    if rclpy.ok():
        rclpy.shutdown()


def spin_until(executor, condition, timeout=2.):
    end = time.monotonic()+timeout
    while time.monotonic() < end:
        executor.spin_once(timeout_sec=.01)
        if condition():
            return
    assert condition(), 'condition was not reached before bounded timeout'


def seed_map(node, when=None):
    now = node.now() if when is None else when
    pose = Pose(now-.01, [0, 0, .6], [0, 0, 0, 1], np.full(6, .0001))
    offsets = np.array([(x, y) for y in np.linspace(-.04, .04, 5) for x in np.linspace(-.04, .04, 5)])
    xy = (GRID[:, None]+offsets).reshape(-1, 2)
    points = np.column_stack((xy, np.zeros(len(xy))))
    with node.state_lock:
        node.pipeline.add_pose(pose)
        node.pipeline.mapper.ingest(points, np.linalg.norm(points-pose.position, axis=1), pose)
        node.pipeline.last_scan = pose
        node.last_pose_receive = node.last_cloud_receive = time.monotonic()
    return pose


@pytest.mark.parametrize('busy_python', [False, True])
def test_real_gos_groups_keep_pose_and_publish_running_during_scan_stall(environment, monkeypatch, busy_python):
    node = rn.GosNode()
    source = rclpy.create_node('offline_source')
    executor = MultiThreadedExecutor(num_threads=3)
    for n in (node, source):
        executor.add_node(n)
    entered, release = threading.Event(), threading.Event()
    outputs, sent_stamps = [], []
    original_pack = rn.pack_job_inputs

    def delayed_pack(msg, poses):
        entered.set()
        end = time.monotonic()+1.
        while not release.is_set() and time.monotonic() < end:
            if not busy_python:
                release.wait(.005)
        return original_pack(msg, poses)

    monkeypatch.setattr(rn, 'pack_job_inputs', delayed_pack)
    pub = source.create_publisher(Odometry, node.cfg['topics']['odom'], rn.sensor_qos(32))
    source.create_subscription(TerrainObservation, node.cfg['topics']['terrain'],
                               lambda m: outputs.append((time.monotonic(), rn.stamp(m))), rn.sensor_qos())

    def odom():
        msg = Odometry()
        msg.header.stamp = source.get_clock().now().to_msg()
        msg.header.frame_id = node.cfg['frames']['world']
        msg.child_frame_id = node.cfg['frames']['base']
        msg.pose.pose.position.z = .6
        msg.pose.pose.orientation.w = 1.
        sent_stamps.append(rn.stamp(msg))
        pub.publish(msg)

    timer = source.create_timer(.005, odom)
    try:
        seed_map(node)
        spin_until(executor, lambda: node.last_pose_receive is not None and len(node.pipeline.poses.poses) > 3)
        # Re-seed recent map support after DDS discovery; no fake publish callback.
        with node.state_lock:
            p = node.pipeline.poses.poses[-1]
            node.pipeline.mapper.patches[-1].cell_stamps[:] = p.stamp
            node.pipeline.mapper.patches[-1].cell_ends[:] = p.stamp
        cloud = PointCloud2()
        cloud.header.stamp = source.get_clock().now().to_msg()
        cloud.header.frame_id = 'lidar'
        cloud.height, cloud.width, cloud.point_step, cloud.row_step = 1, 0, 12, 0
        cloud.fields = [PointField(name=n, offset=i*4, datatype=PointField.FLOAT32, count=1)
                        for i, n in enumerate(('x', 'y', 'z'))]
        node.cloud(cloud)
        spin_until(executor, entered.is_set)
        start = time.monotonic()
        with node.state_lock:
            before = node.pipeline.poses.poses[-1].stamp
        spin_until(executor, lambda: any(received > start and stamp > before for received, stamp in outputs), .8)
        assert not release.is_set()
        with node.state_lock:
            assert node.pipeline.poses.poses[-1].stamp > before
        assert any(stamp in sent_stamps for _, stamp in outputs)
        assert node.worker.submissions == 0  # job is still blocked before submission
    finally:
        release.set()
        timer.cancel()
        executor.shutdown()
        source.destroy_node()
        node.destroy_node()


@pytest.mark.parametrize('change', ['expiry', 'epoch', 'worker_fault', 'cloud_watchdog', 'new_patch', 'none',
                                  'publish_budget', 'publish_budget_fresh', 'publish_budget_retry'])
def test_publish_revalidates_after_private_query(environment, monkeypatch, change):
    node = rn.GosNode()
    for timer in node.timers:
        timer.cancel()
    fixed = [10.03]
    monkeypatch.setattr(node, 'now', lambda: fixed[0])
    seed = seed_map(node, 10.01)
    if change.startswith('publish_budget'):
        node.cfg['runtime']['max_publish_query_age_s'] = .02
        if change == 'publish_budget_fresh':
            fixed[0] = 10.01
    original = rn.Pipeline.snapshot
    published = []
    real_publish = node.output.publish
    extra = None
    if change == 'new_patch':
        import copy
        extra = copy.deepcopy(node.pipeline.mapper.patches[-1])
        extra.anchor_stamp += .001

    def snapshot(pipeline, now):
        result = original(pipeline, now)
        assert pipeline is not node.pipeline
        assert pipeline.mapper.patches is not node.pipeline.mapper.patches
        if change == 'expiry':
            fixed[0] += .04
        elif change == 'epoch':
            with node.state_lock:
                node.pipeline.reset()
                node.session += 1
                node.map_generation += 1
        elif change == 'worker_fault':
            node.worker_fault = 'injected worker failure'
        elif change == 'cloud_watchdog':
            node.last_cloud_receive = time.monotonic()  # cannot freshen captured receipt
        elif change == 'new_patch':
            with node.state_lock:
                node.pipeline.mapper.patches.append(extra)
                node.map_generation += 1
        elif change == 'publish_budget_retry' and pipeline.poses.poses[-1].stamp == seed.stamp:
            node.push_pose(Pose(10.025, seed.position, seed.quaternion, seed.variance))
        return result

    def observe_publish(msg):
        published.append(msg)
        real_publish(msg)

    monkeypatch.setattr(rn.Pipeline, 'snapshot', snapshot)
    monkeypatch.setattr(node.output, 'publish', observe_publish)
    if change == 'cloud_watchdog':
        node.last_cloud_receive = time.monotonic()-1.
    try:
        node.tick()
        if change in ('expiry', 'epoch', 'worker_fault', 'publish_budget'):
            assert not published and node.sequence == 0 and node.last_error
        else:
            assert len(published) == 1 and node.sequence == 1
            expected_stamp = 10.025 if change == 'publish_budget_retry' else seed.stamp
            assert rn.stamp(published[0]) == expected_stamp
            if change == 'publish_budget_retry':
                assert node.query_retries == 1
                assert expected_stamp != fixed[0]
                assert min(a for a, valid in zip(published[0].age, published[0].validity) if valid) > .02
            if change == 'cloud_watchdog':
                assert not any(published[0].validity)
            else:
                assert any(published[0].validity)
            if extra is not None:
                assert any(p is extra for p in node.pipeline.mapper.patches)
    finally:
        node.destroy_node()


def test_worker_failure_latches_and_stops_publication(environment):
    node = rn.GosNode()
    for timer in node.timers:
        timer.cancel()
    try:
        seed_map(node)
        failure = Future()
        failure.set_exception(RuntimeError('worker unavailable'))
        node.future = failure
        node.job_tick()
        assert node.worker_fault == 'RuntimeError: worker unavailable'
        node.tick()
        assert node.sequence == 0 and 'worker failed' in node.last_error
        node.job_tick()
        assert node.worker.submissions == 0
    finally:
        node.destroy_node()


def test_native_imu_forwarding_continues_while_cloud_reconstruction_blocks(environment, monkeypatch):
    _, path = environment
    binary = path.parent.parent/'install/m20_native_inputs/lib/m20_native_inputs/native_lidar_input'
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b'offline identity fixture, never executed')
    read_text = Path.read_text
    monkeypatch.setattr(Path, 'read_text', lambda p, *a, **kw: 'CA9B_GOS' if str(p) == '/opt/robot/os_type' else read_text(p, *a, **kw))
    monkeypatch.setattr(ni.NativeInputs, 'reader', lambda self: self.stop.wait())
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def blocked_cloud(*args, **kwargs):
        entered.set()
        release.wait(1.)
        finished.set()
        raise ValueError('controlled cloud reconstruction failure')

    monkeypatch.setattr(ni, 'cloud_message', blocked_cloud)
    node = ni.NativeInputs()
    source = rclpy.create_node('offline_imu_source')
    executor = MultiThreadedExecutor(num_threads=3)
    for n in (node, source):
        executor.add_node(n)
    pub = source.create_publisher(Imu, '/IMU', rn.sensor_qos())
    received = []
    source.create_subscription(Imu, node.cfg['topics']['imu'], lambda m: received.append(m), rn.sensor_qos())

    def imu():
        msg = Imu()
        msg.header.stamp = source.get_clock().now().to_msg()
        msg.header.frame_id = 'imu_link'
        msg.linear_acceleration.z = 9.81
        pub.publish(msg)

    timer = source.create_timer(.005, imu)
    try:
        spin_until(executor, lambda: len(received) >= 3)
        with node.lock:
            node.latest = ({}, b'')
        spin_until(executor, entered.is_set)
        before = node.imu_count
        spin_until(executor, lambda: node.imu_count >= before+3, .7)
        assert not release.is_set() and not finished.is_set()
        assert all(m.linear_acceleration.z == 9.81 for m in received)
    finally:
        release.set()
        timer.cancel()
        executor.shutdown()
        source.destroy_node()
        node.destroy_node()
