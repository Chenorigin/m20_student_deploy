#!/usr/bin/env python3
"""Bounded CPU physical-contact benchmark for the delivered passive estimator.

The force oracle is never an observer input. This does not start a CTS policy,
ROS, an optimizer, or hardware, and cannot grant hardware contact validity.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/m20_heightmap'))
from m20_heightmap.contact_estimator import ContactModel, ORDER, rpy_rotation

spec = importlib.util.spec_from_file_location('passive_capture', ROOT / 'scripts/capture_cts_contact.py')
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fmt(value):
    return ' '.join(map(str, np.asarray(value).reshape(-1)))


def quaternion(rotation):
    result = np.empty(4)
    mujoco.mju_mat2Quat(result, np.asarray(rotation).reshape(-1))
    return result


def build_model(protocol, geometry, variant, scene):
    tree = ET.parse(protocol['mjcf'])
    root = tree.getroot()
    # Remove only visual meshes. Explicit inertials and physical geoms remain.
    for parent in root.iter():
        for child in list(parent):
            if child.tag == 'mesh' or (child.tag == 'geom' and child.get('type') == 'mesh'):
                parent.remove(child)
    option = root.find('option')
    if option is None:
        option = ET.SubElement(root, 'option')
    option.set('timestep', str(protocol['dt_s']))
    option.set('gravity', '0 0 -9.81')
    option.set('integrator', 'Euler')
    world = root.find('worldbody')
    if scene == 'airborne':
        world.remove(world.find("geom[@name='floor']"))
    if scene == 'wall':
        wall = protocol.get('wall_geometry', dict(pos=[.59, 0, .55], size=[.02, 1, .55]))
        ET.SubElement(world, 'geom', name='test_wall', type='box',
                      pos=fmt(wall['pos']), size=fmt(wall['size']), friction='1 .01 .001',
                      condim='3', solref='.005 1')
    bodies = {b.get('name'): b for b in world.iter('body')}
    if variant != 'source_mjcf':
        for leg in geometry['legs']:
            for joint in leg['joints']:
                body = bodies[joint['child']]
                body.set('pos', fmt(joint['origin']['xyz']))
                body.set('quat', fmt(quaternion(rpy_rotation(joint['origin']['rpy']))))
                body.find('joint').set('axis', fmt(joint['axis']))
                body.remove(body.find('inertial'))
                inertia_rotation = rpy_rotation(joint['com']['rpy'])
                inertia = inertia_rotation @ np.asarray(joint['inertia']) @ inertia_rotation.T
                eigenvalues, axes = np.linalg.eigh(inertia)
                if np.linalg.det(axes) < 0:
                    axes[:, 0] *= -1
                ET.SubElement(body, 'inertial', mass=str(joint['mass']),
                              pos=fmt(joint['com']['xyz']), diaginertia=fmt(eigenvalues),
                              quat=fmt(quaternion(axes)))
            wheel = bodies[leg['joints'][-1]['child']]
            collision = leg['cylinder']
            for geom in wheel.findall('geom'):
                wheel.remove(geom)
            ET.SubElement(wheel, 'geom', type='capsule' if variant == 'matched_capsule' else 'cylinder',
                          size=fmt([collision['radius'], collision['half_width']]),
                          pos=fmt(collision['origin']['xyz']),
                          quat=fmt(quaternion(rpy_rotation(collision['origin']['rpy']))),
                          **{'class': 'collision'})
    xml = ET.tostring(root, encoding='unicode')
    model = mujoco.MjModel.from_xml_string(xml)
    # Same wheel armature as the existing deployment loader; never change legs.
    for name in ORDER[3::4]:
        model.dof_armature[model.joint(name).dofadr[0]] = .00243216
    return model, xml


def contact_oracle(model, data, wheel_bodies):
    """Independent actual contact wrench sum; sign checked by generalized force."""
    net = np.zeros((4, 3))
    support = np.zeros(4)
    generalized = np.zeros(model.nv)
    wheel_map = {int(body): i for i, body in enumerate(wheel_bodies)}
    nonwheel_contacts = side_contacts = 0
    for index in range(data.ncon):
        contact = data.contact[index]
        local = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, local)
        frame = contact.frame.reshape(3, 3)
        force, moment = frame.T @ local[:3], frame.T @ local[3:]
        bodies = [int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])]
        if np.linalg.norm(local) < 1e-10:
            continue
        if not any(body in wheel_map for body in bodies):
            nonwheel_contacts += 1
        for body, sign in zip(bodies, (-1., 1.)):
            if body != 0:
                jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
                mujoco.mj_jac(model, data, jacp, jacr, contact.pos, body)
                generalized += sign * (jacp.T @ force + jacr.T @ moment)
            if body in wheel_map:
                wheel = wheel_map[body]
                net[wheel] += sign * force
                normal_z = sign * frame[0, 2]
                if normal_z >= .8:
                    support[wheel] += max(0., local[0]) * normal_z
                else:
                    side_contacts += 1
    # Limits/friction are constraints too: retain their residual, do not hide it.
    residual = float(np.max(np.abs(generalized - data.qfrc_constraint)))
    return net, support, nonwheel_contacts, side_contacts, residual


def force_array(result):
    if result is None:
        return np.full((4, 3), np.nan)
    return np.asarray(result['estimated_force_world_N'], dtype=float)


def wire_payload(state, torque, direction, offset, stamp_ns):
    header = {'stamp': {'sec': stamp_ns // 10**9, 'nanosec': stamp_ns % 10**9}}
    q = quaternion(state['base_rotation'])
    imu = dict(header=header, orientation=dict(zip(('w', 'x', 'y', 'z'), q.tolist())),
               angular_velocity=dict(zip('xyz', state['omega_body'].tolist())),
               linear_acceleration=dict(zip('xyz', state['base_specific_force'].tolist())))
    raw_q = (state['q'] - offset) * direction
    raw_dq = state['dq'] * direction
    raw_tau = torque * direction
    joints = dict(header=header, data={'joints_data': [
        dict(name=list(str(i).encode()) + [0], position=float(raw_q[i]),
             velocity=float(raw_dq[i]), torque=float(raw_tau[i])) for i in range(16)]})
    return imu, joints


def fsm(force, settings, dt):
    """Only contact recurrence from landing.py; no terrain/phase/gain changes."""
    current, pending = np.zeros(4, bool), np.zeros(4)
    history = []
    for row in force:
        valid = np.isfinite(row)
        wants = valid & (row >= np.where(current, settings['contact_off_force_N'],
                                        settings['contact_on_force_N']))
        pending = np.where(wants, pending + dt, 0.)
        current = wants & (current | (pending >= settings['contact_confirm_s']))
        history.append(current.copy())
    return np.asarray(history)


def metrics(truth, estimate, protocol, stride=1):
    truth, estimate = truth[::stride], estimate[::stride]
    valid = np.isfinite(estimate)
    error = np.abs(estimate - truth)
    a = fsm(truth, protocol['settings'], protocol['dt_s'] * stride)
    b = fsm(estimate, protocol['settings'], protocol['dt_s'] * stride)
    gates = protocol['gates']
    finite_fraction = float(valid.mean())
    p95 = float(np.quantile(error[valid], .95)) if valid.any() else None
    agreement = float((a == b).mean())
    loaded = truth >= protocol['settings']['contact_on_force_N']
    air = truth < protocol['settings']['contact_off_force_N']
    rising_delays, missed_events = [], 0
    horizon = max(1, int(.04 / (protocol['dt_s'] * stride)))
    for wheel in range(4):
        for index in np.flatnonzero(a[1:, wheel] & ~a[:-1, wheel]) + 1:
            candidates = np.flatnonzero(b[index:index + horizon + 1, wheel])
            if len(candidates):
                rising_delays.append(float(candidates[0] * protocol['dt_s'] * stride))
            else:
                missed_events += 1
    return dict(samples=int(truth.size), finite_fraction=finite_fraction,
                force_mae_N=float(error[valid].mean()) if valid.any() else None,
                force_abs_error_p95_N=p95, force_abs_error_max_N=float(error[valid].max()) if valid.any() else None,
                fsm_agreement=agreement, fsm_false_support_samples=int((b & ~a).sum()),
                fsm_missed_support_samples=int((a & ~b).sum()),
                raw_air_samples=int(air.sum()), raw_loaded_samples=int(loaded.sum()),
                raw_air_false_on_samples=int((air & (estimate >= protocol['settings']['contact_on_force_N'])).sum()),
                raw_loaded_false_off_or_invalid_samples=int((loaded & ((estimate < protocol['settings']['contact_off_force_N']) | ~valid)).sum()),
                touchdown_delays_s=rising_delays, touchdown_missed_within_40ms=missed_events,
                screening_pass=bool(finite_fraction >= gates['finite_estimate_fraction_min']
                                    and p95 is not None and p95 <= gates['force_abs_error_p95_N']
                                    and agreement >= gates['contact_fsm_agreement_min']))


def run_case(protocol, geometry, robot, variant, scene, seed):
    model, xml = build_model(protocol, geometry, variant, scene)
    data = mujoco.MjData(model)
    qp = np.array([model.joint(name).qposadr[0] for name in ORDER])
    dv = np.array([model.joint(name).dofadr[0] for name in ORDER])
    actuators = np.array([int(np.flatnonzero(model.actuator_trnid[:, 0] == model.joint(name).id)[0]) for name in ORDER])
    wheel_bodies = [model.body(name[:-6]).id for name in ORDER[3::4]]
    base = model.body('base_link').id
    rng = np.random.default_rng(seed)
    target = np.asarray(robot['joints']['default_position_policy_rad'])[robot['joints']['sdk_from_policy_indices']]
    data.qpos[qp] = target
    data.qpos[2] = dict(airborne=1.5, flat=.58, drop=.78, wall=.58)[scene]
    if scene == 'wall':
        data.qvel[0] = .8
    data.qvel[dv] = rng.normal(0., .01, 16)
    observer = capture.Diagnostics(robot, geometry)
    core = ContactModel(geometry)
    kp, kd = (np.asarray(robot['joints'][key]) for key in ('kp_sdk', 'kd_sdk'))
    records = []
    previous_state = previous_torque = None
    for tick in range(round(protocol['scenes'][scene] / protocol['dt_s'])):
        requested_torque = kp * (target - data.qpos[qp]) - kd * data.qvel[dv]
        requested_torque = np.clip(requested_torque, -np.tile([76.4, 76.4, 76.4, 21.6], 4), np.tile([76.4, 76.4, 76.4, 21.6], 4))
        data.ctrl[actuators] = requested_torque
        # All oracle/observer fields below refer to this SAME pre-step snapshot.
        mujoco.mj_forward(model, data)
        rotation = data.xmat[base].reshape(3, 3).copy()
        state = dict(q=data.qpos[qp].copy(), dq=data.qvel[dv].copy(), ddq=data.qacc[dv].copy(),
                     base_rotation=rotation,
                     base_specific_force=rotation.T @ (data.qacc[:3] - model.opt.gravity),
                     omega_body=data.qvel[3:6].copy(), alpha_body=data.qacc[3:6].copy())
        # Measured motor force is actual actuator generalized force, not ctrl.
        torque = data.qfrc_actuator[dv].copy()
        ideal = core.diagnose(torque, **state)
        stamp = 10**9 + round(data.time * 10**9)
        imu, joints = wire_payload(state, torque, observer.direction, np.deg2rad(robot['joints']['sdk_initial_offset_deg']), stamp)
        observer.consume('/IMU', imu, stamp)
        causal = observer.consume('/JOINTS_DATA', joints, stamp)
        # Experimental ablation only: backward velocity differences belong to
        # the previous Euler interval, so use its pose, torque and specific
        # force as well. It is available NOW with a 5ms source age. This is not
        # a production implementation or a timestamp calibration for hardware.
        aligned = None
        if previous_state is not None:
            aligned_state = copy.copy(previous_state)
            aligned_state['ddq'] = (state['dq'] - previous_state['dq']) / protocol['dt_s']
            aligned_state['alpha_body'] = (state['omega_body'] - previous_state['omega_body']) / protocol['dt_s']
            aligned = core.diagnose(previous_torque, **aligned_state)
        net, support, nonwheel, side, residual = contact_oracle(model, data, wheel_bodies)
        # Separate perturbation: uncompensated physical IMU lever arm.
        lever = model.site('imu_site').pos
        shifted = copy.copy(state)
        shifted['base_specific_force'] = (state['base_specific_force']
            + np.cross(state['alpha_body'], lever)
            + np.cross(state['omega_body'], np.cross(state['omega_body'], lever)))
        # Evaluate lever-arm sensitivity at 50 Hz to bound CPU use.
        lever_estimate = force_array(core.diagnose(torque, **shifted)) if tick % 4 == 0 else np.full((4, 3), np.nan)
        records.append(dict(time_s=float(data.time), truth_world_N=net.tolist(), upward_normal_N=support.tolist(),
                            ideal_world_N=force_array(ideal).tolist(), causal_world_N=force_array(causal).tolist(),
                            aligned_delayed_world_N=force_array(aligned).tolist(),
                            aligned_source_time_s=float(data.time - protocol['dt_s']) if aligned is not None else None,
                            uncompensated_imu_world_N=lever_estimate.tolist(), nonwheel_contacts=nonwheel,
                            side_contacts=side, constraint_wrench_residual_Nm=residual,
                            base_z_m=float(data.qpos[2]), jacobian_condition=ideal['jacobian_condition'],
                            observer_input={key: value.tolist() for key, value in state.items() if key not in ('ddq', 'alpha_body')},
                            measured_torque_sdk_nm=torque.tolist()))
        previous_state, previous_torque = state, torque
        # Actual physical evolution; no qpos/qvel teleportation in the loop.
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or max(w.number for w in data.warning) > 0:
            raise RuntimeError('Nonfinite state or MuJoCo warning; experiment aborted')
    truth = np.maximum(np.asarray([r['truth_world_N'] for r in records])[:, :, 2], 0.)
    support = np.asarray([r['upward_normal_N'] for r in records])
    summary = dict(variant=variant, scene=scene, seed=seed, steps=len(records),
                   generated_model_sha256=hashlib.sha256(xml.encode()).hexdigest(),
                   body_mass_kg=model.body_mass.tolist(), joint_order=list(ORDER),
                   nonwheel_contact_frames=sum(r['nonwheel_contacts'] > 0 for r in records),
                   side_contact_frames=sum(r['side_contacts'] > 0 for r in records),
                   max_constraint_wrench_residual_Nm=max(r['constraint_wrench_residual_Nm'] for r in records),
                   truth_vs_adapter_abs_error_p95_N=float(np.quantile(np.abs(truth-support), .95)),
                   comparisons={})
    for key in ('ideal_world_N', 'causal_world_N', 'uncompensated_imu_world_N'):
        estimate = np.asarray([r[key] for r in records])[:, :, 2]
        stride = 4 if key == 'uncompensated_imu_world_N' else 1
        summary['comparisons'][key] = {
            'world_z': metrics(truth, estimate, protocol, stride),
            'upward_normal': metrics(support, estimate, protocol, stride)}
    aligned = np.asarray([r['aligned_delayed_world_N'] for r in records])[:, :, 2]
    summary['comparisons']['experimental_aligned_delayed'] = {
        'source_age_s': protocol['dt_s'], 'startup_unavailable_frames': 1,
        'world_z': metrics(truth[:-1], aligned[1:], protocol),
        'upward_normal': metrics(support[:-1], aligned[1:], protocol),
        'scope': '5ms delayed source-time comparison, not current-time zero-latency force'}
    # First sample has no finite difference; retain it as an invalid observation.
    summary['negative_controls'] = {
        'zero_force': metrics(truth, np.zeros_like(truth), protocol),
        'reverse_force': metrics(truth, -truth, protocol)}
    return records, summary, xml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    for path, digest in protocol['source_hashes'].items():
        if sha(path) != digest:
            raise ValueError('Source changed after protocol freeze: ' + path)
    signal.alarm(protocol['resource']['wall_timeout_s'])
    args.output.mkdir(parents=True, exist_ok=False)
    geometry = json.loads((ROOT / 'deploy/m20_contact_geometry.json').read_text())
    robot = json.loads((ROOT / 'deploy/robot.json').read_text())
    report = dict(schema='m20_contact_physical_sim_results.v1', protocol_sha256=sha(args.protocol),
                  script_sha256=sha(__file__), mujoco_version=mujoco.__version__,
                  estimator_path=str(ROOT / 'src/m20_heightmap/m20_heightmap/contact_estimator.py'),
                  hardware_validated=False, hardware_admission_granted=False,
                  policy_performance_tested=False, simulation_steps=0, cases=[])
    (args.output / 'protocol.json').write_bytes(args.protocol.read_bytes())
    for variant in protocol['variants']:
        for scene in protocol['scenes']:
            for seed in protocol['seeds']:
                records, summary, xml = run_case(protocol, geometry, robot, variant, scene, seed)
                name = f'{variant}_{scene}_{seed}'
                # JSON represents unavailable estimates as null, never valid zero.
                def clean(value):
                    if isinstance(value, float) and not np.isfinite(value):
                        return None
                    if isinstance(value, list):
                        return [clean(v) for v in value]
                    if isinstance(value, dict):
                        return {k: clean(v) for k, v in value.items()}
                    return value
                trace = args.output / (name + '.jsonl')
                trace.write_text(''.join(json.dumps(clean(row), allow_nan=False) + '\n' for row in records))
                (args.output / (name + '.xml')).write_text(xml)
                summary['trace_sha256'] = sha(trace)
                report['cases'].append(summary)
                report['simulation_steps'] += summary['steps']
                if report['simulation_steps'] > protocol['resource']['max_sim_steps']:
                    raise ValueError('Simulation step budget exceeded')
                (args.output / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
                print(json.dumps(dict(case=name, causal=summary['comparisons']['causal_world_N']['world_z'])), flush=True)
    report['completed'] = True
    report['screening_all_cases_pass'] = all(case['comparisons']['causal_world_N']['world_z']['screening_pass'] for case in report['cases'])
    (args.output / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
