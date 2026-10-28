"""Production C++ execution vs the pinned training controller; no motor access."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
TRAINING = Path(os.environ.get('M20_TRAINING_ROOT', ROOT.parent / 'rl_training'))
sys.path.insert(0, str(TRAINING / 'source/rl_training'))
from task_support.stair_quiet_v1.landing import LandingController, LandingSettings
from task_support.stair_quiet_v1.terrain_memory import TerrainMemory


def budget():
    resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))


@pytest.fixture(scope='module')
def probe(tmp_path_factory):
    path = tmp_path_factory.mktemp('cts_execution') / 'probe'
    sdk = ROOT / 'src/m20_student_sdk'
    subprocess.run(['g++', '-std=c++17', '-O1', '-I/usr/include/eigen3',
                    '-I' + str(sdk / 'plugin'), '-I' + str(sdk / 'vendor/include/types'),
                    '-I' + str(sdk / 'vendor/include/utils'),
                    str(sdk / 'plugin/cts_execution_probe.cpp'), '-o', str(path)],
                   check=True, capture_output=True, text=True, timeout=90, preexec_fn=budget)
    return path


@pytest.fixture
def contract():
    value = json.loads((ROOT / 'models/cts_model10000/observation_contract.json').read_text())
    source = TRAINING / 'source/rl_training/task_support/stair_quiet_v1'
    for name, key in [('landing.py', 'source_sha256'), ('terrain_memory.py', 'memory_source_sha256')]:
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == value['quiet_execution_control'][key]
    return value


def run(probe, contract, tmp_path, mode, request, *, check=True):
    path = tmp_path / 'contract.json'
    path.write_text(json.dumps(contract))
    result = subprocess.run([str(probe), str(path), mode], input=json.dumps(request, allow_nan=False),
                            capture_output=True, text=True, timeout=15, check=False)
    if check:
        assert result.returncode == 0, result.stderr
    return json.loads(result.stdout) if check else result


def landing_frames(count=160):
    rng = np.random.default_rng(8421)
    frames = []
    for tick in range(count):
        q = rng.normal(0, .15, (4, 3)).astype('f')
        jac = np.repeat(np.eye(3, dtype='f')[None] * .15, 4, axis=0)
        jac += rng.normal(0, .01, jac.shape).astype('f')
        if 110 <= tick < 120:
            jac[0] = 0
        normal = np.tile([0., 0., 1.], (4, 1)).astype('f')
        forces = np.array([0 if tick % 60 < 35 else 40, 40, 40, 40], dtype='f')
        if 80 <= tick < 100:
            forces[1:] = [4., 11.999, 12.]
        valid = np.ones(4, dtype=bool)
        if 45 <= tick < 55:
            valid[0] = False
        terrain = np.ones(4, dtype=bool)
        if tick % 40 > 30:
            terrain[0] = False
        target = q.copy()
        target[:, 2] -= .5
        frames.append(dict(nominal_leg_target=target.tolist(), nominal_wheel_velocity=[1., -2., 3., -4.],
                           joint_position=q.tolist(), joint_velocity=rng.normal(0, .1, (4, 3)).astype('f').tolist(),
                           normal_jacobian=jac[:, 2].tolist(), point_jacobian=jac.tolist(), surface_normal=normal.tolist(),
                           normal_velocity=[-.4, -.2, .0, -.3], clearance=[.025, .01, .02, .03],
                           terrain_valid=terrain.tolist(), contact_force=forces.tolist(), contact_valid=valid.tolist(),
                           command_active=tick < 145, dt=.005))
    return frames


@pytest.mark.parametrize('full', [True, False])
def test_landing_matches_training_state_sequence(probe, contract, tmp_path, full):
    frames = landing_frames()
    if not full:
        for frame in frames:
            del frame['point_jacobian'], frame['surface_normal']
    core = LandingController(1, 'cpu', LandingSettings(**contract['quiet_execution_control']['settings']))
    expected = []
    for frame in frames:
        args = {k: torch.as_tensor([v]) for k, v in frame.items() if k != 'dt'}
        expected.append(core.step(**args, dt=frame['dt']))
    result = run(probe, contract, tmp_path, 'landing', frames)
    for actual, reference in zip(result, expected):
        for key in ('leg_target', 'correction', 'wheel_velocity', 'kp_scale', 'kd_scale', 'safe_speed'):
            np.testing.assert_allclose(actual[key], reference[key][0].numpy(), atol=2e-6, rtol=2e-5, err_msg=key)
        for key in ('phase', 'modified', 'invalid', 'infeasible'):
            np.testing.assert_array_equal(actual[key], reference[key][0].numpy(), err_msg=key)
    assert result[-1]['counts'] == core.counts.tolist()
    assert result[-1]['counts'][1] > 0 and result[-1]['counts'][2] > 0 and result[-1]['counts'][4] > 0


def test_source_and_nominal_fail_closed(probe, contract, tmp_path):
    altered = copy.deepcopy(contract)
    altered['quiet_execution_control']['source_sha256'] = '0' * 64
    result = run(probe, altered, tmp_path, 'landing', [], check=False)
    assert result.returncode == 2 and 'source/semantics' in result.stderr
    frames = landing_frames(1)
    frames[0]['dt'] = 0
    result = run(probe, contract, tmp_path, 'landing', frames, check=False)
    assert result.returncode == 2 and 'dt' in result.stderr


def test_terrain_memory_capture_age_conflicts_and_pose_epochs(probe, contract, tmp_path):
    settings = LandingSettings(**contract['quiet_execution_control']['settings'])
    count = 8
    core = TerrainMemory(1, count, 'cpu', settings)
    points = np.array([[.5, 0, .1], [-.5, 0, .2], [.1, .1, .1], [-.1, -.1, .2],
                       [.025, .025, .1], [-.025, -.025, .1], [.5, .0, .1], [.7, .3, .2]], dtype='f')
    frames, expected = [], []
    for tick in range(80):
        now = np.float32(tick * .05)
        pose = np.array([.3 if tick >= 48 else 0., 0, 0], dtype='f')
        yaw = np.float32(.5 if tick >= 62 else 0.)
        p = points.copy()
        if tick % 5 == 0:
            p[2, 0] = np.nextafter(np.float32(.1), np.float32(-np.inf))
        if 20 <= tick < 30:
            p[6, 2] += .1
        age = np.full(count, .1 if tick < 35 else .4, dtype='f')
        valid = np.full(count, tick < 38 or tick >= 45, dtype=bool)
        if tick == 42:
            now = np.float32(.5)
        frame = dict(points_w=p.tolist(), valid=valid.tolist(), age_s=age.tolist(),
                     pose_w=pose.tolist(), yaw_w=float(yaw), time_s=float(now),
                     wheel_xy=[[.5, 0, 0], [-.5, 0, 0], [.1, .1, 0], [-.1, -.1, 0]])
        core.update(points_w=torch.tensor(p[None]), valid=torch.tensor(valid[None]), age_s=torch.tensor(age[None]),
                    pose_w=torch.tensor(pose[None]), yaw_w=torch.tensor([yaw]), time_s=float(now))
        reference = core.sample(torch.tensor([frame['wheel_xy']], dtype=torch.float32)[..., :2], time_s=float(now))
        expected.append(([v[0].numpy() for v in reference], core.counts.tolist()))
        frames.append(frame)
    result = run(probe, contract, tmp_path, 'memory', dict(points_count=count, frames=frames))
    for actual, (reference, counts) in zip(result, expected):
        np.testing.assert_array_equal(actual['valid'], reference[1])
        np.testing.assert_allclose(np.array(actual['height'], dtype=float), reference[0], atol=2e-6, rtol=0, equal_nan=True)
        np.testing.assert_allclose(np.array(actual['age'], dtype=float), reference[2], atol=2e-6, rtol=0, equal_nan=True)
        assert actual['counts'] == counts


def binding():
    return dict(source_id='offline_fixture', hardware_serial='offline_fixture', world_frame='odom',
                binding_sha256='a' * 64, publisher_node='offline_fixture')


def feedback(tick, *, contact_valid=True):
    now = 100. + tick * .005
    sample = dict(schema='m20_cts_landing_feedback.v1', joint_order='sdk16', leg_order='FL_FR_HL_HR',
                  jacobian_columns='policy_leg_joints', kinematics_valid=True,
                  **{k: v for k, v in binding().items() if k != 'publisher_node'}, session=1, sequence=tick + 1,
                  capture_monotonic_ns=round(now * 1e9), joint_position_sdk_rad=[0.] * 16,
                  joint_velocity_sdk_rad_s=[0.] * 16, contact_support_force_N=[0., 40., 40., 40.],
                  contact_valid=[contact_valid] * 4, contact_capture_monotonic_ns=[round(now * 1e9)] * 4,
                  support_normal_velocity_world_m_s=[-.4, 0., 0., 0.],
                  support_position_world_m=[.5, 0., .025] * 4,
                  point_jacobian_world_m_per_rad=(np.eye(3, dtype='f') * .15).flatten().tolist() * 4,
                  base_position_world_m=[0., 0., .6], base_yaw_world_rad=0.,
                  terrain_query=dict(valid=True, session=1, sequence=1, stamp_ros_s=50.,
                                     position_world_m=[0., 0., .6], yaw_world_rad=0.))
    return sample


@pytest.mark.parametrize('corruption', ['missing', 'zero_contact_stamp', 'future', 'source', 'order', 'validity', 'replay', 'session'])
def test_measured_feedback_rejects_corrupt_or_stale_source(probe, contract, tmp_path, corruption):
    first, second = feedback(0), feedback(1)
    if corruption == 'missing':
        del second['contact_valid']
    elif corruption == 'zero_contact_stamp':
        second['contact_capture_monotonic_ns'][0] = 0
    elif corruption == 'future':
        second['capture_monotonic_ns'] += 1_000_000_000
    elif corruption == 'source':
        second['source_id'] = 'another'
    elif corruption == 'order':
        second['leg_order'] = 'FR_FL_HL_HR'
    elif corruption == 'validity':
        second['contact_valid'][0] = 1
    elif corruption == 'replay':
        second = copy.deepcopy(first)
    else:
        second['session'] = 2
    result = run(probe, contract, tmp_path, 'feedback', dict(binding=binding(), maximum_age_s=.04,
        frames=[dict(sample=first, received_s=100., now_s=100.),
                dict(sample=second, received_s=100.005, now_s=100.005),
                dict(sample=feedback(2), received_s=100.01, now_s=100.01)]))
    assert result[0]['passed'] is True
    assert result[1]['passed'] is False and result[2]['passed'] is False


def test_four_substeps_one_policy_tick_and_missing_tick_rejected(probe, contract, tmp_path):
    rows = [dict(tick=i, time_s=i * .005) for i in range(8)]
    rows += [dict(tick=9, time_s=.045), dict(tick=0, time_s=1., reset=True), dict(tick=1, time_s=1.01)]
    result = run(probe, contract, tmp_path, 'clock', rows)
    assert [i for i, row in enumerate(result[:8]) if row['infer']] == [0, 4]
    assert all(row['passed'] for row in result[:8])
    assert not result[8]['passed'] and not result[-1]['passed']


@pytest.mark.parametrize('change', ['pose', 'expired_source'])
def test_execution_noncompounding_gains_hold_and_pose_jump(probe, contract, tmp_path, change):
    robot = json.loads((ROOT / 'deploy/robot.json').read_text())
    nominal = np.zeros((16, 5), dtype='f')
    nominal[:, 0] = 80.
    nominal[:, 2] = 2.
    nominal[np.arange(16) % 4 == 3, 0] = 0.
    nominal[np.arange(16) % 4 == 3, 3] = [1., 2., 3., 4.]
    nominal[np.arange(16) % 4 != 3, 1] = -.3
    nominal[np.arange(16) % 4 == 2, 1] = -.5
    obs = np.zeros(789, dtype='f')
    obs[57:244] = .1
    obs[244:618] = 1.
    frames = []
    for tick in range(70):
        sample = feedback(tick)
        sample['base_position_world_m'][0] = .31 if tick >= 60 and change == 'pose' else 0.
        frame = dict(now_s=100. + tick * .005, nominal=nominal.tolist(), feedback=sample, raw_command=[.5, 0., 0.])
        if tick % 4 == 0 and tick < 60:
            frame.update(obs=obs.tolist(), age_s=[.01] * 187, map_session=1, map_sequence=1, query_stamp_ros_s=50.)
        if tick == 59:
            frame['hold'] = True
        if tick == 60 and change == 'expired_source':
            frame['terrain_source_session'] = 2  # no usable map accompanies it
        frames.append(frame)
    result = run(probe, contract, tmp_path, 'execution', dict(robot=robot, start_s=100., binding=binding(), frames=frames))
    commands = np.array([r['command'] for r in result])
    leg = np.arange(16) % 4 != 3
    assert commands[:, leg, 0].min() >= 80 * .85 - 1e-5
    assert commands[:, leg, 0].min() < 80.
    moving = np.delete(commands, 59, axis=0)
    np.testing.assert_array_equal(moving[:, ~leg, 3], np.tile([1., 2., 3., 4.], (69, 1)))
    np.testing.assert_array_equal(commands[59, :, :3], commands[58, :, :3])
    assert result[-1]['counts'][2] > result[59]['counts'][2]


@pytest.mark.parametrize('value', [1, 2**63 - 1, 2**63, 2**64 - 1])
def test_uint64_feedback_and_terrain_sessions(probe, contract, tmp_path, value):
    sample = feedback(0)
    sample['session'] = sample['terrain_query']['session'] = value
    result = run(probe, contract, tmp_path, 'feedback', dict(binding=binding(), maximum_age_s=.04,
                 frames=[dict(sample=sample, received_s=100., now_s=100.)]))
    assert result[0]['passed']


@pytest.mark.parametrize('value', [-1, 0, True, 1.5])
def test_nonpositive_or_noninteger_session_rejected(probe, contract, tmp_path, value):
    sample = feedback(0)
    sample['terrain_query']['session'] = value
    result = run(probe, contract, tmp_path, 'feedback', dict(binding=binding(), maximum_age_s=.04,
                 frames=[dict(sample=sample, received_s=100., now_s=100.)]))
    assert not result[0]['passed']


def test_live_replay_verifier_rejects_nominal_only_dead_wire(tmp_path):
    from m20_heightmap.cts import CTSInference
    spec = importlib.util.spec_from_file_location('cts_live_verify', ROOT / 'scripts/replay_cts_live.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_dir = ROOT / 'models/cts_model10000'
    inference = CTSInference(model_dir / 'policy.onnx', model_dir / 'observation_contract.json')
    history = [np.zeros(57, dtype='f')] * 3
    rows = []
    for tick in range(12):
        raw = np.array([0., 0., 0.] if tick < 4 else [.4, 0., 0.], dtype='f')
        obs = np.zeros(789, dtype='f')
        obs[8] = -.01 if tick < 4 else 0.
        obs[618:] = np.concatenate(history)
        action = inference(obs[None], raw[None])[0]
        nominal = np.zeros((16, 5), dtype='f')
        nominal[:, 0] = 80
        steps = np.tile(nominal, (4, 1, 1))
        rows.append(dict(obs=obs.tolist(), raw_command=raw.tolist(), nominal_action=action.tolist(),
                         nominal_command=nominal.tolist(), substeps=steps.tolist(),
                         landing_counts=[(tick + 1) * 16, 1, 0, 0, 0, 0]))
        history = history[1:] + [obs[:57].copy()]
    report = tmp_path / 'deadwire.json'
    report.write_text(json.dumps(dict(hardware_commands=0, command_publishers=0, frames=rows)))
    with pytest.raises(ValueError, match='nominal-only'):
        module.verify(report, model_dir)
    # Positive control for the new discriminator; the actual live integration
    # positive is a separate ARM replay receipt, never this synthetic report.
    rows[-1]['substeps'][0][0][0] = 72.
    report.write_text(json.dumps(dict(hardware_commands=0, command_publishers=0, frames=rows)))
    assert module.verify(report, model_dir)['modified_substeps'] == 1



@pytest.mark.parametrize("delta,accepted", [(.001499,False),(.0015,True),
    (.002266411,True),(.005,True),(.007677502,True),(.0085,True),(.008501,False)])
def test_clock_small_jitter_boundary(probe, contract, tmp_path, delta, accepted):
    result = run(probe, contract, tmp_path, 'clock',
                 [dict(tick=0,time_s=0.),dict(tick=1,time_s=delta)])
    assert result[1]['passed'] is accepted


@pytest.mark.parametrize("base", [100., 1.e6])
@pytest.mark.parametrize("delta,accepted", [(.001499,False),(.001501,True),
    (.002266411,True),(.007677502,True),(.008499,True),(.008501,False)])
def test_clock_jitter_independent_of_timestamp_origin(probe, contract, tmp_path, base, delta, accepted):
    # SteadySeconds is a floating timestamp: exercise both sides with a margin
    # above subtraction roundoff rather than claim exact endpoint equivalence.
    result = run(probe, contract, tmp_path, 'clock',
                 [dict(tick=0,time_s=base),dict(tick=1,time_s=base+delta)])
    assert result[1]['passed'] is accepted
