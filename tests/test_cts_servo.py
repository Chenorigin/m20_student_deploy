"""Compile/run the real RLControlState without ROS, DDS or motor interfaces.

The include overlay replaces only dependencies. The servo step and observation
publication code come directly from the production RLControlState header.
"""
import json
from pathlib import Path
import resource
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SDK = ROOT / 'src/m20_student_sdk'
SCENARIOS = [
    'healthy_four_substeps', 'observed_worker_jitter', 'interval_above_8_5ms',
    'interval_below_1_5ms', 'duplicate_generation', 'deadline_5ms_boundary', 'deadline_over_5ms',
    'deadline_under_5ms', 'snapshot_expired_before_inference',
    'snapshot_deadline_after_candidate', 'generation_advance_during_inference',
    'noninference_snapshot_expiry', 'missed_first_tick',
    'missing_tick_then_continuous_hold', 'discarded_first_candidate_holds_handoff',
    'publish_returns_after_deadline', 'generation_advances_in_publish',
    'body_expires_during_publish', 'commit_crosses_deadline',
    'blocked_receipt', 'publisher_throws', 'duplicate_receipt',
    'receipt_time_in_future', 'receipt_time_before_publish', 'receipt_time_nan',
    'invalid_body', 'stale_body', 'negative_body_age_before_queue',
    'measured_position_out_of_bounds', 'body_expires_during_inference',
    'body_invalid_even_if_schedule_expired', 'stale_body_after_hold',
    'invalid_snapshot_nan', 'snapshot_from_future', 'clock_nan',
    'clock_regression_between_steps', 'r2_during_inference',
    'clock_regression_after_hold', 'r2_after_hold', 'ownership_lost_after_hold',
    'safe_mode_during_inference', 'liedown_during_inference',
    'exit_during_inference', 'generation_snapshot_pair_stress', 'posture_boundary',
]


def limits():
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))


def compile_probe(directory, overlay=None):
    executable = directory / 'cts_servo_probe'
    includes = [ROOT / 'tests/rl_stubs']
    if overlay is not None:
        includes.append(overlay)
    includes += [SDK / 'plugin', SDK / 'vendor/include/types',
                 SDK / 'vendor/include/utils', SDK / 'vendor/state_machine']
    command = ['g++', '-std=c++17', '-O0', '-pthread', '-I/usr/include/eigen3']
    command += ['-I' + str(path) for path in includes]
    command += [str(ROOT / 'tests/cts_servo_probe.cpp'), '-o', str(executable)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60,
                            preexec_fn=limits)
    assert result.returncode == 0, result.stdout + result.stderr
    return executable


def run_probe(executable, scenario):
    result = subprocess.run([str(executable), str(ROOT / 'deploy/robot.json'), scenario],
                            capture_output=True, text=True, timeout=10,
                            preexec_fn=limits)
    report = json.loads(result.stdout.strip().splitlines()[-1])
    return result, report


@pytest.fixture(scope='session')
def servo_probe(tmp_path_factory):
    return compile_probe(tmp_path_factory.mktemp('actual_rl_servo'))


@pytest.mark.parametrize('scenario', SCENARIOS)
def test_actual_rl_control_state(servo_probe, scenario):
    result, report = run_probe(servo_probe, scenario)
    assert result.returncode == 0 and report['passed'], (report, result.stderr)
    assert report['motor_commands'] == 0 and report['dds_publishers'] == 0


@pytest.mark.parametrize('mutation', ['premature_candidate_commit', 'timing_fault_becomes_hard'])
def test_servo_verifier_rejects_broken_production_copy(tmp_path, mutation):
    """Prove the oracle turns red using isolated mutated source copies."""
    overlay = tmp_path / 'include'
    overlay.mkdir()
    if mutation == 'premature_candidate_commit':
        text = (SDK / 'plugin/student_control_core.hpp').read_text()
        before = 'if(publication_commit_)candidate_pending_=true;'
        assert text.count(before) == 1
        text = text.replace(before,
                            'if(publication_commit_){last_=target;previous_=next_previous;candidate_pending_=true;}')
        (overlay / 'student_control_core.hpp').write_text(text)
        scenario = 'snapshot_deadline_after_candidate'
        expected = 'unpublished candidate advanced Previous'
    else:
        text = (SDK / 'vendor/state_machine/quadruped_wheel/rl_control_state.hpp').read_text()
        before = '}catch(const m20_cts::SchedulingFault& error){'
        assert text.count(before) == 1
        text = text.replace(before, before + 'throw;')
        (overlay / 'quadruped_wheel').mkdir()
        (overlay / 'quadruped_wheel/rl_control_state.hpp').write_text(text)
        scenario = 'deadline_5ms_boundary'
        expected = '5ms deadline'
    executable = compile_probe(tmp_path, overlay)
    result, report = run_probe(executable, scenario)
    assert result.returncode != 0 and not report['passed'], report
    assert expected in report['error'], report
