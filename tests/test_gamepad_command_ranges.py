import json
import os
from pathlib import Path
import resource
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def limits():
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))


def test_actual_gamepad_matches_mujoco_ranges(tmp_path):
    sdk = ROOT / 'src/m20_student_sdk'
    include = ['plugin', 'vendor/include/types', 'vendor/include/utils',
               'vendor/interface/user_command', 'vendor/interface/network']
    exe = tmp_path / 'gamepad_ranges'
    command = ['g++', '-std=c++17', '-O0', '-pthread', '-I/usr/include/eigen3']
    command += ['-I' + str(sdk / p) for p in include]
    command += [str(ROOT / 'tests/gamepad_command_ranges.cpp'), '-o', str(exe)]
    subprocess.run(command, capture_output=True, text=True, timeout=60,
                   check=True, preexec_fn=limits)
    env = dict(os.environ, M20_ROBOT_CONFIG=str(ROOT / 'deploy/robot.json'))
    env.pop('M20_FORWARD_ONLY', None)
    result = subprocess.run([str(exe)], env=env, capture_output=True, text=True,
                            timeout=10, preexec_fn=limits)
    assert result.returncode == 0, result.stdout + result.stderr
    report = next(json.loads(line) for line in result.stdout.splitlines()
                  if line.startswith('{') and '"checks"' in line)
    assert report['passed'] and report['checks'] == 36 and report['motor_commands'] == 0
    # The same real parser must reject the previous wider range as a match.
    old = json.loads((ROOT / 'deploy/robot.json').read_text())
    old['command_operator_ranges'] = {'lin_vel_x': [-1.2, 1.2],
                                      'lin_vel_y': [-.7, .7], 'ang_vel_z': [-1., 1.]}
    fixture = tmp_path / 'old_robot.json'
    fixture.write_text(json.dumps(old))
    env['M20_ROBOT_CONFIG'] = str(fixture)
    result = subprocess.run([str(exe)], env=env, capture_output=True, text=True,
                            timeout=10, preexec_fn=limits)
    assert result.returncode != 0 and 'differs from MuJoCo' in result.stderr
