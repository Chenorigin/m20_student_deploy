"""Exercise the production C++ target/hold core without ROS or motor access."""
import json
from pathlib import Path
import resource
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def test_production_control_core(tmp_path):
    sdk = ROOT / 'src/m20_student_sdk'
    executable = tmp_path / 'control_core'

    def limits():
        resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))

    subprocess.run([
        'g++', '-std=c++17', '-O0', '-pthread', '-I/usr/include/eigen3',
        '-I' + str(sdk / 'plugin'),
        '-I' + str(sdk / 'vendor/include/types'),
        '-I' + str(sdk / 'vendor/include/utils'),
        str(ROOT / 'tests/student_control_core.cpp'), '-o', str(executable),
    ], check=True, capture_output=True, text=True, timeout=60, preexec_fn=limits)
    # This probe freezes the original rate/clip contract. The deployed MuJoCo
    # mode is separately exercised by the reference-alignment C++ probe.
    legacy = json.loads((ROOT / 'deploy/robot.json').read_text())
    legacy['policy'].update(leg_target_slew_enabled=True,
                            joint_target_position_clip_enabled=True,
                            wheel_target_velocity_clip_enabled=True)
    legacy['hardware']['software_velocity_limits_sdk_rad_s'] = [
        10.0 if i % 4 == 3 else 2.5 for i in range(16)]
    legacy['hardware']['software_feedback_position_limits_sdk_rad'] = \
        legacy['hardware']['software_position_limits_sdk_rad']
    fixture = tmp_path / 'legacy_robot.json'
    fixture.write_text(json.dumps(legacy))
    result = subprocess.run([str(executable), str(fixture)],
                            check=True, capture_output=True, text=True,
                            timeout=10, preexec_fn=limits)
    checks = json.loads(result.stdout)
    assert checks['bad'] == 0
    assert checks['ok'] == 26
