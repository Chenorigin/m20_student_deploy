#!/usr/bin/env python3
"""Clear the candidate's software block only after real AOS offline checks pass."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    if Path('/opt/robot/os_type').read_text().strip()!='CA9B_AOS':
        raise ValueError('run on the actual AOS board after its build')
    result=subprocess.run([sys.executable,str(ROOT/'scripts/offline_board_models.py')],
                          capture_output=True,text=True,timeout=240,check=False)
    print(result.stdout,end='')
    if result.stderr:print(result.stderr,file=sys.stderr,end='')
    if result.returncode:raise SystemExit(result.returncode)
    tests=json.loads(result.stdout)
    if tests.get('status')!='passed':raise ValueError('offline acceptance did not pass')
    files=['models/v1/policy.onnx','models/v1/observation_contract.json','models/v1/robot_training_contract.json',
           'deploy/robot.json','runtime/heightmap_v1.local.json',
           'install/m20_student_sdk/lib/m20_student_sdk/student_control',
           'src/m20_student_sdk/plugin/student_policy_runner.hpp',
           'src/m20_student_sdk/plugin/student_heading_feedback.hpp']
    receipt={'scope':'AOS ARM CPU offline only; no physical control acceptance',
             'serial':Path('/proc/device-tree/serial-number').read_bytes().rstrip(b'\0').decode(),
             'tests':tests,'sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files},
             'hardware_commands':0}
    (ROOT/'runtime/offline_acceptance.json').write_text(json.dumps(receipt,indent=2)+'\n')
    marker=ROOT/'STAGED_NOT_ACCEPTED.txt'
    if marker.is_symlink():raise ValueError('unexpected candidate marker symlink')
    if marker.exists():marker.unlink()
    print('Offline software acceptance recorded; operator commissioning and physical validation remain separate.')


if __name__=='__main__':main()
