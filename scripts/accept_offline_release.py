#!/usr/bin/env python3
"""Clear the candidate's software block only after real AOS offline checks pass."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    from check_execution_runtime import require_hardware_runtime
    contract=json.loads((ROOT/'models/v1/observation_contract.json').read_text())
    cfg=json.loads((ROOT/'runtime/heightmap_v1.local.json').read_text())
    mode=require_hardware_runtime(contract,cfg)
    if Path('/opt/robot/os_type').read_text().strip()!='CA9B_AOS':
        raise ValueError('run on the actual AOS board after its build')
    cts=contract.get('deployment_family')=='m20_cts_student'
    command=([sys.executable,str(ROOT/'scripts/offline_cts.py'),'--model-dir',str(ROOT/'models/v1'),
              '--probe',str(ROOT/'install/m20_student_sdk/lib/m20_student_sdk/student_onnx_probe')]
             if cts else [sys.executable,str(ROOT/'scripts/offline_board_models.py')])
    result=subprocess.run(command,
                          capture_output=True,text=True,timeout=240,check=False)
    print(result.stdout,end='')
    if result.stderr:print(result.stderr,file=sys.stderr,end='')
    if result.returncode:raise SystemExit(result.returncode)
    tests=json.loads(result.stdout)
    if tests.get('status')!='passed':raise ValueError('offline acceptance did not pass')
    if cts:
        env=dict(os.environ,M20_HEIGHTMAP_CONFIG=str(ROOT/'runtime/heightmap_v1.local.json'),
                 M20_POLICY_PATH=str(ROOT/'models/v1/policy.onnx'),
                 M20_VERIFIED_MODEL_CONTRACT_SHA256=hashlib.sha256((ROOT/'models/v1/observation_contract.json').read_bytes()).hexdigest())
        guard=subprocess.run([str(ROOT/'install/m20_student_sdk/lib/m20_student_sdk/student_onnx_probe'),'--runtime-check'],
                             env=env,capture_output=True,text=True,check=True,timeout=30)
        tests['runtime_check']=json.loads(guard.stdout.strip().splitlines()[-1])
    files=['models/v1/policy.onnx','models/v1/observation_contract.json',
           'deploy/robot.json','runtime/heightmap_v1.local.json',
           'install/m20_student_sdk/lib/m20_student_sdk/student_control',
           'src/m20_student_sdk/plugin/student_policy_runner.hpp',
           'src/m20_student_sdk/plugin/student_heading_feedback.hpp']
    files+=(['src/m20_student_sdk/plugin/cts_execution_mode.hpp',
             'src/m20_student_sdk/plugin/student_model_io.hpp'] if cts else ['models/v1/robot_training_contract.json'])
    receipt={'scope':'AOS ARM CPU offline only; no physical control acceptance',
             'serial':Path('/proc/device-tree/serial-number').read_bytes().rstrip(b'\0').decode(),
             'tests':tests,'cts_execution_mode':mode,'landing_control_enabled':mode=='landing',
             'sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files},
             'hardware_commands':0}
    (ROOT/'runtime/offline_acceptance.json').write_text(json.dumps(receipt,indent=2)+'\n')
    marker=ROOT/'STAGED_NOT_ACCEPTED.txt'
    if marker.is_symlink():raise ValueError('unexpected candidate marker symlink')
    if marker.exists():marker.unlink()
    print('Offline software acceptance recorded; operator commissioning and physical validation remain separate.')


if __name__=='__main__':main()
