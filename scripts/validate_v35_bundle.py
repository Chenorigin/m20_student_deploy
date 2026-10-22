#!/usr/bin/env python3
"""Validate immutable student-v3.5 model/config pairing before SDK initialization."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/m20_heightmap'))
from m20_heightmap.contract import contract, validate_model_contract


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(root, check_metadata=False):
    model = root/'models/v1'
    deployment = json.loads((model/'deployment.json').read_text())
    exported = json.loads((model/'policy.onnx.json').read_text())
    actual = validate_model_contract(model/'observation_contract.json', contract('v1', 4))
    if ("quiet_execution_control" in actual or "quiet_execution_control" in deployment
            or "quiet_execution_control" in deployment.get("training_provenance", {})):
        raise ValueError("StairQuiet execution controller is not implemented by this SDK bundle")
    if actual.get('deployment_family') != 'm20_student_v3_5':
        raise ValueError('not a student-v3.5 model bundle')
    feedback = actual.get('command_feedback', {})
    if feedback.get('version') != 'legacy_v1_v3_5' or feedback != deployment.get('command_feedback'):
        raise ValueError('model/feedback semantics disagree')
    for record in (deployment, exported):
        if record['contract_sha256'] != sha(model/'observation_contract.json'):
            raise ValueError('contract digest mismatch')
    if exported['policy_sha256'] != sha(model/'policy.onnx'):
        raise ValueError('ONNX digest mismatch')
    if exported['torchscript_sha256'] != deployment['policy_sha256']:
        raise ValueError('ONNX/TorchScript provenance mismatch')
    training_path = model/'robot_training_contract.json'
    for record in (deployment, exported):
        if record.get('robot_training_contract_sha256') != sha(training_path):
            raise ValueError('training parameter digest mismatch')
    expected = json.loads(training_path.read_text())
    schema = {'policy': {'dt_s','normalized_action_clip','processed_target_clip_policy'},
              'joints': {'policy_order','sdk_order','default_position_policy_rad',
                         'action_scale_policy','kp_sdk','kd_sdk'},
              'command_training_ranges': {'lin_vel_x','lin_vel_y','ang_vel_z'}}
    if set(expected) != set(schema) or any(set(expected[key]) != fields for key,fields in schema.items()):
        raise ValueError('incomplete training parameter schema')
    if check_metadata:
        probe = root/'install/m20_student_sdk/lib/m20_student_sdk/student_onnx_probe'
        result = subprocess.run([str(probe),str(model/'policy.onnx')],
                                capture_output=True,text=True,timeout=30,check=True)
        embedded = json.loads(result.stdout)
        if (embedded.get('deployment_family') != actual['deployment_family'] or
            embedded.get('contract_sha256') != sha(model/'observation_contract.json') or
            json.loads(embedded.get('command_feedback','null')) != feedback):
            raise ValueError('ONNX embedded metadata mismatch')
    robot = json.loads((root/'deploy/robot.json').read_text())
    for group, fields in expected.items():
        for key, value in fields.items():
            if robot[group][key] != value:
                raise ValueError('robot/training mismatch: '+group+'.'+key)
    return {'passed': True, 'deployment_family': actual['deployment_family'],
            'checkpoint_sha256': deployment['checkpoint_sha256'],
            'onnx_sha256': exported['policy_sha256'], 'input_shape': [1, 789],
            'feedback_version': feedback['version'], 'hardware_commands': 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--model-metadata',action='store_true',help='AOS: read metadata using the actual ARM ORT probe')
    args = parser.parse_args()
    print(json.dumps(validate(args.root.resolve(),args.model_metadata), indent=2))


if __name__ == '__main__':
    main()
