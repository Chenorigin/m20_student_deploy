#!/usr/bin/env python3
"""Check CTS config, real capture geometry and built consumers without hardware I/O."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/m20_heightmap'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_files(root, role):
    files = [p for p in (root/'src').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc']
    installed=list((root/'install/m20_heightmap/lib').glob('python*/site-packages/m20_heightmap/*.py'))
    for p in installed:
        if sha(p)!=sha(root/'src/m20_heightmap/m20_heightmap'/p.name):
            raise ValueError('Installed Python differs from source: '+p.name)
    files += installed
    native=root/'install/m20_native_inputs/lib/m20_native_inputs/native_lidar_input'
    if not native.is_file():raise ValueError('Built native lidar consumer missing')
    files.append(native)
    if not any('site-packages' in str(p) for p in files):
        raise ValueError('Built installed Python package missing')
    if role == 'AOS':
        for name in ('student_control','student_onnx_probe','cts_live_replay'):
            p=root/'install/m20_student_sdk/lib/m20_student_sdk'/name
            if not p.is_file():raise ValueError('Built SDK binary missing: '+name)
            files.append(p)
    return {str(p.relative_to(root)):sha(p) for p in sorted(files)}


def check_config(root):
    from m20_heightmap.runtime import load_config
    from m20_heightmap.cts import validate_cts_contract, validate_mapper
    from m20_heightmap.capture import capture_geometry, capture_memory_budget
    from m20_heightmap.estimator import ElevationMapper, EstimatorConfig
    from m20_heightmap.contract import GRID
    cfg=load_config(root/'runtime/heightmap_v1.local.json')
    for key,name in [('model_contract','observation_contract.json'),('model_path','policy.onnx')]:
        if Path(cfg['aos'][key]).resolve()!=(root/'models/v1'/name).resolve():
            raise ValueError('CTS selected model path mismatch: '+key)
    actual=json.loads((root/'models/v1/observation_contract.json').read_text())
    validate_cts_contract(actual);validate_mapper(cfg,actual)
    expected=json.loads((root/'deploy/cts_runtime_baseline.json').read_text())
    robot=json.loads((root/'deploy/robot.json').read_text())
    for group,values in expected['robot'].items():
        if robot.get(group)!=values:raise ValueError('CTS baseline mismatch: robot.'+group)
    for group,values in expected['runtime'].items():
        for name,value in values.items():
            if cfg[group].get(name)!=value:raise ValueError('CTS baseline mismatch: '+group+'.'+name)
    p=actual['perception']
    for config_key,model_key in [('min_range_m','lidar_min_range_m'),('max_range_m','lidar_max_range_m'),('map_capture_padding_m','map_capture_padding_m')]:
        if cfg['estimator'].get(config_key)!=p.get(model_key):
            raise ValueError('CTS perception mismatch: '+config_key)
    mapper=ElevationMapper(cfg['profile'],EstimatorConfig(**cfg['estimator']))
    _,shape,_=capture_geometry(p['map_capture_padding_m'])
    if mapper.capture_shape!=shape or len(GRID)!=187:raise ValueError('Actual capture/output grid mismatch')
    return dict(command_operator_ranges=robot['command_operator_ranges'],hold_enabled=robot['policy']['hold_enabled'],
        terrain_source_session_hold_enabled=robot['policy']['terrain_source_session_hold_enabled'],
        mapping_region=cfg['runtime']['mapping_region'],lidar_range_m=[cfg['estimator']['min_range_m'],cfg['estimator']['max_range_m']],
        capture=capture_memory_budget(cfg['estimator']),output_points=len(GRID),execution_mode=cfg['aos']['cts_execution_mode'])


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=ROOT)
    ap.add_argument('--record-build',action='store_true',help='Only after build_board.sh successfully built this tree')
    ap.add_argument('--config-only',action='store_true')
    args=ap.parse_args(argv);root=args.root.resolve()
    contract=json.loads((root/'models/v1/observation_contract.json').read_text())
    if contract.get('deployment_family') != 'm20_cts_student':
        print(json.dumps({'applicable':False,'reason':'non-CTS model; existing legacy gates apply'}))
        return
    report=check_config(root)
    role=Path('/opt/robot/os_type').read_text().strip().split('_')[-1] if not args.config_only else None
    if role is not None:
        if role not in ('AOS','GOS'):raise ValueError('Unknown board role')
        path=root/'runtime/cts_build_identity.json'
        actual={'role':role,'files':build_files(root,role),'baseline_sha256':sha(root/'deploy/cts_runtime_baseline.json')}
        if args.record_build:
            path.write_text(json.dumps(actual,indent=2)+'\n')
        elif not path.is_file() or json.loads(path.read_text())!=actual:
            raise ValueError('CTS build identity missing/stale: rebuild matching source and installed consumers')
        report['build_identity_sha256']=sha(path)
    report.update(passed=True,scope='config, capture construction and build identity; no motor or live timing test',hardware_commands=0)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
