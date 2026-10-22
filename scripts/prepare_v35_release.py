#!/usr/bin/env python3
"""Create a new deployment copy for a later v3.5 checkpoint; never edit a release."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--env-yaml', type=Path)
    args = parser.parse_args()
    checkpoint, destination = args.checkpoint.resolve(), args.output.resolve()
    if destination.exists():
        raise ValueError('output must not exist; each checkpoint gets a new release')
    destination.mkdir(parents=True)
    for name in ('src','scripts','deploy','tests','tests_ros','docs'):
        shutil.copytree(ROOT/name, destination/name,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(ROOT/'README.md',destination/'README.md')
    runtime = destination/'runtime'
    runtime.mkdir()
    for name in ('heightmap_v1.local.json','dds_udp.xml','native_input.json'):
        source = ROOT/'runtime'/name
        if source.is_file():
            shutil.copy2(source, runtime/name)
    (destination/'evidence').mkdir()
    (destination/'STAGED_NOT_ACCEPTED.txt').write_text(
        'New student-v3.5 checkpoint. Build and offline numerical checks on the target AOS are required. '
        'No commissioning receipt or operator approval is inherited.\n')
    model = destination/'models/v1'
    command = [sys.executable, str(destination/'scripts/export_checkpoint.py'),
               '--checkpoint', str(checkpoint), '--profile', 'v1', '--output', str(model)]
    if args.env_yaml:
        command += ['--env-yaml', str(args.env_yaml.resolve())]
    subprocess.run(command, check=True)
    subprocess.run([sys.executable, str(destination/'scripts/export_onnx.py'),
                    '--model', str(model/'policy.pt'), '--contract', str(model/'observation_contract.json'),
                    '--profile','v1','--output',str(model/'policy.onnx')], check=True)
    subprocess.run([sys.executable, str(destination/'scripts/validate_v35_bundle.py')], check=True)
    historical = ROOT/'evidence/offline_models/historical_v3/input.f32'
    if not historical.is_file():
        raise ValueError('fixed historical numerical corpus missing from base release')
    shutil.copy2(historical,destination/'evidence/historical_observations.f32')
    subprocess.run([sys.executable, str(destination/'scripts/make_offline_fixtures.py')], check=True)
    print('Prepared '+str(destination)+'; original release unchanged; board checks and commissioning pending.')


if __name__ == '__main__':
    main()
