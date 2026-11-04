#!/usr/bin/env python3
"""Launch a verified stair-finetune bundle through the existing CTS operator gates."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/m20_heightmap'))
from validate_cts_bundle import validate
from m20_heightmap.cts_stair_finetune import validate_stair_finetune_contract

DEFAULT_BUNDLE = 'logs/evaluation/cts_stair_v2_1515_deploy_20261104/candidate/models/v1'
DEFAULT_CHECKPOINT = '4482253200d4157092f4071e5d1b242d4652ff33aa38297eae2ceff6c7230e07'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', nargs='?', default='start', choices=['start', 'deploy', 'status', 'check', 'stop', 'logs'])
    ap.add_argument('--training', type=Path, default=ROOT.parent / 'rl_training')
    ap.add_argument('--bundle', type=Path, help='Explicit exported stair-finetune bundle; never auto-select latest')
    ap.add_argument('--aos', required=True, help='SSH alias or user@host for the connected robot')
    ap.add_argument('--gos', help='GOS SSH target reached through AOS')
    ap.add_argument('--release', help='Explicit adapted built AOS release required for start/deploy')
    ap.add_argument('--gos-release')
    ap.add_argument('--expect-aos-serial', default='4c7428ea7d51f4b')
    ap.add_argument('--expect-gos-serial', default='6a3df90cc64ca70b')
    ap.add_argument('--dry-run', action='store_true', help='Validate bundle and print delegated arguments; no SSH')
    args = ap.parse_args(argv)
    if args.action in ('start', 'deploy', 'check') and not args.gos:
        ap.error('--gos is required for this action')
    if args.action in ('start', 'deploy') and not args.release:
        ap.error('--release must select the adapted isolated built release')
    entry = args.training.resolve() / 'scripts/tools/m20_robot.py'
    if not entry.is_file():
        ap.error('Existing CTS operator entry missing: ' + str(entry))
    command = [args.action, '--aos', args.aos, '--release', args.release or '/home/user/m20_heightmap/current']
    for name in ('gos', 'gos_release', 'expect_aos_serial', 'expect_gos_serial'):
        value = getattr(args, name)
        if value:
            command += ['--' + name.replace('_', '-'), value]
    if args.action in ('start', 'deploy'):
        bundle = (args.bundle or args.training / DEFAULT_BUNDLE).resolve()
        result = validate(bundle, check_model=False)
        validate_stair_finetune_contract(json.loads((bundle / 'observation_contract.json').read_text()))
        if args.bundle is None and result['checkpoint_sha256'] != DEFAULT_CHECKPOINT:
            raise ValueError('Default validated checkpoint changed; select a new bundle explicitly')
        command += ['--bundle', str(bundle)]
    elif args.bundle:
        ap.error('stop/status/check/logs do not accept model changes')
    spec = importlib.util.spec_from_file_location('m20_cts_operator', entry)
    operator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(operator)
    operator.parser().parse_args(command)  # Validate SSH target syntax even in dry-run.
    if args.dry_run:
        print(json.dumps({'entry': str(entry), 'argv': command, 'ssh_connections': 0, 'hardware_commands': 0}, indent=2))
        return 0
    return operator.main(command)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, OSError) as error:
        print('楼梯微调策略操作未完成：' + str(error), file=sys.stderr)
        sys.exit(1)
    except (KeyboardInterrupt, EOFError):
        print('已取消；请通过 status/stop 核实板端状态。', file=sys.stderr)
        sys.exit(130)
