#!/usr/bin/env python3
"""Launch a verified stair-noise bundle through the existing CTS operator gates."""
import argparse
import importlib.util
import json
import shlex
import subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/m20_heightmap'))
from validate_cts_bundle import validate
from m20_heightmap.cts_stair_noise import validate_stair_noise_contract

DEFAULT_BUNDLE = 'logs/evaluation/cts_noise_deploy_20261104/candidate_8300/models/v1'
DEFAULT_CHECKPOINT = '88804e9c2b0f1b4c55aaeb9e90a716e69b80a4d26cf89924f88f09b40169aa60'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', nargs='?', default='start', choices=['start', 'deploy', 'status', 'check', 'stop', 'logs'])
    ap.add_argument('--training', type=Path, default=ROOT.parent / 'rl_training')
    ap.add_argument('--bundle', type=Path, help='Explicit exported stair-noise bundle; never auto-select latest')
    ap.add_argument('--aos', required=True, help='SSH alias or user@host for the connected robot')
    ap.add_argument('--gos', help='GOS SSH target reached through AOS')
    ap.add_argument('--release', default='/var/opt/robot/data/cts_noise_aligned_20261104')
    ap.add_argument('--gos-release')
    ap.add_argument('--expect-aos-serial')
    ap.add_argument('--expect-gos-serial')
    ap.add_argument('--dry-run', action='store_true', help='Validate bundle and print delegated arguments; no SSH')
    args = ap.parse_args(argv)
    if args.action in ('start', 'deploy', 'check') and not args.gos:
        ap.error('--gos is required for this action')
    entry = args.training.resolve() / 'scripts/tools/m20_robot.py'
    if not entry.is_file():
        ap.error('Existing CTS operator entry missing: ' + str(entry))
    command = [args.action, '--aos', args.aos, '--release', args.release]
    for name in ('gos', 'gos_release', 'expect_aos_serial', 'expect_gos_serial'):
        value = getattr(args, name)
        if value:
            command += ['--' + name.replace('_', '-'), value]
    if args.action in ('start', 'deploy'):
        bundle = (args.bundle or args.training / DEFAULT_BUNDLE).resolve()
        result = validate(bundle, check_model=False)
        validate_stair_noise_contract(json.loads((bundle / 'observation_contract.json').read_text()))
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
    original_deploy = operator.deploy
    def aligned_deploy(options, aos, gos, bundle):
        for board in (aos, gos):
            board.ssh('cd ' + shlex.quote(str(board.root)) +
                      ' && python3 scripts/check_cts_runtime_alignment.py')
        return original_deploy(options, aos, gos, bundle)
    operator.deploy = aligned_deploy
    return operator.main(command)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print('降音策略操作未完成：' + str(error), file=sys.stderr)
        sys.exit(1)
    except (KeyboardInterrupt, EOFError):
        print('已取消；请通过 status/stop 核实板端状态。', file=sys.stderr)
        sys.exit(130)
