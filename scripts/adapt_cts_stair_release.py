#!/usr/bin/env python3
"""Offline board adapter. Default inspection only; never starts/stops a process."""
import argparse
import fcntl
from contextlib import ExitStack
import hashlib
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
STATE_HOME = Path('/run/m20_one_command')
PROC_ROOT = Path('/proc')
ACTIVE_RELEASE = Path('/home/user/m20_heightmap/current')
OLD = {'cts.py': '811d43922c12f7b5465116a6729920ea571fd6ce42537e51ebb47376fe03c351',
       'validate_cts_bundle.py': '842c027b72366879794b098e6bdc8ffe64dfb75ae8ad09b1d0468064903ee88c'}

def plan(release):
    packages = [release / 'src/m20_heightmap/m20_heightmap']
    packages += list((release / 'install/m20_heightmap/lib').glob('python*/site-packages/m20_heightmap'))
    if len(packages) < 2:
        raise ValueError('Expected built release with both source and installed Python package')
    copies = [(ROOT / 'scripts/validate_cts_bundle.py', release / 'scripts/validate_cts_bundle.py')]
    for package in packages:
        for name in ('cts.py', 'cts_stair_finetune.py'):
            copies.append((ROOT / 'src/m20_heightmap/m20_heightmap' / name, package / name))
    if release == ACTIVE_RELEASE.resolve() or ACTIVE_RELEASE.resolve() in release.parents:
        raise ValueError("Adapt only an isolated release, never current")
    if not (release / "STAGED_NOT_ACCEPTED.txt").is_file():
        raise ValueError("Isolated release must remain STAGED_NOT_ACCEPTED during adaptation")
    result = {}
    for source, target in copies:
        target = target.resolve()
        if release not in target.parents:
            raise ValueError("Runtime symlink escapes isolated release: " + str(target))
        value = source.read_bytes()
        if target.exists():
            previous = target.read_bytes()
            if previous == value:
                continue
            if hashlib.sha256(previous).hexdigest() != OLD.get(target.name):
                raise ValueError('Unknown/newer target content; refusing overwrite: ' + str(target))
        elif target.name != 'cts_stair_finetune.py':
            raise ValueError('Required runtime file missing: ' + str(target))
        result[target] = value
    return result

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', required=True, type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args(argv)
    changes = plan(args.release.resolve())
    for target in changes:
        print(('APPLY ' if args.apply else 'WOULD UPDATE ') + str(target))
    if not args.apply:
        return
    # Match the existing one-command operator lock order. Never stop its owner.
    with ExitStack() as stack:
        state = STATE_HOME
        state.mkdir(mode=0o755, exist_ok=True)
        for name in ('sdk.lock', 'metadata.lock'):
            handle = stack.enter_context((state / name).open('a+'))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (state / 'deploy.json').exists() or (args.release / 'runtime/one_command_deploy.json').exists():
            raise RuntimeError('Pending deployment; refusing runtime adaptation')
        for proc in PROC_ROOT.glob('[0-9]*/cmdline'):
            try:
                command = proc.read_bytes().split(b'\0')
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            if any(Path(arg.decode(errors='replace')).name in
                   ('student_control', 'gos_mapper', 'start_student.py', 'start_student.sh') for arg in command):
                raise RuntimeError('Controller/mapper/startup is active')
        changes = plan(args.release.resolve())  # Recheck after owning both locks.
        previous = {target: target.read_bytes() if target.exists() else None for target in changes}
        for target in changes:
            backup = target.with_name(target.name + '.before_stair_v2')
            if target.exists() and backup.exists() and backup.read_bytes() != previous[target]:
                raise ValueError('Existing backup differs: ' + str(backup))
            if target.with_name(target.name + '.stair_v2.tmp').exists():
                raise ValueError('Unfinished adapter temporary file: ' + str(target))
        applied = []
        temporary = []
        # Dependencies first; caller code is replaced only after all files stage.
        ordered = sorted(changes, key=lambda p: p.name != 'cts_stair_finetune.py')
        try:
            for target in ordered:
                backup = target.with_name(target.name + '.before_stair_v2')
                if previous[target] is not None and not backup.exists():
                    shutil.copy2(target, backup)
                temp = target.with_name(target.name + '.stair_v2.tmp')
                with temp.open('xb') as stream:
                    temporary.append(temp)
                    stream.write(changes[target])
            for target in ordered:
                target.with_name(target.name + '.stair_v2.tmp').replace(target)
                applied.append(target)
        except BaseException:
            for target in reversed(applied):
                if previous[target] is None:
                    target.unlink()
                else:
                    shutil.copy2(target.with_name(target.name + '.before_stair_v2'), target)
            raise
        finally:
            for temp in temporary:
                if temp.exists():
                    temp.unlink()
    print('Adaptation complete. No model switch or process start performed.')

if __name__ == '__main__':
    main()
