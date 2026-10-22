#!/usr/bin/env python3
"""Check the selected node and its local libraries before starting perception."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


def verify(release, site):
    release = Path(release).resolve(strict=True)
    expected = json.loads(Path(site).read_text())
    if not isinstance(expected, dict):
        raise ValueError('LIO site configuration must be an object')
    checked = {}
    for relative, key in (
        ('bin/super_lio_node', 'lio_binary_sha256'),
        ('lib/liblio.so', 'lio_library_sha256'),
        ('lib/libbasic.so', 'lio_basic_library_sha256'),
    ):
        value = expected.get(key)
        if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
            raise ValueError('Missing or invalid LIO identity: ' + key)
        path = release / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != value:
            raise ValueError('LIO identity differs: ' + relative)
        checked[relative] = actual
    return {'release': str(release), 'checked_sha256': checked}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('release', type=Path)
    parser.add_argument('site', type=Path)
    args = parser.parse_args()
    try:
        result = verify(args.release, args.site)
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
