#!/usr/bin/env python3
"""Unpack a verified payload into a new board directory. Starts no applications."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--sha256',required=True)
    parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--serial',required=True)
    args=parser.parse_args()
    serial=Path('/proc/device-tree/serial-number').read_bytes().rstrip(b'\0').decode()
    if serial!=args.serial:
        raise ValueError('board serial differs from the measured target')
    if hashlib.sha256(args.archive.read_bytes()).hexdigest()!=args.sha256:
        raise ValueError('payload digest mismatch')
    destination=args.destination.resolve()
    if destination.exists() or args.destination.is_symlink():
        raise ValueError('new directory required; original deployment is preserved')
    with tarfile.open(args.archive) as archive:
        members=archive.getmembers()
        for member in members:
            path=Path(member.name)
            if path.is_absolute() or '..' in path.parts or not (member.isfile() or member.isdir()):
                raise ValueError('payload must contain only regular relative paths')
        destination.mkdir(parents=True)
        archive.extractall(destination,members=members)
    print(json.dumps({'board_serial':serial,'destination':str(destination),'payload_sha256':args.sha256,'application_started':False}))


if __name__=='__main__':main()
