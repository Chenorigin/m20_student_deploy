#!/usr/bin/env python3
"""Package the new candidate only; excludes old releases, logs and build products."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('new archive path required')
    files=[]
    for name in ('src','scripts','deploy','tests','docs','runtime','models/v1','evidence/offline_models'):
        files.extend(p for p in (ROOT/name).rglob('*') if p.is_file()
                     and '__pycache__' not in str(p) and p.suffix!='.pyc')
    for name in ('STAGED_NOT_ACCEPTED.txt','README.md'):
        if (ROOT/name).is_file():files.append(ROOT/name)
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (ROOT/'PAYLOAD_SHA256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    files.append(ROOT/'PAYLOAD_SHA256.json')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(args.output,'w:gz',dereference=True) as archive:
        for path in files:archive.add(path,arcname=str(path.relative_to(ROOT)),recursive=False)
    print(json.dumps({'archive':str(args.output),'bytes':args.output.stat().st_size,
                      'sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),'files':len(files)}))


if __name__=='__main__':main()
