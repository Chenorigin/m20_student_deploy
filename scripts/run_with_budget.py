#!/usr/bin/env python3
"""Run one offline command with a per-process address-space ceiling and receipt."""
import argparse
import json
from pathlib import Path
import resource
import subprocess
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--limit-mb',type=int,required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command or args.limit_mb<=0:raise ValueError('command and positive limit required')
    available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))//1024
    if available<args.limit_mb:raise ValueError('insufficient MemAvailable for requested offline budget')
    resource.setrlimit(resource.RLIMIT_AS,(args.limit_mb*1024**2,)*2)
    start=time.monotonic()
    result=subprocess.run(command,check=False)
    stats=resource.getrusage(resource.RUSAGE_CHILDREN)
    receipt={'command':command,'returncode':result.returncode,'elapsed_s':time.monotonic()-start,
             'max_child_rss_kb':stats.ru_maxrss,'address_space_limit_mb':args.limit_mb,
             'mem_available_before_mb':available}
    args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))
    raise SystemExit(result.returncode)


if __name__=='__main__':main()
