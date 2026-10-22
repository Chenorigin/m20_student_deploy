#!/usr/bin/env python3
"""Root read-only process/socket identity; no DDS service call or publisher."""
import argparse
import json
import re
import subprocess
from pathlib import Path

def command(*args):
    result = subprocess.run(args, text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(f"{' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout

def check_port(ss_output, basic_cgroup, proc_root=Path("/proc"),
               expected_exe=Path("/opt/robot/share/basic_server/bin/basic_server")):
    sockets = [line for line in ss_output.splitlines() if line.strip()]
    if len(sockets) != 1 or not basic_cgroup.startswith("/system.slice/"):
        raise ValueError("UDP/30000 socket or service cgroup is missing or ambiguous")
    owners = re.findall(r'"([^\"]+)",pid=(\d+)', sockets[0])
    if len(owners) != 1 or "users:" not in sockets[0]:
        raise ValueError("UDP/30000 owner could not be identified")
    name, pid = owners[0]
    if name != "basic_server":
        raise ValueError("UDP/30000 has an unknown owner")
    process = proc_root / pid
    cgroups = (process / "cgroup").read_text().splitlines()
    if not any(line.split(":", 2)[-1] == basic_cgroup for line in cgroups):
        raise ValueError("UDP/30000 owner is outside basic_server.service")
    if (process / "exe").resolve(strict=True) != expected_exe.resolve(strict=True):
        raise ValueError("UDP/30000 owner executable is not the vendor basic_server")


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pid-only',action='store_true');args=ap.parse_args()
    for unit in ('basic_server.service','rl_deploy.service'):
        if subprocess.run(('systemctl','is-active','--quiet',unit),check=False).returncode:
            raise RuntimeError('Required vendor service not active: '+unit)
    basic_group=command('systemctl','show','--property','ControlGroup','--value','basic_server.service').strip()
    check_port(command('ss','-H','-lunp','( sport = :30000 )'),basic_group)
    group=command('systemctl','show','--property','ControlGroup','--value','rl_deploy.service').strip()
    if not group.startswith('/system.slice/'):
        raise RuntimeError('Vendor controller cgroup is not verified')
    workers=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            if (p/'comm').read_text().strip()!='rl_deploy':continue
            if (p/'exe').resolve(strict=True)!=Path('/opt/robot/share/rl_deploy/bin/rl_deploy').resolve(strict=True):
                raise RuntimeError('Unknown rl_deploy executable')
            if not any(line.split(':',2)[-1]==group for line in (p/'cgroup').read_text().splitlines()):
                raise RuntimeError('Vendor worker not in expected service')
            workers.append(int(p.name))
        except FileNotFoundError:continue
    if len(workers)!=1 or not 0<workers[0]<65536:
        raise RuntimeError('Vendor worker PID identity is missing or ambiguous')
    print(workers[0] if args.pid_only else json.dumps({'passed':True,'vendor_worker_pid':workers[0],'commands_sent':0}))

if __name__=='__main__':main()
