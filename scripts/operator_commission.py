#!/usr/bin/env python3
"""Explicit physical-check receipt. Run interactively on each board, never automatically."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--profile',choices=['v0','v1'],default='v1');ap.add_argument('--prepare-preview',action='store_true');args=ap.parse_args()
    if not sys.stdin.isatty():raise ValueError('operator interactive terminal required')
    runtime=ROOT/'runtime';runtime.mkdir(exist_ok=True)
    path=runtime/('heightmap_'+args.profile+'.local.json')
    template=path if path.exists() else ROOT/'deploy'/('heightmap_'+args.profile+'.json')
    if args.prepare_preview:
        print('Prepare factory-mounting candidate for perception preview only; SDK control remains blocked without a physical-check receipt.')
        if input('Type PREPARE_STATIC_PREVIEW: ')!='PREPARE_STATIC_PREVIEW':raise SystemExit(2)
        cfg=json.loads(template.read_text());cfg['calibration_confirmed']=True;path.write_text(json.dumps(cfg,indent=2)+'\n');return
    print('Check: factory sensor mounting, fused lidar_link matches training base, IMU axes, 187-cell static height signs/coverage, conservative self filtering, software joint/speed limits, stable support, physical E-stop, and manufacturer SDK control ownership.')
    print('This records your checks; it does not measure or prove them. It starts no software and sends no command.')
    if input('Only after performing those checks, type CONFIRM_PHYSICAL_CHECKS: ')!='CONFIRM_PHYSICAL_CHECKS':raise SystemExit(2)
    cfg=json.loads(template.read_text());cfg['calibration_confirmed']=True;path.write_text(json.dumps(cfg,indent=2)+'\n')
    receipt={'profile':args.profile,'factory_geometry_checked':True,'software_limits_checked':True,'operator_controls_start':True,'physical_estop_available':True,'wall_time':time.time(),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'config_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'robot_sha256':hashlib.sha256((ROOT/'deploy/robot.json').read_bytes()).hexdigest(),'site_sha256':hashlib.sha256((ROOT/'deploy/site.json').read_bytes()).hexdigest(),'lio_sha256':hashlib.sha256((ROOT/'deploy/gos_lio.yaml').read_bytes()).hexdigest()}
    (runtime/'commissioned.json').write_text(json.dumps(receipt,indent=2))
    print('Local commissioning recorded. Repeat on the other board with the same profile/config; applications remain stopped.')

if __name__=='__main__':main()
