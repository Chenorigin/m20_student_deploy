#!/usr/bin/env python3
"""Compare operator-labelled unloaded/loaded wheel recordings, without admission."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def load_capture(path, condition):
    path = Path(path)
    manifest = json.loads((path/'manifest.json').read_text())
    data = (path/'samples.jsonl').read_bytes()
    for key in ('hardware_serial', 'boot_id'):
        if not isinstance(manifest.get(key),str) or not manifest[key].strip():
            raise ValueError('Missing capture identity: '+key)
    for key in ('robot_sha256','geometry_sha256','samples_sha256'):
        value=manifest.get(key)
        if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Invalid capture digest: '+key)
    publishers=manifest.get('publishers_before')
    if not isinstance(publishers,dict):
        raise ValueError('Missing capture publisher graph')
    for topic in ('/JOINTS_DATA','/IMU'):
        source=publishers.get(topic)
        if (not isinstance(source,list) or len(source)!=1 or not isinstance(source[0],dict)
                or not isinstance(source[0].get('gid'),list) or not source[0]['gid']
                or any(type(x) is not int or not 0<=x<=255 for x in source[0]['gid'])
                or not isinstance(source[0].get('node'),str) or not source[0]['node']):
            raise ValueError('Missing or ambiguous capture publisher identity: '+topic)
    if (hashlib.sha256(data).hexdigest() != manifest['samples_sha256']
            or manifest.get('origin') != 'hardware_recording'
            or manifest.get('condition') != condition
            or manifest.get('publisher_set_stable') is not True
            or manifest.get('limits_reached') is not False):
        raise ValueError('Capture digest, label, publisher set or limit rejected')
    leg = manifest.get('labelled_leg')
    if leg not in ('FL','FR','HL','HR'):
        raise ValueError('Capture needs an explicit wheel label')
    index = ('FL','FR','HL','HR').index(leg)
    forces, times = [], []
    for line in data.splitlines():
        row = json.loads(line)
        diagnostic = row.get('diagnostic')
        if diagnostic is None:
            continue
        force = diagnostic['estimated_force_world_N'][index]
        if force is not None and len(force)==3 and all(type(v) in (float,int) and np.isfinite(v) for v in force):
            forces.append(force[2]); times.append(row['receive_monotonic_ns']*1e-9)
    if len(forces)<30 or times[-1]-times[0]<2:
        raise ValueError('Need at least 30 finite diagnostic frames spanning two seconds')
    return manifest, np.asarray(forces)


def compare(unloaded, loaded):
    air, a = load_capture(unloaded, 'wheel_unloaded')
    ground, b = load_capture(loaded, 'wheel_loaded')
    for key in ('hardware_serial','boot_id','labelled_leg','robot_sha256','geometry_sha256','publishers_before'):
        if air.get(key) != ground.get(key):
            raise ValueError('Paired capture identity/config changed: '+key)
    qa, qb = np.quantile(a,[.05,.5,.95]), np.quantile(b,[.05,.5,.95])
    separated = bool(qb[0]>max(0.,qa[2]))
    return dict(schema='m20_contact_labelled_comparison.v1',leg=air['labelled_leg'],
                unloaded_samples=len(a),loaded_samples=len(b),
                unloaded_signed_Fz_N_p05_p50_p95=qa.tolist(),loaded_signed_Fz_N_p05_p50_p95=qb.tolist(),
                separation_rule='loaded p05 > max(0, unloaded p95); frozen before paired collection',
                diagnostic_distributions_separated=separated,
                hardware_validated=False,hardware_admission_granted=False,
                contact_label_origin='operator report; force magnitude/direction not independently calibrated',
                limitations=['Pair separation is not dynamic closed-loop validation or measured force calibration',
                             'Same-condition quantiles are descriptive; not independent generalization evidence',
                             'Other body contacts, torque bias and clock alignment still need separate evidence'],
                inputs={'unloaded_manifest_sha256':hashlib.sha256((Path(unloaded)/'manifest.json').read_bytes()).hexdigest(),
                        'loaded_manifest_sha256':hashlib.sha256((Path(loaded)/'manifest.json').read_bytes()).hexdigest()})


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unloaded',type=Path,required=True)
    parser.add_argument('--loaded',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=compare(args.unloaded,args.loaded)
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(result,indent=2))
