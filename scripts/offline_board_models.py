#!/usr/bin/env python3
"""Operator-independent offline validation: never loads ROS or an SDK interface."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def main():
    actual=json.loads((ROOT/'models/v1/observation_contract.json').read_text())
    if actual.get('deployment_family')=='m20_cts_student':
        subprocess.run([sys.executable,str(ROOT/'scripts/offline_cts.py'),
                        '--probe',str(ROOT/'install/m20_student_sdk/lib/m20_student_sdk/student_onnx_probe')],check=True)
        return
    probe=ROOT/'install/m20_student_sdk/lib/m20_student_sdk/student_onnx_probe'
    if not probe.is_file():raise RuntimeError('Build m20_student_sdk on AOS first')
    from validate_v35_bundle import validate
    validate(ROOT,check_metadata=True)
    reports=[]
    passed=True
    for profile in ('v1','historical_v3'):
        fixtures=ROOT/'evidence/offline_models'/profile
        with tempfile.TemporaryDirectory(prefix='m20_offline_') as tmp:
            out=Path(tmp)/'actions.f32'
            frames=(fixtures/'input.f32').stat().st_size//(789*4)
            timeout=max(30.,10.+frames*.02)  # one policy period per fixed frame plus startup budget
            p=subprocess.run([str(probe),str(ROOT/'models/v1/policy.onnx'),str(fixtures/'input.f32'),str(out)],capture_output=True,text=True,timeout=timeout,check=True)
            expected=np.fromfile(str(fixtures/'expected.f32'),dtype='<f4').reshape(-1,16)
            actual=np.fromfile(str(out),dtype='<f4').reshape(-1,16)
            if actual.shape!=expected.shape:raise ValueError('offline frame/action shape mismatch')
            error=np.abs(actual-expected)
            okay=bool(np.isfinite(actual).all() and np.all(error<=2e-5+2e-5*np.abs(expected)))
            passed &= okay
            reports.append({'profile':profile,'passed':okay,'frames':len(actual),'mismatched_elements':int(np.count_nonzero(error>2e-5+2e-5*np.abs(expected))),'max_absolute_error':float(np.max(np.abs(actual-expected))),'probe':json.loads(p.stdout),'hardware_commands':0})
    request=ROOT/'evidence/offline_models/heading_input.json'
    expected=json.loads((ROOT/'evidence/offline_models/heading_expected.json').read_text())
    result=subprocess.run([str(ROOT/'install/m20_student_sdk/lib/m20_student_sdk/student_heading_probe')],
                          input=request.read_text(),capture_output=True,text=True,timeout=30,check=True)
    actual=json.loads(result.stdout)
    def compare(a,b):
        if isinstance(b,dict):
            if set(a)!=set(b):raise ValueError('heading result keys mismatch')
            for key in b:compare(a[key],b[key])
        elif isinstance(b,list):
            if len(a)!=len(b):raise ValueError('heading result length mismatch')
            for av,bv in zip(a,b):compare(av,bv)
        elif isinstance(b,(float,int)) and not isinstance(b,bool):
            np.testing.assert_allclose(a,b,atol=2e-6,rtol=0)
        elif a!=b:raise ValueError('heading result mismatch')
    compare(actual,expected)
    reports.append({'profile':'v3.5_heading','scenarios':len(expected),'hardware_commands':0})
    print(json.dumps({'status':'passed' if passed else 'failed','scope':'AOS CPU ONNX offline; not policy control startup','models':reports},indent=2))
    if not passed:raise SystemExit(2)

if __name__=='__main__':main()
