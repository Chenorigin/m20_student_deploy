#!/usr/bin/env python3
"""CPU-only shape/ABI probe for a user-selected exported model. No robot interfaces."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src/m20_heightmap"))
from m20_heightmap.contract import contract,validate_model_contract


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",type=Path,required=True)
    ap.add_argument("--contract",type=Path,required=True)
    ap.add_argument("--profile",choices=("v0","v1"),required=True)
    ap.add_argument("--history-steps",type=int)
    ap.add_argument("--observation",type=Path,help="Optional .npy containing one exact model input vector")
    ap.add_argument("--raw-command",type=float,nargs=3,help="CTS: separately recorded requested command, before heading feedback")
    args=ap.parse_args()
    spec=contract(args.profile,args.history_steps);actual=validate_model_contract(args.contract,spec)
    x=np.zeros((1,spec["input_dim"]),dtype=np.float32)
    x[:,5]=-1.
    if args.observation:
        x=np.load(args.observation,allow_pickle=False).astype(np.float32).reshape(1,-1)
    if x.shape!=(1,spec["input_dim"]) or not np.isfinite(x).all():raise ValueError("Invalid observation")
    cts=actual.get('deployment_family')=='m20_cts_student'
    if cts:
        if args.raw_command is None:raise ValueError('CTS requires --raw-command; do not infer it from observation[6:9]')
        raw=np.asarray([args.raw_command],dtype=np.float32)
        if not np.isfinite(raw).all():raise ValueError('Nonfinite raw command')
    if cts and args.model.suffix.lower()=='.onnx':
        from m20_heightmap.cts import CTSInference
        y=CTSInference(args.model,args.contract)(x,raw)
    elif args.model.suffix.lower()==".onnx":
        import onnxruntime as ort
        options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
        options.add_session_config_entry("session.disable_prepacking", "1")
        session=ort.InferenceSession(str(args.model),sess_options=options,providers=["CPUExecutionProvider"])
        if len(session.get_inputs())!=1:raise ValueError("Expected single actor input")
        outputs=session.run(None,{session.get_inputs()[0].name:x})
        if len(outputs)!=1:raise ValueError("Expected single action output")
        y=outputs[0]
    else:
        import torch
        torch.set_num_threads(1)
        extras={'cts_contract.json':'','quiet_execution_control.json':''}
        model=torch.jit.load(str(args.model),map_location="cpu",_extra_files=extras).eval()
        if cts and (json.loads(extras['cts_contract.json'])!=actual or json.loads(extras['quiet_execution_control.json'])!=actual['quiet_execution_control']):
            raise ValueError('CTS embedded/sidecar execution contracts differ')
        if not cts and extras['quiet_execution_control.json']:raise ValueError('Quiet execution requires its dedicated runtime')
        with torch.inference_mode():
            y=(model(torch.from_numpy(x),torch.from_numpy(raw)) if cts else model(torch.from_numpy(x))).cpu().numpy()
    if y.shape!=(1,16) or not np.isfinite(y).all():raise ValueError("Model output must be finite [1,16]")
    digest=hashlib.sha256(args.model.read_bytes()).hexdigest()
    print(json.dumps({"model":str(args.model),"sha256":digest,"input_shape":list(x.shape),"output_shape":list(y.shape),
                      "raw_command_shape":[1,3] if cts else None,"nominal_only":cts,
                      "finite":True,"device":"cpu","scope":"ABI probe only, no locomotion validation"},indent=2))


if __name__=="__main__":main()
