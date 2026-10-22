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
    args=ap.parse_args()
    spec=contract(args.profile,args.history_steps);validate_model_contract(args.contract,spec)
    x=np.zeros((1,spec["input_dim"]),dtype=np.float32)
    x[:,5]=-1.
    if args.observation:
        x=np.load(args.observation,allow_pickle=False).astype(np.float32).reshape(1,-1)
    if x.shape!=(1,spec["input_dim"]) or not np.isfinite(x).all():raise ValueError("Invalid observation")
    if args.model.suffix.lower()==".onnx":
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
        model=torch.jit.load(str(args.model),map_location="cpu").eval()
        with torch.inference_mode():y=model(torch.from_numpy(x)).cpu().numpy()
    if y.shape!=(1,16) or not np.isfinite(y).all():raise ValueError("Model output must be finite [1,16]")
    digest=hashlib.sha256(args.model.read_bytes()).hexdigest()
    print(json.dumps({"model":str(args.model),"sha256":digest,"input_shape":list(x.shape),"output_shape":list(y.shape),
                      "finite":True,"device":"cpu","scope":"ABI probe only, no locomotion validation"},indent=2))


if __name__=="__main__":main()
