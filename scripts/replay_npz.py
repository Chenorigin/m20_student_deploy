#!/usr/bin/env python3
"""Offline reproduction of the live pipeline from fixed-size, NaN-padded raw scan arrays."""
import argparse
import json
from pathlib import Path
import sys
import zipfile
from types import SimpleNamespace as NS
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src/m20_heightmap"))
from m20_heightmap.runtime import load_config,Pipeline
from m20_heightmap.geometry import Pose


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("input",type=Path);ap.add_argument("--config",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    cfg=load_config(args.config);p=Pipeline(cfg)
    with zipfile.ZipFile(args.input) as archive:
        if sum(item.file_size for item in archive.infolist()) > 256*1024**2:
            raise ValueError("Replay input exceeds 256 MiB decompressed budget; split the capture")
    with np.load(args.input,allow_pickle=False) as d:
        required={"points","scan_stamps","pose_stamps","positions","quaternions_xyzw","cloud_frame"}
        if not required.issubset(d.files): raise ValueError("Missing arrays: "+str(required-set(d.files)))
        d={name:d[name] for name in d.files}
        if d["points"].ndim!=3 or d["points"].shape[2]!=3 or len(d["points"])>10000:
            raise ValueError("Expected points [frames<=10000,points,3]")
        if len(d["scan_stamps"])!=len(d["points"]): raise ValueError("Scan length mismatch")
        if not np.isfinite(d["scan_stamps"]).all() or not np.all(np.diff(d["scan_stamps"])>0):raise ValueError("Scan stamps must increase")
        k=len(d["pose_stamps"])
        if d["positions"].shape!=(k,3) or d["quaternions_xyzw"].shape!=(k,4):raise ValueError("Pose length mismatch")
        idx=0; outputs=[];diagnostics=[]
        for i,when in enumerate(d["scan_stamps"]):
            xyz=np.asarray(d["points"][i],dtype="<f4")
            offsets=d["point_offsets_s"][i] if "point_offsets_s" in d else None
            last=when+(max(0.,float(np.nanmax(offsets))) if offsets is not None else 0.)
            while idx<len(d["pose_stamps"]) and (not p.poses.poses or p.poses.poses[-1].stamp<last):
                p.add_pose(Pose(float(d["pose_stamps"][idx]),d["positions"][idx],d["quaternions_xyzw"][idx],np.square(cfg["runtime"]["pose_std"])))
                idx+=1
            fields=[NS(name=n,offset=k*4,count=1,datatype=7) for k,n in enumerate(("x","y","z"))]
            data=xyz
            if cfg["runtime"]["point_time_field"]:
                if offsets is None: raise ValueError("Config requires point_offsets_s")
                data=np.column_stack((xyz,offsets/cfg["runtime"]["point_time_scale_s"])).astype("<f4")
                fields.append(NS(name=cfg["runtime"]["point_time_field"],offset=12,count=1,datatype=7))
            step=data.shape[1]*4
            msg=NS(header=NS(frame_id=str(d["cloud_frame"].item())),height=1,width=len(data),point_step=step,
                   row_step=len(data)*step,is_bigendian=False,data=data.tobytes(),fields=fields)
            try:
                p.ingest_message(msg,float(when),last)
                channels,var,age=p.mapper.snapshot(p.poses.at(float(when)))
                diagnostics.append({"frame":i,"status":"ok","valid_fraction":float(channels[1].mean())})
            except ValueError as e:
                channels=np.zeros((3,187),dtype=np.float32);var=np.zeros(187,dtype=np.float32);age=np.full(187,-1,dtype=np.float32)
                diagnostics.append({"frame":i,"status":"rejected","reason":str(e)})
            outputs.append(np.vstack((channels,var,age)))
        np.savez_compressed(args.output,observations=np.asarray(outputs),scan_stamps=d["scan_stamps"])
        args.output.with_suffix(".json").write_text(json.dumps(diagnostics,indent=2)+"\n")
        print(json.dumps({"frames":len(outputs),"rejected":sum(x["status"]!="ok" for x in diagnostics),"output":str(args.output)}))


if __name__=="__main__": main()
