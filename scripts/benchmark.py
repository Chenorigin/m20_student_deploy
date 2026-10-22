#!/usr/bin/env python3
"""Bounded synthetic CPU benchmark and independent terrain-error report."""
import argparse
import json
import platform
from pathlib import Path
import resource
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src/m20_heightmap"))
from m20_heightmap.estimator import ElevationMapper
from m20_heightmap.geometry import Pose
from m20_heightmap.contract import GRID


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--points",type=int,default=20000)
    ap.add_argument("--frames",type=int,default=30)
    ap.add_argument("--output",type=Path,default=ROOT/"evidence/cpu_benchmark.json")
    args=ap.parse_args()
    if not 1000<=args.points<=200000 or not 1<=args.frames<=200:
        raise ValueError("Benchmark limited to 1k..200k points, 1..200 frames")
    rng=np.random.default_rng(42)
    xy=rng.uniform([-.85,-.55],[.85,.55],(args.points,2))
    truth=lambda x,y:.2*x-.1*y
    points=np.column_stack((xy,truth(xy[:,0],xy[:,1])+rng.normal(0,.01,args.points)))
    ranges=np.linalg.norm(points-[0,0,.6],axis=1)
    m=ElevationMapper("v1")
    ingest_ms,query_ms=[],[]
    for i in range(args.frames):
        p=Pose(1.+.1*i,[0,0,.6],[0,0,0,1],np.full(6,.0001))
        start=time.perf_counter(); m.ingest(points,ranges,p); ingest_ms.append((time.perf_counter()-start)*1000)
        start=time.perf_counter(); channels,_,_=m.snapshot(p); query_ms.append((time.perf_counter()-start)*1000)
    error=np.abs(channels[0]-(.1-truth(GRID[:,0],GRID[:,1])))
    result={"machine":platform.machine(),"cpu":platform.processor(),"points":args.points,"frames":args.frames,
            "ingest_ms":dict(zip(("p50","p95","max"),map(float,[np.median(ingest_ms),np.percentile(ingest_ms,95),max(ingest_ms)]))),
            "query_ms":dict(zip(("p50","p95","max"),map(float,[np.median(query_ms),np.percentile(query_ms,95),max(query_ms)]))),
            "peak_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            "valid_fraction":float(channels[1].mean()),"height_error_p95_m":float(np.percentile(error[channels[1]==1],95)),
            "scope":"synthetic desktop CPU core only; excludes ROS serialization, scan deskew and GOS hardware"}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()
