#!/usr/bin/env python3
"""Bounded offline measured-support and model-ABI check; no ROS or actuators."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
from m20_heightmap.capture import capture_memory_budget
from m20_heightmap.contract import GRID, ObservationPacker
from m20_heightmap.estimator import ElevationMapper, EstimatorConfig
from m20_heightmap.geometry import Pose


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--padding',type=float,default=None)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Preserve previous receipts: output must be new')
    cfg=EstimatorConfig()
    if args.config:
        from m20_heightmap.runtime import load_config
        cfg=EstimatorConfig(**load_config(args.config)['estimator'])
    if args.padding is not None:cfg.map_capture_padding_m=args.padding
    cfg.validate()
    budget=capture_memory_budget(asdict(cfg))
    memory={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:','MemAvailable:'))}
    reserve=512*1024**2+budget['capture_reserve_bytes']
    if memory['MemAvailable']<reserve:raise RuntimeError('Insufficient available memory for bounded offline check')
    offsets=np.array([(x,y) for y in np.linspace(-.04,.04,5) for x in np.linspace(-.04,.04,5)])
    begin=time.monotonic();cases=[]
    for padding in (0.,cfg.map_capture_padding_m):
        settings=asdict(cfg);settings['map_capture_padding_m']=padding
        mapper=ElevationMapper(cfg=EstimatorConfig(**settings))
        xy=(mapper.capture_grid[:,None]+offsets).reshape(-1,2)
        points=np.column_stack((xy,.1*xy[:,0]+.2*xy[:,1]))
        anchor=Pose(1.,[0.,0.,.6],[0.,0.,0.,1.],np.zeros(6))
        fit=time.monotonic();mapper.ingest(points,np.linalg.norm(points-anchor.position,axis=1),anchor)
        fit_ms=(time.monotonic()-fit)*1000
        current=Pose(1.1,[.4,0.,.6],[0.,0.,0.,1.],np.zeros(6))
        observation,variance,age=mapper.snapshot(current)
        target=(GRID[:,0]>.79)&(abs(GRID[:,1])<1e-6)
        assert observation.shape==(3,187) and variance.shape==age.shape==(187,)
        assert observation[1,target].item()==float(padding>=.4)
        if padding>=.4:np.testing.assert_allclose(observation[0,target],[-.02],atol=1e-6)
        assert len(ObservationPacker('v1',4).pack(np.zeros(57),observation,0,1.1))==789
        expired=Pose(2.,[.4,0.,.6],[0.,0.,0.,1.],np.zeros(6))
        assert not mapper.snapshot(expired)[0].any()
        cases.append({'padding_m':padding,'source_shape':list(mapper.capture_shape),
                      'source_cells':len(mapper.capture_grid),'query_valid_cells':int(observation[1].sum()),
                      'front_rows':int(observation[1].reshape(11,17)[:,12:].any(0).sum()),'fit_ms':fit_ms})
    from m20_heightmap import capture,estimator,runtime
    modules={module.__name__:{'path':str(Path(module.__file__).resolve()),
                'sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}
             for module in (capture,estimator,runtime)}
    result={'passed':True,'cases':cases,'capture_budget':budget,'memory_before':memory,
            'offline_reserved_bytes':reserve,'modules':modules,'wall_s':time.monotonic()-begin,
            'max_rss_kb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'motor_publishers_created':0,'physical_motion_verified':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
