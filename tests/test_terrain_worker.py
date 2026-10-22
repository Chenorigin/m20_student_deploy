from collections import deque
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing
import os
from types import SimpleNamespace
import numpy as np
import pytest
from m20_heightmap.runtime import Pipeline
from m20_heightmap.terrain_worker import estimate_patch, estimate_packed_patch, pack_job_inputs, unpack_job_inputs
from test_interfaces import config,cloud_message
from test_estimator import pose,surface


@pytest.mark.parametrize('profile',['v0','v1'])
@pytest.mark.parametrize('compact',[False,True])
def test_spawn_transfers_actual_job_and_preserves_patch_and_errors(tmp_path,profile,compact):
    cfg=config(tmp_path,profile=profile,history_steps=1 if profile=='v0' else 4)
    msg=cloud_message(surface(noise=.004)-[0,0,.6])
    msg.header.stamp=SimpleNamespace(sec=1,nanosec=0)
    poses=deque([pose(.99),pose(1.01)])
    baseline=Pipeline(cfg);baseline.poses.poses=poses
    baseline.ingest_message(msg,1.,1.01,processing_now=lambda:1.02)
    with ProcessPoolExecutor(1,mp_context=multiprocessing.get_context('spawn')) as pool:
        assert pool.submit(os.getpid).result(timeout=15)!=os.getpid()
        fn=estimate_packed_patch if compact else estimate_patch
        def arguments():
            return (cfg,pack_job_inputs(msg,poses),1.,1.01,-float('inf'),partial(float,1.02)) if compact else (cfg,msg,1.,1.01,poses,-float('inf'),partial(float,1.02))
        patch,stats,scan,elapsed=pool.submit(fn,*arguments()).result(timeout=15)
        # Compute durations belong to separate executions and cannot be equal.
        # All deterministic values, including support/rejection diagnostics,
        # must survive the process boundary exactly.
        for values in (stats, baseline.mapper.stats):
            timing=values['pipeline_timings_ms']
            assert set(timing)=={'decode','prefilter','deskew','fit'}
            assert all(np.isfinite(value) and value>=0 for value in timing.values())
        assert {k:v for k,v in stats.items() if k!='pipeline_timings_ms'} == {
            k:v for k,v in baseline.mapper.stats.items() if k!='pipeline_timings_ms'}
        for name in ('z','valid','variance','blocked','pose_variance','cell_stamps','block_stamps','cell_ends','block_ends'):
            np.testing.assert_array_equal(getattr(patch,name),getattr(baseline.mapper.patches[-1],name))
        assert scan.stamp==1. and elapsed>=0
        msg.data=msg.data[:-1]
        with pytest.raises(ValueError):
            pool.submit(fn,*arguments()).result(timeout=15)


def test_compact_wire_and_parent_pose_values_are_exact_and_mutations_rejected():
    msg=cloud_message(np.arange(36).reshape(-1,3)*.01,big=True,rows=3,padding=12)
    msg.header.stamp=SimpleNamespace(sec=123,nanosec=987654321)
    poses=deque([pose(123.,yaw=-1.2),pose(123.04,yaw=-1.19)])
    packed=pack_job_inputs(msg,poses);restored,decoded=unpack_job_inputs(packed)
    assert restored.data==msg.data and restored.row_step==msg.row_step and restored.is_bigendian
    assert restored.header.stamp.nanosec==987654321
    for a,b in zip(poses,decoded):
        for name in ('stamp','position','quaternion','variance','speed','angular_speed','R','yaw'):
            np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
    for column,value in ((0,float('nan')),(8,-1.),(14,-1.),(4,10.),(16,2.),(25,1.)):
        bad=packed[1].copy();bad[0,column]=value
        with pytest.raises(ValueError):unpack_job_inputs((packed[0],bad))


def test_uninitialized_worker_cannot_silently_skip_real_processing_clock(tmp_path):
    with pytest.raises(RuntimeError,match='clock not initialized'):
        estimate_patch(config(tmp_path),None,1.,1.01,deque(),-float('inf'))
