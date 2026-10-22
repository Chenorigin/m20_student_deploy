import json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
from m20_heightmap.cloud import read_cloud
from m20_heightmap.contract import ObservationPacker, make_proprio, GRID
from m20_heightmap.runtime import load_config, Pipeline
from m20_heightmap.receiver import receive_terrain
from test_estimator import pose, surface

ROOT = Path(__file__).resolve().parents[1]


def config(tmp_path, **updates):
    cfg = json.loads((ROOT/"src/m20_heightmap/config/v1.json").read_text())
    cfg.update(calibration_confirmed=True, **updates)
    cfg["lidar_extrinsics"] = {"lidar": {"translation_m": [0,0,0], "quaternion_xyzw": [0,0,0,1]}}
    path = tmp_path/"config.json"; path.write_text(json.dumps(cfg))
    return load_config(path)


def cloud_message(points, big=False, rows=1, padding=0):
    points = np.asarray(points)
    width = len(points)//rows
    row_step = width*16+padding
    data = bytearray(rows*row_step)
    for col in range(3):
        a = np.ndarray((rows, width), dtype=">f4" if big else "<f4", buffer=data, offset=col*4, strides=(row_step,16))
        a[:] = points[:, col].reshape(rows, width)
    return NS(header=NS(frame_id="lidar"), height=rows, width=width, point_step=16, row_step=row_step,
              is_bigendian=big, data=bytes(data), fields=[NS(name=n, offset=i*4, count=1, datatype=7) for i,n in enumerate(("x","y","z"))])


@pytest.mark.parametrize("big", [False,True])
def test_cloud_row_padding_and_malformed(big):
    points = np.arange(36).reshape(12,3)*.1
    msg = cloud_message(points,big,3,12)
    np.testing.assert_allclose(read_cloud(msg)[0],points,atol=1e-6)
    msg.data = msg.data[:-1]
    with pytest.raises(ValueError): read_cloud(msg)
    msg = cloud_message(points)
    with pytest.raises(ValueError): read_cloud(msg,max_points=2)
    msg.fields[1].offset=15
    with pytest.raises(ValueError): read_cloud(msg)


def test_history_order_reset_and_discontinuity():
    terrain = np.zeros((3,187),dtype=np.float32)
    terrain[0]=np.arange(187)/187; terrain[1]=1; terrain[2]=.5
    p = ObservationPacker("v1",4)
    for step in range(5):
        obs = p.pack(np.full(57,step+1),terrain,step,1.+step*.02)
        assert len(obs)==789
        np.testing.assert_array_equal(obs[:57],step+1)
        np.testing.assert_array_equal(obs[57:618],terrain.ravel())
        np.testing.assert_array_equal(obs[618:].reshape(3,57)[:,0], [max(0,step-2),max(0,step-1),step])
    with pytest.raises(ValueError): p.pack(np.ones(57),terrain,7,1.14)
    assert not p.pack(np.ones(57),terrain,8,1.16)[618:].any()
    assert len(ObservationPacker("v0").pack(np.ones(57),terrain,0,0))==618
    with pytest.raises(ValueError): ObservationPacker("v0",4)
    for _ in range(3):
        with pytest.raises(ValueError): p.pack(np.ones(57),terrain,8,1.16)


def test_proprio_values():
    p=make_proprio([1,2,3],[0,0,-1],[.1,.2,.3],np.ones(16),np.full(16,2),np.zeros(16),np.arange(16))
    np.testing.assert_array_equal(p[:3],[.25,.5,.75])
    assert not p[21:25].any()
    np.testing.assert_allclose(p[25:41],.1)
    np.testing.assert_array_equal(p[41:],np.arange(16))


def test_live_processing_path_and_pose_jump(tmp_path):
    cfg=config(tmp_path)
    pipe=Pipeline(cfg)
    pipe.add_pose(pose(.99));pipe.add_pose(pose(1.01))
    cloud=surface(); msg=cloud_message(cloud-[0,0,.6])
    pipe.ingest_message(msg,1.,1.01)
    anchor,(channels,var,age)=pipe.snapshot(1.02)
    np.testing.assert_allclose(channels[0],.1,atol=1e-6)
    assert channels[1].all() and np.all(var>0) and np.all(age>0)
    msg.header.frame_id="wrong"
    with pytest.raises(ValueError): pipe.ingest_message(msg,1.,1.02)
    with pytest.raises(ValueError): pipe.snapshot(1.5)
    with pytest.raises(ValueError): pipe.add_pose(pose(1.02,(5,0,.6)))
    assert not pipe.mapper.patches


def test_receiver_stale_future_unknown(tmp_path):
    cfg=config(tmp_path)
    msg=NS(header=NS(stamp=NS(sec=1,nanosec=0),frame_id="base_yaw"),schema_version=2,profile="v1",
           height=[.1]*187,validity=[1.]*187,confidence=[.5]*187,variance=[.0025]*187,age=[0.]*187)
    assert receive_terrain(msg,1.02,cfg)[2,0]<.5
    with pytest.raises(ValueError): receive_terrain(msg,1.2,cfg)
    with pytest.raises(ValueError): receive_terrain(msg,.9,cfg)
    msg.validity[0]=0
    with pytest.raises(ValueError): receive_terrain(msg,1.02,cfg)


def test_one_second_cells_still_require_fresh_query_pose(tmp_path):
    cfg=config(tmp_path)
    cfg['estimator']['max_map_age_s']=1.
    msg=NS(header=NS(stamp=NS(sec=1,nanosec=0),frame_id='base_yaw'),schema_version=2,profile='v1',
           height=[.1]*187,validity=[1.]*187,confidence=[.5]*187,variance=[.0025]*187,age=[.97]*187)
    channels=receive_terrain(msg,1.02,cfg)
    assert channels[1].all() and (channels[2] < .5).all()
    with pytest.raises(ValueError,match='Insufficient valid terrain coverage'):
        receive_terrain(msg,1.04,cfg)
    msg.age=[0.]*187
    # Measured ages at the two real RL failures must remain rejected even for
    # the new one-second cell lifetime. These are different freshness limits.
    for elapsed in [.0608975887,.0726275444]:
        with pytest.raises(ValueError,match='Terrain/query pose stale'):
            receive_terrain(msg,1.+elapsed,cfg)


def test_publication_budget_reserves_consumer_deadline(tmp_path):
    cfg = config(tmp_path)
    assert cfg['runtime']['max_publish_query_age_s'] == cfg['aos']['max_alignment_age_s']
    cfg['runtime']['max_publish_query_age_s'] = .02
    path = tmp_path/'config.json'
    path.write_text(json.dumps(cfg))
    assert load_config(path)['runtime']['max_publish_query_age_s'] == .02
    cfg['runtime']['max_publish_query_age_s'] = .061
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='Publication query age must not exceed alignment'):
        load_config(path)


@pytest.mark.parametrize('filename', ['heightmap_v1.json', 'heightmap_v1_20261016_14800.offline.json'])
def test_v1_candidate_relaxed_query_still_enforces_coverage_and_cell_age(tmp_path, filename):
    candidate=json.loads((ROOT/'deploy'/filename).read_text())
    candidate['calibration_confirmed']=True  # Offline fixture, no physical receipt.
    candidate['visibility']['site_config']=str(ROOT/'deploy/site.json')
    path=tmp_path/'candidate.json';path.write_text(json.dumps(candidate))
    cfg=load_config(path)
    assert cfg['aos']['max_alignment_age_s']==candidate['aos']['max_alignment_age_s']
    assert cfg['aos']['min_valid_fraction']==candidate['aos']['min_valid_fraction']
    assert cfg['runtime']['max_publish_query_age_s']==candidate['runtime']['max_publish_query_age_s']
    assert cfg['runtime']['pose_max_gap_s']==candidate['runtime']['pose_max_gap_s']
    for flag in ('coverage_hold_enabled','terrain_timing_hold_enabled'):
        assert cfg['aos'][flag] is candidate['aos'].get(flag,True)
        cfg['aos'][flag]=True  # This existing scenario explicitly exercises strict admission.
    minimum=int(np.ceil(cfg['aos']['min_valid_fraction']*187))
    unknown=187-minimum
    query_limit=cfg['aos']['max_alignment_age_s']
    msg=NS(header=NS(stamp=NS(sec=1,nanosec=0),frame_id=cfg['frames']['output']),
           schema_version=2,profile='v1',height=[.1]*minimum+[0.]*unknown,
           validity=[1.]*minimum+[0.]*unknown,confidence=[.5]*minimum+[0.]*unknown,
           variance=[.0025]*187,age=[0.]*minimum+[-1.]*unknown)
    assert receive_terrain(msg,1.+query_limit*.8,cfg)[1].sum()==minimum
    with pytest.raises(ValueError,match='Terrain/query pose stale'):
        receive_terrain(msg,1.+query_limit+.001,cfg)
    msg.validity[minimum-1]=msg.height[minimum-1]=msg.confidence[minimum-1]=0.
    with pytest.raises(ValueError,match='Insufficient valid terrain coverage'):
        receive_terrain(msg,1.+query_limit*.1,cfg)
    msg.validity[minimum-1]=1.;msg.height[minimum-1]=.1;msg.confidence[minimum-1]=.5
    msg.age[:minimum]=[cfg['estimator']['max_map_age_s']-query_limit*.25]*minimum
    with pytest.raises(ValueError,match='Insufficient valid terrain coverage'):
        receive_terrain(msg,1.+query_limit*.5,cfg)


def test_grid_and_uncalibrated_config(tmp_path):
    np.testing.assert_allclose(GRID[0],[-.8,-.5])
    np.testing.assert_allclose(GRID[16],[.8,-.5])
    np.testing.assert_allclose(GRID[17],[-.8,-.4])
    np.testing.assert_allclose(GRID[-1],[.8,.5])
    with pytest.raises(ValueError): load_config(ROOT/"src/m20_heightmap/config/v1.json")
    cfg=config(tmp_path)
    cfg["estimator"]["made_up_option"]=3
    path=tmp_path/"bad.json";path.write_text(json.dumps(cfg))
    with pytest.raises(TypeError): load_config(path)


def test_point_time_offsets_age_entire_scan(tmp_path):
    cfg=config(tmp_path)
    cfg["runtime"]["point_time_field"]="t"
    pipe=Pipeline(cfg)
    for t in (9.9,9.92,9.96,10.):pipe.add_pose(pose(t))
    xyz=surface()-[0,0,.6]
    points=np.column_stack((xyz,np.full(len(xyz),-.08))).astype("<f4")
    msg=NS(header=NS(frame_id="lidar"),height=1,width=len(points),point_step=16,row_step=len(points)*16,
           is_bigendian=False,data=points.tobytes(),fields=[NS(name=n,offset=i*4,count=1,datatype=7) for i,n in enumerate(("x","y","z","t"))])
    pipe.ingest_message(msg,10.,10.)
    assert not pipe.mapper.snapshot(pose(10.24))[0].any()


def test_imu_mode_no_history_and_short_hold(tmp_path):
    cfg=config(tmp_path,pose_mode="imu")
    pipe=Pipeline(cfg)
    pipe.add_pose(pose(.99,(0,0,0)));pipe.add_pose(pose(1.01,(0,0,0)))
    pipe.ingest_message(cloud_message(surface()-[0,0,.6]),1.,1.01)
    assert pipe.snapshot(1.02)[1][0][1].all()
    with pytest.raises(ValueError):pipe.snapshot(1.05)
    assert pipe.mapper.retain is False


def test_empty_cloud_and_nan_returns(tmp_path):
    msg=NS(height=1,width=0,point_step=12,row_step=0,is_bigendian=False,data=b"",
           fields=[NS(name=n,offset=i*4,count=1,datatype=7) for i,n in enumerate(("x","y","z"))])
    assert read_cloud(msg)[0].shape==(0,3)
    cfg=config(tmp_path)
    p=Pipeline(cfg);p.add_pose(pose(.99));p.add_pose(pose(1.01))
    xyz=surface()-[0,0,.6];xyz[0]=np.nan
    p.ingest_message(cloud_message(xyz),1.,1.01)
    assert p.snapshot(1.02)[1][0][1].all()


def test_real_airy_absolute_timestamp_and_relative_equivalence(tmp_path):
    header = 1720572001.2
    xyz = surface()-[0,0,.6]
    data = bytearray(len(xyz)*26)
    for i in range(3):
        np.ndarray((len(xyz),),dtype='<f4',buffer=data,offset=i*4,strides=(26,))[:] = xyz[:, i]
    offsets = np.full(len(xyz), .02)
    np.ndarray((len(xyz),),dtype='<f8',buffer=data,offset=18,strides=(26,))[:] = header+offsets
    msg=NS(header=NS(frame_id='lidar',stamp=NS(sec=1720572001,nanosec=200000000)),height=1,width=len(xyz),point_step=26,row_step=len(data),is_bigendian=False,data=bytes(data),fields=[NS(name=n,offset=o,count=1,datatype=t) for n,o,t in [('x',0,7),('y',4,7),('z',8,7),('timestamp',18,8)]])
    _, measured=read_cloud(msg,time_field='timestamp',time_mode='absolute')
    np.testing.assert_allclose(measured, offsets, atol=2e-7)
    cfg=config(tmp_path);cfg['runtime'].update(point_time_field='timestamp',point_time_mode='absolute')
    pipe=Pipeline(cfg)
    for t in (header-.02,header+.02,header+.04):pipe.add_pose(pose(t))
    pipe.ingest_message(msg,header,header+.04)
    assert pipe.snapshot(header+.04)[1][0][1].all()
    cfg['runtime']['point_time_mode']='relative'
    with pytest.raises(ValueError):Pipeline(cfg).ingest_message(msg,header,header+.04)


@pytest.mark.parametrize("quality", ["variance", "speed", "angular_speed"])
def test_mid_scan_pose_quality_reduces_confidence(tmp_path, quality):
    cfg = config(tmp_path)
    cfg["runtime"].update(point_time_field="t", scan_duration_s=.21)
    xyz = surface()-[0, 0, .6]
    samples = np.column_stack((xyz, np.linspace(0, .2, len(xyz)))).astype("<f4")
    msg = NS(header=NS(frame_id="lidar"), height=1, width=len(samples), point_step=16,
             row_step=samples.nbytes, is_bigendian=False, data=samples.tobytes(),
             fields=[NS(name=n, offset=i*4, count=1, datatype=7) for i, n in enumerate(("x", "y", "z", "t"))])
    snapshots = []
    for degraded in (False, True):
        pipe = Pipeline(cfg)
        for time in (1., 1.04, 1.08, 1.12, 1.16, 1.2, 1.224):
            p = pose(time)
            if degraded and 1.04 <= time <= 1.16:
                if quality == "variance":
                    p.variance[4] = .25
                else:
                    setattr(p, quality, 3.)
            pipe.add_pose(p)
        pipe.ingest_message(msg, 1., 1.224)
        snapshots.append(pipe.snapshot(1.224)[1])
    low, high = snapshots
    valid = low[0][1].astype(bool) & high[0][1].astype(bool)
    assert valid.any()
    np.testing.assert_allclose(low[0][0, valid], high[0][0, valid], atol=1e-7)
    # surface() supplies 25 inliers per cell in row order. Each cell ages
    # from its own earliest actual return, independent of pose uncertainty.
    expected_age=.224-samples[:,3].reshape(187,25)[:,0]
    np.testing.assert_allclose(high[2][valid],expected_age[valid],atol=1e-6)
    np.testing.assert_array_equal(low[2][valid],high[2][valid])
    assert np.any(high[1][valid] > low[1][valid]+1e-6)
    assert np.any(high[0][2, valid] < low[0][2, valid])
