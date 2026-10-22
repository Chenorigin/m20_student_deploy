import numpy as np
import pytest
from types import SimpleNamespace as NS
from m20_heightmap.estimator import ElevationMapper,row_nanmedian
from m20_heightmap.runtime import Pipeline
from test_interfaces import config
from test_estimator import pose,surface


def test_unrelated_early_cell_does_not_expire_fresh_cell_in_same_patch():
    cloud=surface();times=np.repeat(np.where(np.arange(187)<90,.1,.28),25)
    m=ElevationMapper('v1');m.ingest(cloud,np.linalg.norm(cloud-pose(.1).position,axis=1),pose(.1),capture_stamp=.1,sample_stamps=times)
    channels,_,age=m.snapshot(pose(.45))
    assert not channels[1,:90].any()
    assert channels[1,90:].all()
    np.testing.assert_allclose(age[90:],.17,atol=1e-7)
    assert len(m.patches)==1
    assert not m.snapshot(pose(.59))[0][1].any()
    assert not m.patches


def test_excluded_early_outlier_never_sets_plane_measurement_time():
    cloud=surface();cloud[::25,2]+=.4
    times=np.full(len(cloud),.2);times[::25]=0.
    m=ElevationMapper('v1');m.ingest(cloud,np.linalg.norm(cloud-pose(0).position,axis=1),pose(0),capture_stamp=0.,sample_stamps=times)
    channels,_,age=m.snapshot(pose(.35))
    assert channels[1].all()
    np.testing.assert_allclose(age,.15,atol=1e-7)
    np.testing.assert_allclose(channels[0],.1,atol=1e-7)


@pytest.mark.parametrize('profile',['v0','v1'])
@pytest.mark.parametrize('old_offset,upper,expected_valid',[(0.,False,True),(.1,True,False)])
def test_processing_age_filters_only_expired_evidence(tmp_path,profile,old_offset,upper,expected_valid):
    cfg=config(tmp_path,profile=profile,history_steps=1 if profile=='v0' else 4)
    cfg['runtime'].update(point_time_field='t',scan_duration_s=.21)
    floor=surface()-[0,0,.6]
    newer=floor.copy()
    if upper:newer[:,2]+=.2
    samples=np.vstack((np.column_stack((floor,np.full(len(floor),old_offset))),
                       np.column_stack((newer,np.full(len(floor),.2))))).astype('<f4')
    msg=NS(header=NS(frame_id='lidar'),height=1,width=len(samples),point_step=16,
           row_step=samples.nbytes,is_bigendian=False,data=samples.tobytes(),
           fields=[NS(name=n,offset=i*4,count=1,datatype=7) for i,n in enumerate(('x','y','z','t'))])
    pipeline=Pipeline(cfg)
    for t in np.arange(10.,10.426,.025):pipeline.add_pose(pose(float(t)))
    pipeline.add_pose(pose(10.43))
    pipeline.ingest_message(msg,10.,10.34,processing_now=lambda:10.35)
    channels,_,age=pipeline.snapshot(10.43)[1]
    if expected_valid:
        assert channels[1].all()
        np.testing.assert_allclose(age,.23,atol=1e-6)
    else:
        assert not channels[1].any()  # Both still-fresh layers remain contradictory.
    backwards=iter((10.36,10.35))
    with pytest.raises(ValueError,match='Processing clock nonfinite/reversed'):
        pipeline.ingest_message(msg,10.,10.34,processing_now=lambda:next(backwards))


def test_fast_row_median_is_identical_for_masked_support():
    rng=np.random.default_rng(841)
    for width in (5,32,128):
        data=rng.normal(size=(200,width));data[rng.random(data.shape)<.4]=np.nan
        data[:,0]=0.  # Every row has support; includes odd/even cardinalities.
        np.testing.assert_array_equal(row_nanmedian(data),np.nanmedian(data,axis=1))
