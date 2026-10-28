"""Extended measured support must not alter the model's fixed terrain ABI."""
import math
import numpy as np
import pytest
from m20_heightmap.capture import capture_geometry, capture_memory_budget
from m20_heightmap.contract import GRID, ObservationPacker
from m20_heightmap.estimator import ElevationMapper, EstimatorConfig
from m20_heightmap.runtime import Pipeline
from test_estimator import pose, ingest
from test_interfaces import config, cloud_message


def measured_surface(grid, function=lambda x,y: .1*x+.2*y):
    offsets=np.array([(x,y) for y in np.linspace(-.04,.04,5) for x in np.linspace(-.04,.04,5)])
    xy=(grid[:,None]+offsets).reshape(-1,2)
    return np.column_stack((xy,function(xy[:,0],xy[:,1])))


def mapper(padding):
    return ElevationMapper(cfg=EstimatorConfig(map_capture_padding_m=padding,max_map_age_s=.5))


def test_geometry_and_invalid_configuration():
    grid,shape,bounds=capture_geometry(.6)
    assert shape==(23,29) and len(grid)==667
    np.testing.assert_allclose(grid.reshape(*shape,2)[6:-6,6:-6].reshape(-1,2),GRID,atol=1e-15)
    np.testing.assert_array_equal(capture_geometry(0)[0],GRID)
    np.testing.assert_allclose(bounds,[1.45,1.15])
    for bad in [-.1,2.1,float('nan'),float('inf'),True,'0.6']:
        with pytest.raises(ValueError):capture_geometry(bad)
    budget=capture_memory_budget(vars(EstimatorConfig(map_capture_padding_m=.6)))
    assert budget['capture_points']==667 and budget['capture_reserve_bytes']>0


@pytest.mark.parametrize('yaw,xy',[(0.,(.4,0.)),(math.pi/2,(0.,0.))])
def test_fixed_world_surface_survives_translation_and_yaw(yaw,xy):
    outputs=[]
    for padding in (0.,.6):
        m=mapper(padding);ingest(m,measured_surface(m.capture_grid),pose())
        out,var,age=m.snapshot(pose(1.2,(*xy,.6),yaw))
        assert out.shape==(3,187) and var.shape==age.shape==(187,)
        outputs.append(out)
        world=GRID@np.array([[math.cos(yaw),math.sin(yaw)],[-math.sin(yaw),math.cos(yaw)]])+xy
        # Aligned grid queries should read the old measured world surface.
        expected=.1-.1*world[:,0]-.2*world[:,1]
        usable=out[1]>0
        np.testing.assert_allclose(out[0,usable],expected[usable],atol=1e-6)
        assert len(ObservationPacker('v1',4).pack(np.zeros(57),out,0,1.2))==789
        assert not m.snapshot(pose(1.6,(*xy,.6),yaw))[0].any()
    target=(GRID[:,0]>.79)&(abs(GRID[:,1])<1e-6)
    assert outputs[0][1,target].item()==0
    assert outputs[1][1,target].item()==1


def test_unknown_outside_real_support_is_not_filled():
    m=mapper(.6);ingest(m,measured_surface(GRID),pose())
    out=m.snapshot(pose(1.2,(.4,0.,.6)))[0]
    assert not out[1,GRID[:,0]>.6].any()
    m.reset();assert not m.patches


def test_new_obstacle_invalidates_old_surface_in_expanded_cells():
    m=mapper(.6);cloud=measured_surface(m.capture_grid)
    ingest(m,cloud,pose())
    upper=cloud.copy();upper[:,2]+=.2
    ingest(m,np.vstack((cloud,upper)),pose(1.1))
    assert not m.snapshot(pose(1.2,(.4,0.,.6)))[0].any()


def test_pipeline_prefilter_and_visibility_use_expanded_source(tmp_path):
    cfg=config(tmp_path);cfg['estimator']['map_capture_padding_m']=.6
    # Exercise the real visibility branch on source cells, not the output187.
    cfg['visibility']={'horizontal_fov_deg':360.,'sensors':[
        {'position_base_m':[.32028,0.,-.013],'yaw_rad':0.},
        {'position_base_m':[-.32028,0.,-.013],'yaw_rad':math.pi}]}
    p=Pipeline(cfg);p.add_pose(pose(.99));p.add_pose(pose(1.01))
    world=measured_surface(p.mapper.capture_grid)
    p.ingest_message(cloud_message(world-[0,0,.6]),1.,1.01)
    patch=p.mapper.patches[-1]
    assert len(patch.z)==667 and patch.source_shape==(23,29)
    assert patch.valid[p.mapper.capture_grid[:,0]>1.].any()
    assert p.snapshot(1.02)[1][0].shape==(3,187)
    # A late/mismatched worker packet must never be queried with another layout.
    patch.source_shape=(11,17)
    with pytest.raises(ValueError,match='Capture packet shape'):
        p.mapper.snapshot(pose(1.02))
