import copy,json,math
from pathlib import Path
import numpy as np
import pytest
from m20_heightmap.visibility import validate_visibility,visible_body
from m20_heightmap.contract import GRID
from m20_heightmap.runtime import Pipeline,load_config
from test_interfaces import config,cloud_message
from test_estimator import pose,surface

ROOT=Path(__file__).resolve().parents[1]


def visibility():
    return validate_visibility(json.loads((ROOT/'deploy/site.json').read_text())['lidar_visibility'])


def test_dual_135_matches_independent_angle_test_and_boundary():
    cfg=visibility();rng=np.random.default_rng(416);points=rng.uniform(-2,2,(3000,3))
    expected=[]
    for p in points:
        yes=False
        for s in cfg['sensors']:
            d=p-np.array(s['position_base_m']);angle=math.atan2(d[1],d[0])-s['yaw_rad']
            angle=math.atan2(math.sin(angle),math.cos(angle))
            yes |= math.hypot(*d[:2])>1e-8 and abs(angle)<=math.radians(67.5)+1e-7
        expected.append(yes)
    np.testing.assert_array_equal(visible_body(points,cfg),expected)
    s=cfg['sensors'][0]['position_base_m'];a=math.radians(67.5)
    boundary=np.array([[s[0]+math.cos(a),math.sin(a),-.6],
                       [s[0]+math.cos(a+1e-4),math.sin(a+1e-4),-.6],
                       [0,1,-.6],[1,0,-.6],[-1,0,-.6]])
    np.testing.assert_array_equal(visible_body(boundary,cfg),[True,False,False,True,True])


@pytest.mark.parametrize('profile',['v0','v1'])
@pytest.mark.parametrize('lifetime',[.5,1.])
def test_physical_135_blind_cells_stay_unknown_and_map_expires(tmp_path,profile,lifetime):
    cfg=config(tmp_path,profile=profile,history_steps=1 if profile=='v0' else 4)
    cfg['visibility']=visibility();cfg['estimator']['max_map_age_s']=lifetime
    cfg['runtime']['scan_stamp_reference']='start'
    p=Pipeline(cfg);p.add_pose(pose(.99));p.add_pose(pose(1.01))
    p.ingest_message(cloud_message(surface()-[0,0,.6]),1.,1.01)
    mask=visible_body(np.column_stack((GRID,np.full(187,-.6))),cfg['visibility'])
    channels,var,age=p.mapper.snapshot(pose(1.+lifetime-.01))
    assert channels[1].sum()>60
    assert not channels[:,~mask].any() and not var[~mask].any()
    np.testing.assert_allclose(age[channels[1]==1],lifetime-.01,atol=1e-6)
    assert not p.mapper.snapshot(pose(1.+lifetime+.01))[0].any()


def test_deploy_profiles_share_physical_135_and_profile_timing_limits(tmp_path):
    for profile in ('v0','v1'):
        cfg=json.loads((ROOT/f'deploy/heightmap_{profile}.json').read_text());cfg['calibration_confirmed']=True
        folder=tmp_path/profile;folder.mkdir();(folder/'site.json').write_bytes((ROOT/'deploy/site.json').read_bytes())
        cfg['visibility']={'site_config':'site.json'};path=folder/'config.json';path.write_text(json.dumps(cfg));got=load_config(path)
        assert got['visibility']==visibility()
        assert got['estimator']['max_map_age_s']==cfg['estimator']['max_map_age_s']
        assert got['runtime']['input_timeout_s']==cfg['runtime']['input_timeout_s']
        assert got['aos']['max_alignment_age_s']==cfg['aos']['max_alignment_age_s']
        assert got['aos']['min_valid_fraction']==cfg['aos']['min_valid_fraction']
        for flag in ('coverage_hold_enabled','terrain_timing_hold_enabled'):
            assert got['aos'][flag] is cfg['aos'].get(flag,True)
        missing=copy.deepcopy(cfg);missing.pop('visibility');path.write_text(json.dumps(missing))
        with pytest.raises(ValueError,match='requires calibrated visibility'):load_config(path)
    bad=copy.deepcopy(visibility());bad['horizontal_fov_deg']=361
    with pytest.raises(ValueError):validate_visibility(bad)
