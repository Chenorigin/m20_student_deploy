import math
import numpy as np
import pytest
from m20_heightmap.contract import GRID
from m20_heightmap.geometry import Pose, rotation, to_world, PoseBuffer, deskew
from m20_heightmap.estimator import ElevationMapper, EstimatorConfig, confidence


def pose(t=1., xyz=(0., 0., .6), yaw=0.):
    return Pose(t, xyz, [0, 0, math.sin(yaw/2), math.cos(yaw/2)], np.full(6, .0001))


def surface(function=lambda x, y: 0*x, noise=0., seed=42):
    rng = np.random.default_rng(seed)
    offsets = np.array([(x, y) for y in np.linspace(-.04, .04, 5) for x in np.linspace(-.04, .04, 5)])
    xy = (GRID[:, None]+offsets).reshape(-1, 2)
    z = function(xy[:, 0], xy[:, 1])+rng.normal(0, noise, len(xy))
    return np.column_stack((xy, z))


def ingest(mapper, points, p):
    mapper.ingest(points, np.linalg.norm(points-p.position, axis=1), p)


@pytest.mark.parametrize("profile", ["v0", "v1"])
@pytest.mark.parametrize("kind", ["flat", "slope", "ascent", "descent"])
def test_known_surfaces(profile, kind):
    functions = {"flat": lambda x,y: 0*x, "slope": lambda x,y: .2*x-.1*y,
                 "ascent": lambda x,y: .15*np.floor((x+.85)/.3),
                 "descent": lambda x,y: -.15*np.floor((x+.85)/.3)}
    f = functions[kind]
    points = surface(f, .004)
    # A few isolated gross returns in each cell, absent in the independent truth.
    points[::25, 2] += .4
    m = ElevationMapper(profile)
    ingest(m, points, pose())
    h, valid, c = m.snapshot(pose())[0]
    expected = np.clip(.1-f(GRID[:, 0], GRID[:, 1]), -1, 1)
    assert valid.mean() > .85
    assert np.percentile(np.abs(h[valid == 1]-expected[valid == 1]), 95) < .012
    assert np.all((c > 0) & (c <= 1))


def test_age_noise_dropout_profiles():
    cloud = surface()
    for profile in ("v0", "v1"):
        m = ElevationMapper(profile)
        ingest(m, cloud, pose())
        first = m.snapshot(pose())[0]
        aged = m.snapshot(pose(1.1))[0]
        assert np.all(aged[2] < first[2])
        ingest(m, cloud[cloud[:, 0] > 0.1], pose(1.11))
        h, valid, c = m.snapshot(pose(1.11))[0]
        assert np.all(valid[GRID[:, 0] < 0] == (1 if profile == "v1" else 0))
        assert not m.snapshot(pose(1.5))[0].any()
    quiet, noisy = ElevationMapper(), ElevationMapper()
    ingest(quiet, surface(noise=.001), pose())
    ingest(noisy, surface(noise=.025), pose())
    assert noisy.snapshot(pose())[0][2].mean() < quiet.snapshot(pose())[0][2].mean()


def test_unknown_multilayer_and_vertical_wall():
    m = ElevationMapper()
    ingest(m, np.empty((0, 3)), pose())
    assert not m.snapshot(pose())[0].any()
    points = surface()
    upper = points.copy()
    upper[:, 2] += .2
    ingest(m, np.vstack((points, upper)), pose(1.1))
    assert not m.snapshot(pose(1.1))[0].any()
    y, z = np.meshgrid(np.linspace(-.5, .5, 100), np.linspace(-.5, .4, 20))
    wall = np.column_stack((np.zeros(y.size), y.ravel(), z.ravel()))
    ingest(m, wall, pose(1.2))
    assert not m.snapshot(pose(1.2))[0].any()


def test_blocked_surface_invalidates_old_memory():
    m = ElevationMapper()
    p = surface()
    ingest(m, p, pose())
    upper = p.copy(); upper[:, 2] += .2
    ingest(m, np.vstack((p, upper)), pose(1.1))
    assert not m.snapshot(pose(1.1))[0].any()


def test_world_translation_yaw_invariance_and_reprojection():
    f = lambda x,y: .2*x-.1*y
    cloud = surface(f)
    a, b = ElevationMapper(), ElevationMapper()
    ingest(a, cloud, pose())
    yaw = .8
    moved = pose(xyz=(2., -3., 1.6), yaw=yaw)
    transformed = cloud@moved.R.T+[2., -3., 1.]
    ingest(b, transformed, moved)
    np.testing.assert_allclose(a.snapshot(pose())[0], b.snapshot(moved)[0], atol=1e-6)
    # Query +0.1m in X: same fixed world terrain, height must follow new XY.
    out = a.snapshot(pose(1.02, (.1, 0., .6)))[0]
    expected = .1-f(GRID[:, 0]+.1, GRID[:, 1])
    usable = out[1] == 1
    np.testing.assert_allclose(out[0, usable], expected[usable], atol=1e-6)
    assert np.all(out[1, GRID[:, 0] > .7] == 0)


def test_extrinsic_pitch_and_invalid_quaternion():
    q = [0, math.sin(.3), 0, math.cos(.3)]
    p = Pose(1., [1, 2, .6], q, np.zeros(6))
    qb = [math.sin(.1), 0, 0, math.cos(.1)]
    ext = np.array([.1, .03, .15])
    world = surface()
    local = ((world-p.position)@p.R-ext)@rotation(qb)
    np.testing.assert_allclose(to_world(local, p, ext, qb), world, atol=1e-12)
    with pytest.raises(ValueError):
        rotation([0, 0, 0, 0])


def test_deskew_and_bracketing():
    buffer = PoseBuffer(max_gap=.2)
    a, b = pose(1.), pose(1.1, (.1, 0., .6))
    buffer.add(a); buffer.add(b)
    offsets = np.linspace(0, .1, 100)
    points = np.column_stack((1.-offsets, np.zeros(100), np.full(100, -.6)))
    world = deskew(points, offsets, a, buffer, [0,0,0], [0,0,0,1], .001)
    assert np.max(np.abs(world[:, 0]-1)) < .001
    assert np.max(np.abs(to_world(points, a, [0,0,0], [0,0,0,1])[:, 0]-1)) > .09
    with pytest.raises(ValueError):
        buffer.at(1.2)
    with pytest.raises(ValueError):
        buffer.add(pose(.9))


def test_confidence_closed_form_and_bad_inputs():
    cfg = EstimatorConfig()
    out = confidence(np.ones(4, dtype=bool), np.array([.0025, .0025, np.nan, -.1]), np.array([0, .1, 0, 0]), cfg)
    np.testing.assert_allclose(out[:2], [.5, .5*np.exp(-.2)])
    assert not out[2:].any()


def test_query_vertical_covariance_and_conservative_capture_age():
    m = ElevationMapper()
    cloud = surface()
    m.ingest(cloud, np.linalg.norm(cloud-pose().position,axis=1), pose(), capture_stamp=.92)
    low = m.snapshot(pose(1.02))[0]
    p = pose(1.02); p.variance[2]=.25
    high = m.snapshot(p)[0]
    assert high[2].mean() < low[2].mean()*.1
    assert not m.snapshot(pose(1.24))[0].any()


def test_near_unit_quaternion_is_normalized():
    from m20_heightmap.runtime import Pipeline
    from pathlib import Path
    import json
    cfg=json.loads((Path(__file__).resolve().parents[1]/"src/m20_heightmap/config/v1.json").read_text())
    p=Pipeline(cfg)
    p.add_pose(Pose(1.,[0,0,.6],[0,0,0,.998],np.zeros(6)))
    p.add_pose(Pose(1.02,[0,0,.6],[0,0,0,.998],np.zeros(6)))
    assert len(p.poses.poses)==2
