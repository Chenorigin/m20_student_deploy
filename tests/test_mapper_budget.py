import copy
import numpy as np
import pytest
from m20_heightmap.cloud import validate_cloud_time
from m20_heightmap.runtime import Pipeline
from m20_heightmap.estimator import ElevationMapper
from test_interfaces import config
from test_estimator import pose, surface, ingest


def test_cloud_deadline_does_not_relax_pose_or_modify_capture_times(tmp_path):
    cfg = config(tmp_path)
    cfg['runtime'].update(input_timeout_s=.2, cloud_input_timeout_s=.35)
    times = np.array([0., .1])
    before = times.copy()
    assert validate_cloud_time(10., 10.4, cfg['runtime'], times) == (10., 10.1)
    np.testing.assert_array_equal(times, before)
    old = copy.deepcopy(cfg['runtime']); old.pop('cloud_input_timeout_s')
    with pytest.raises(ValueError, match='stale/future'):
        validate_cloud_time(10., 10.4, old, times)
    with pytest.raises(ValueError, match='stale/future'):
        validate_cloud_time(10., 10.451, cfg['runtime'], times)
    pipe = Pipeline(cfg)
    pipe.add_pose(pose(10.))
    with pytest.raises(ValueError, match='Pose stale/future'):
        pipe.snapshot(10.3)


def test_query_source_trace_preserves_output_and_expires_with_support():
    baseline = ElevationMapper()
    traced = ElevationMapper(); traced.trace_sources = True
    p = pose(1.)
    for mapper in (baseline, traced):
        ingest(mapper, surface(), p)
    ordinary, observed = baseline.snapshot(p), traced.snapshot(p)
    for a, b in zip(ordinary, observed):
        np.testing.assert_array_equal(a, b)
    valid = observed[0][1] == 1
    np.testing.assert_array_equal(traced.query_sources['scan_stamp_s'][valid], np.ones(valid.sum()))
    np.testing.assert_array_equal(traced.query_sources['source_cell'][valid], np.flatnonzero(valid))
    traced.snapshot(pose(2.))
    assert np.isnan(traced.query_sources['scan_stamp_s']).all()
    assert (traced.query_sources['source_cell'] == -1).all()
