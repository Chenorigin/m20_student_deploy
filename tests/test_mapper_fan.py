import copy
import math

import numpy as np

from m20_heightmap.cloud import mapper_input_mask
from m20_heightmap.geometry import rotation


def fan_config():
    return {
        'runtime': {'mapping_region': 'front'},
        'lidar_extrinsics': {'lidar': {
            'translation_m': [.13, -.07, .09],
            'quaternion_xyzw': [0., 0., math.sin(.3), math.cos(.3)]}},
        'visibility': {'horizontal_fov_deg': 135., 'sensors': [
            {'position_base_m': [.32028, 0., -.013], 'yaw_rad': 0.},
            {'position_base_m': [-.32028, 0., -.013], 'yaw_rad': math.pi}]}}


def in_lidar(body, cfg):
    ext = cfg['lidar_extrinsics']['lidar']
    return (body - np.asarray(ext['translation_m'])) @ rotation(ext['quaternion_xyzw'])


def test_raw_fan_uses_sensor_origin_and_extrinsics_without_sampling():
    cfg = fan_config()
    angles = np.radians([-68., -67.5, -60., -30., 0., 30., 60., 67.5, 68., 180.])
    # Repeated returns and distinct heights must all survive inside the fan.
    angles = np.repeat(angles, 17)
    body = np.column_stack((.32028 + np.cos(angles), np.sin(angles),
                            np.arange(len(angles)) * .001))
    points = in_lidar(body, cfg)
    before = points.copy()
    keep = mapper_input_mask(points, 'lidar', cfg)
    expected = np.repeat([False, True, True, True, True, True, True, True, False, False], 17)
    np.testing.assert_array_equal(keep, expected)
    np.testing.assert_array_equal(points, before)
    np.testing.assert_array_equal(np.flatnonzero(keep), np.flatnonzero(expected))


def test_raw_fan_rejects_near_body_points_outside_sensor_fan_and_nonfinite():
    cfg = fan_config()
    body = np.array([[.2, 0., 0.], [.5, 1., 0.], [.8, .1, -.2],
                     [.32028, 0., 0.], [np.nan, 0., 0.]])
    np.testing.assert_array_equal(mapper_input_mask(in_lidar(body, cfg), 'lidar', cfg),
                                  [False, False, True, False, False])
    all_cfg = copy.deepcopy(cfg)
    all_cfg['runtime']['mapping_region'] = 'all'
    np.testing.assert_array_equal(mapper_input_mask(in_lidar(body, cfg), 'lidar', all_cfg),
                                  [True, True, True, True, False])
