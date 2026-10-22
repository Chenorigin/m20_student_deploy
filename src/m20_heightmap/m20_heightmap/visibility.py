"""Physical front/rear horizontal view; unknown space is never a flat surface."""
import math
import numpy as np


def validate_visibility(cfg):
    if set(cfg) != {'horizontal_fov_deg', 'sensors'}:
        raise ValueError('Invalid visibility keys')
    fov = cfg['horizontal_fov_deg']
    if isinstance(fov, bool) or not isinstance(fov, (int, float)) or not math.isfinite(fov) or not 0 < fov <= 360:
        raise ValueError('Invalid horizontal field of view')
    if not isinstance(cfg['sensors'], list) or len(cfg['sensors']) != 2:
        raise ValueError('Declare physical front and rear sensors')
    for sensor in cfg['sensors']:
        if set(sensor) != {'position_base_m', 'yaw_rad'}:
            raise ValueError('Invalid visibility sensor keys')
        p=np.asarray(sensor['position_base_m'], dtype=float)
        if p.shape != (3,) or not np.isfinite(p).all() or not math.isfinite(sensor['yaw_rad']):
            raise ValueError('Invalid visibility sensor transform')
    return cfg


def visible_body(points, cfg):
    p=np.asarray(points)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError('Expected body XYZ for visibility')
    finite=np.isfinite(p).all(1)
    if cfg['horizontal_fov_deg'] == 360:
        return finite
    threshold=math.cos(math.radians(cfg['horizontal_fov_deg']/2))
    result=np.zeros(len(p), dtype=bool)
    for sensor in cfg['sensors']:
        delta=p[:, :2]-np.asarray(sensor['position_base_m'])[:2]
        horizontal=np.hypot(delta[:, 0],delta[:, 1])
        yaw=sensor['yaw_rad']
        forward=delta[:, 0]*math.cos(yaw)+delta[:, 1]*math.sin(yaw)
        # Equivalent angular boundary tolerance to the training visibility test.
        result |= finite & (horizontal > 1e-8) & (forward >= horizontal*(threshold-1e-7))
    return result


def front_map_mask(points, cfg):
    """Current yaw-grid mask matching the simulator's front_only profile."""
    front = [s for s in cfg['sensors'] if math.cos(s['yaw_rad']) > 0.]
    if len(front) != 1:
        raise ValueError('Front map output requires one forward-facing sensor')
    sensor = front[0]
    delta = np.asarray(points)-np.asarray(sensor['position_base_m'])[:2]
    yaw = sensor['yaw_rad']
    forward = delta[:, 0]*math.cos(yaw)+delta[:, 1]*math.sin(yaw)
    lateral = -delta[:, 0]*math.sin(yaw)+delta[:, 1]*math.cos(yaw)
    angle = np.abs(np.arctan2(lateral, forward))
    return ((np.linalg.norm(delta, axis=1) > 1e-6) & (forward > 0.) &
            (angle <= math.radians(cfg['horizontal_fov_deg']/2)+1e-7))
