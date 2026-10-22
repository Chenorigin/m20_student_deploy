"""AOS receive-time validation and aging; shares the physical quality equation."""
import numpy as np
from .contract import N
from .estimator import age_confidence


def receive_terrain(msg, now, cfg):
    age_limit = cfg["aos"]["max_alignment_age_s"]
    stamp = msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9
    elapsed = now-stamp
    if msg.schema_version != 2 or msg.profile != cfg["profile"] or msg.header.frame_id != cfg["frames"]["output"]:
        raise ValueError("Terrain ABI/profile/frame mismatch")
    if not np.isfinite([stamp, now, elapsed]).all() or elapsed < -cfg["runtime"]["future_tolerance_s"]:
        raise ValueError("Terrain/query pose stale or from future")
    arrays = [np.asarray(getattr(msg, name), dtype=np.float64) for name in ("height", "validity", "confidence", "variance", "age")]
    if any(a.shape != (N,) or not np.isfinite(a).all() for a in arrays):
        raise ValueError("Terrain arrays malformed/nonfinite")
    h, m, c, var, age = arrays
    if np.any(np.abs(h) > 1) or not np.all((m == 0) | (m == 1)) or np.any((c < 0) | (c > 1)) or np.any(var < 0):
        raise ValueError("Terrain ranges malformed")
    if np.any(h[m == 0] != 0) or np.any(c[m == 0] != 0) or np.any(age[m == 1] < 0):
        raise ValueError("Terrain unknown/age semantics malformed")
    coverage_hold = cfg["aos"].get("coverage_hold_enabled", True)
    timing_hold = cfg["aos"].get("terrain_timing_hold_enabled", True)
    for name, enabled in (("coverage_hold_enabled", coverage_hold), ("terrain_timing_hold_enabled", timing_hold)):
        if type(enabled) is not bool:
            raise ValueError(name + " must be bool")
    from .estimator import EstimatorConfig
    quality = EstimatorConfig(**cfg["estimator"])
    if elapsed > age_limit:
        if timing_hold:
            raise ValueError("Terrain/query pose stale or from future")
        h, valid, c = np.zeros(N), np.zeros(N, dtype=bool), np.zeros(N)
    else:
        elapsed = max(0., elapsed)
        valid = (m == 1) & (age+elapsed <= quality.max_map_age_s)
        c = age_confidence(c, age, elapsed, valid, quality)
        h, c = np.where(valid, h, 0.), np.where(valid, c, 0.)
    if coverage_hold and valid.mean() < cfg["aos"]["min_valid_fraction"]:
        raise ValueError("Insufficient valid terrain coverage")
    return np.stack((h, valid.astype(float), c)).astype(np.float32)
