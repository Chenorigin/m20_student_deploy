"""PointCloud2 reader honoring field offsets, byte order, point/row padding."""
import numpy as np

TYPES = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}


def validate_cloud_time(stamp, now, runtime, offsets):
    """Validate acquisition times without rewriting the wire header or point times.

    A fused scan may start over 200ms ago while its final returns are fresh.
    Measurement aging still uses the oldest point in Pipeline, independently.
    """
    if not np.isfinite([stamp, now]).all():
        raise ValueError("Nonfinite cloud timestamp")
    first = last = float(stamp)
    if offsets is not None:
        times = np.asarray(offsets)
        if not np.isfinite(times).all():
            raise ValueError("Nonfinite per-point time")
        if times.size:
            lo, hi = float(np.min(times)), float(np.max(times))
            span = runtime["scan_duration_s"]
            if max(abs(lo), abs(hi), hi-lo) > span:
                raise ValueError("Per-point offsets outside declared scan duration")
            first, last = stamp+lo, stamp+hi
    if (now-last > runtime.get("cloud_input_timeout_s", runtime["input_timeout_s"]) or
            max(last, stamp)-now > runtime["future_tolerance_s"]):
        raise ValueError("Cloud timestamp stale/future: check clock synchronization")
    return first, last


def read_cloud(msg, max_points=200000, max_bytes=16000000, time_field="", time_scale=1., time_mode="relative"):
    if time_mode not in ("relative", "absolute"):
        raise ValueError("point_time_mode must be relative or absolute")
    count = msg.width*msg.height
    if msg.width < 0 or msg.height <= 0 or count > max_points or len(msg.data) > max_bytes:
        raise ValueError("Cloud dimensions/budget exceeded")
    if msg.point_step <= 0 or msg.row_step < msg.width*msg.point_step or len(msg.data) != msg.height*msg.row_step:
        raise ValueError("Malformed cloud strides/buffer")
    fields = {f.name: f for f in msg.fields}
    if len(fields) != len(msg.fields):
        raise ValueError("Duplicate point fields")
    wanted = ["x", "y", "z"] + ([time_field] if time_field else [])
    values = []
    for name in wanted:
        f = fields.get(name)
        if f is None or f.count != 1 or f.datatype not in TYPES or (name in ("x", "y", "z") and f.datatype not in (7, 8)):
            raise ValueError("Unsupported/missing point field: " + name)
        dtype = np.dtype((">" if msg.is_bigendian else "<")+TYPES[f.datatype])
        if f.offset < 0 or f.offset+dtype.itemsize > msg.point_step:
            raise ValueError("Point field exceeds point_step")
        if count == 0:
            values.append(np.empty(0, dtype=np.float64))
            continue
        view = np.ndarray((msg.height, msg.width), dtype=dtype, buffer=msg.data,
                          offset=f.offset, strides=(msg.row_step, msg.point_step))
        values.append(view.reshape(-1).astype(np.float64))
    xyz = np.column_stack(values[:3])
    offsets = values[3]*time_scale if time_field else None
    if offsets is not None and time_mode == "absolute":
        header = msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9
        if not np.isfinite(header):
            raise ValueError("Nonfinite scan header")
        offsets -= header
    return xyz, offsets


def mapper_input_mask(points, frame, cfg):
    """Keep every original return inside the calibrated front sensor fan.

    This selects the mapper's raw input, independently of the full LIO branch.
    Range, self filters, motion ROI and deskew remain in Pipeline, as does the
    current-grid front output mask. No stride or resampling is applied here.
    """
    from .geometry import rotation
    from .visibility import front_map_mask

    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected cloud XYZ for mapper input filtering")
    result = np.isfinite(points).all(1)
    if cfg["runtime"].get("mapping_region", "all") != "front":
        return result
    ext = cfg["lidar_extrinsics"].get(frame)
    if ext is None:
        raise ValueError("Uncalibrated cloud frame: " + frame)
    transform = rotation(ext["quaternion_xyzw"])
    with np.errstate(invalid="ignore"):
        body_xy = np.column_stack([
            points[:, 0]*row[0] + points[:, 1]*row[1] +
            points[:, 2]*row[2] + ext["translation_m"][axis]
            for axis, row in enumerate(transform[:2])])
        result &= front_map_mask(body_xy, cfg["visibility"])
    return result
