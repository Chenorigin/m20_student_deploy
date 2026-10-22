"""One isolated CPU job; no ROS participant, publisher or map shared with callbacks."""
import os
import time
from collections import deque
from types import SimpleNamespace
import numpy as np
from .geometry import Pose
from .runtime import Pipeline

_clock = None


def pack_job_inputs(msg, poses):
    """Copy exact wire bytes and parent-validated poses into a small object graph."""
    matrix = np.empty((len(poses), 26), dtype=np.float64)
    for i, p in enumerate(poses):
        matrix[i] = (p.stamp, *p.position, *p.quaternion, *p.variance,
                     p.speed, p.angular_speed, *p.R.ravel(), p.yaw)
    cloud = ((msg.header.stamp.sec, msg.header.stamp.nanosec, msg.header.frame_id),
             (msg.height, msg.width, msg.point_step, msg.row_step, msg.is_bigendian),
             tuple((f.name, f.offset, f.datatype, f.count) for f in msg.fields), bytes(msg.data))
    return cloud, matrix


def unpack_job_inputs(packed):
    cloud, matrix = packed
    matrix = np.asarray(matrix)
    if (matrix.dtype != np.float64 or matrix.ndim != 2 or matrix.shape[1] != 26 or
            not 0 < len(matrix) <= 4096 or not np.isfinite(matrix).all() or
            np.any(matrix[:, 8:16] < 0) or np.any(np.diff(matrix[:, 0]) <= 0)):
        raise ValueError('Invalid packed terrain poses')
    q=matrix[:, 4:8]
    norms=np.linalg.norm(q,axis=1)
    if np.any(np.abs(norms-1)>.02):
        raise ValueError('Invalid packed terrain quaternion')
    x,y,z,w=(q/norms[:,None]).T
    expected=np.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                       2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                       2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),axis=1)
    yaw=np.arctan2(matrix[:,19],matrix[:,16])
    error=np.arctan2(np.sin(matrix[:,25]-yaw),np.cos(matrix[:,25]-yaw))
    if not np.allclose(matrix[:,16:25],expected,atol=1e-12,rtol=1e-12) or np.any(np.abs(error)>1e-12):
        raise ValueError('Inconsistent packed terrain orientation')
    # Preserve every existing value exactly; do not re-normalize quaternions or
    # recompute matrices. These internal values were validated by the parent.
    poses = deque(maxlen=len(matrix))
    for row in matrix:
        p = Pose.__new__(Pose)
        p.stamp, p.position, p.quaternion = float(row[0]), row[1:4], row[4:8]
        p.variance, p.speed, p.angular_speed = row[8:14], float(row[14]), float(row[15])
        p.R, p.yaw = row[16:25].reshape(3, 3), float(row[25])
        poses.append(p)
    header, shape, fields, data = cloud
    msg = SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=header[0], nanosec=header[1]), frame_id=header[2]),
        height=shape[0], width=shape[1], point_step=shape[2], row_step=shape[3], is_bigendian=shape[4],
        fields=[SimpleNamespace(name=f[0], offset=f[1], datatype=f[2], count=f[3]) for f in fields], data=data)
    return msg, poses


def estimate_packed_patch(cfg, packed, scan_stamp, now, last_stamp, clock=None):
    msg, poses = unpack_job_inputs(packed)
    return estimate_patch(cfg, msg, scan_stamp, now, poses, last_stamp, clock)


def initialize_worker(cpu):
    # spawn starts a clean interpreter instead of forking live DDS threads.
    # The worker uses the same physical system clock as the live non-sim node.
    from rclpy.clock import Clock, ClockType
    global _clock
    os.sched_setaffinity(0, {cpu} if isinstance(cpu, int) else set(cpu))
    _clock = Clock(clock_type=ClockType.SYSTEM_TIME)


def estimate_patch(cfg, msg, scan_stamp, now, poses, last_stamp, clock=None):
    if clock is None:
        if _clock is None:
            raise RuntimeError('Terrain process clock not initialized')
        clock = lambda: _clock.now().nanoseconds*1e-9
    job = Pipeline(cfg)
    job.poses.poses = poses
    job.mapper.last_stamp = last_stamp
    started = time.monotonic()
    job.ingest_message(msg, scan_stamp, now, processing_now=clock)
    return job.mapper.patches[-1], job.mapper.stats, job.last_scan, (time.monotonic()-started)*1000
