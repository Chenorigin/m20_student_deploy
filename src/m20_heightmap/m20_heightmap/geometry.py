"""All quaternions are xyzw; rotations map child/body to parent/world."""
from dataclasses import dataclass
from collections import deque
import numpy as np
from .contract import finite_vector


def rotation(q):
    q = finite_vector(q, 4, "quaternion")
    norm = np.linalg.norm(q)
    if abs(norm-1) > .02:
        raise ValueError("Quaternion norm outside tolerance")
    x, y, z, w = q/norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def yaw_rotation(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


@dataclass
class Pose:
    stamp: float
    position: np.ndarray
    quaternion: np.ndarray
    # Diagonal covariance: x,y,z in world, roll,pitch,yaw in body (explicit adapter contract).
    variance: np.ndarray
    speed: float = 0.
    angular_speed: float = 0.

    def __post_init__(self):
        self.position = finite_vector(self.position, 3, "position")
        self.quaternion = finite_vector(self.quaternion, 4, "quaternion")
        self.R = rotation(self.quaternion)
        self.quaternion = self.quaternion/np.linalg.norm(self.quaternion)
        self.variance = finite_vector(self.variance, 6, "pose variance")
        if np.any(self.variance < 0) or not np.isfinite([self.stamp, self.speed, self.angular_speed]).all() or min(self.speed, self.angular_speed) < 0:
            raise ValueError("Invalid pose time, variance or speed")
        self.yaw = float(np.arctan2(self.R[1, 0], self.R[0, 0]))


class PoseBuffer:
    def __init__(self, maxlen=256, max_gap=.06):
        self.poses = deque(maxlen=maxlen)
        self.max_gap = max_gap

    def clear(self):
        self.poses.clear()

    def add(self, pose):
        if self.poses and pose.stamp <= self.poses[-1].stamp:
            raise ValueError("Nonmonotonic pose stamp")
        self.poses.append(pose)

    def at(self, stamp):
        if not self.poses or stamp < self.poses[0].stamp or stamp > self.poses[-1].stamp:
            raise ValueError("Pose does not bracket capture time; no extrapolation")
        stamps = np.array([p.stamp for p in self.poses])
        i = int(np.searchsorted(stamps, stamp))
        b = self.poses[i]
        if b.stamp == stamp:
            return b
        a = self.poses[i-1]
        if b.stamp-a.stamp > self.max_gap:
            raise ValueError("Pose interpolation gap too large")
        t = (stamp-a.stamp)/(b.stamp-a.stamp)
        qb = b.quaternion.copy()
        dot = np.dot(a.quaternion, qb)
        if dot < 0:
            qb, dot = -qb, -dot
        theta = np.arccos(np.clip(dot, -1, 1))
        q = ((1-t)*a.quaternion + t*qb) if theta < 1e-5 else (np.sin((1-t)*theta)*a.quaternion+np.sin(t*theta)*qb)/np.sin(theta)
        q /= np.linalg.norm(q)
        return Pose(stamp, (1-t)*a.position+t*b.position, q,
                    np.maximum(a.variance, b.variance), max(a.speed, b.speed), max(a.angular_speed, b.angular_speed))


def to_world(points, pose, translation, quaternion):
    body = np.asarray(points) @ rotation(quaternion).T + np.asarray(translation)
    return body @ pose.R.T + pose.position


def deskew(points, offsets, capture_pose, poses, translation, quaternion, bin_s=.005, *, return_quality=False):
    """Pose interpolation at bounded temporal bin centers; offsets in seconds relative to header."""
    offsets = np.asarray(offsets, dtype=np.float64)
    if offsets.shape != (len(points),) or not np.isfinite(offsets).all():
        raise ValueError("Invalid per-point time offsets")
    world = np.empty_like(points, dtype=np.float64)
    bins = np.floor(offsets/bin_s).astype(np.int64)
    variance = capture_pose.variance.copy()
    speed, rate = capture_pose.speed, capture_pose.angular_speed
    # Number of bins is bounded by the caller's scan duration limit.
    for key in np.unique(bins):
        mask = bins == key
        when = capture_pose.stamp + float(np.mean(offsets[mask]))
        pose = poses.at(when)
        world[mask] = to_world(points[mask], pose, translation, quaternion)
        variance = np.maximum(variance, pose.variance)
        speed, rate = max(speed, pose.speed), max(rate, pose.angular_speed)
    if return_quality:
        # Translation covariance is in world axes. Angular covariance is in each
        # pose's body axes: an isotropic envelope bounds their rotated diagonals
        # in the anchor frame without assuming their axes remain aligned.
        variance[3:] = np.max(variance[3:])
        quality = Pose(capture_pose.stamp, capture_pose.position, capture_pose.quaternion,
                       variance, speed, rate)
        return world, quality
    return world
