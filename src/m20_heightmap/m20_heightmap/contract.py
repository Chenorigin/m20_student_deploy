"""Pinned to rl_training/m20_student_v{0,1}/contract.py; checked by audit_training.py."""
from collections import deque
import json
from pathlib import Path
import numpy as np

ROWS, COLS, N = 11, 17, 187
RESOLUTION, HEIGHT_OFFSET = 0.1, 0.5
POLICY_DT = 0.02
YY, XX = np.meshgrid(np.arange(ROWS)*RESOLUTION-.5,
                     np.arange(COLS)*RESOLUTION-.8, indexing="ij")
GRID = np.column_stack((XX.ravel(), YY.ravel()))


def contract(profile, history_steps=None):
    if profile not in ("v0", "v1"):
        raise ValueError("profile must be v0 or v1")
    steps = (1 if profile == "v0" else 4) if history_steps is None else history_steps
    if type(steps) is not int or not 1 <= steps <= 16 or (profile == "v0" and steps != 1):
        raise ValueError("v0 requires one frame; v1 requires 1..16 total frames")
    return {"profile": profile, "history_steps": steps, "input_dim": 618+57*(steps-1)}


def validate_model_contract(path, expected):
    actual = json.loads(Path(path).read_text())
    version = "m20_student_v0_618" if expected["profile"] == "v0" else "m20_student_v1"
    slices = {"proprio": [0, 57], "height": [57, 244], "validity": [244, 431], "confidence": [431, 618]}
    if expected["history_steps"] > 1:
        slices["past_proprio"] = [618, expected["input_dim"]]
    checks = {
        "version": version, "student_observation_dim": expected["input_dim"],
        "action_dim": 16, "slices": slices,
        "grid": {"rows_y": ROWS, "columns_x": COLS, "resolution_m": RESOLUTION,
                 "x_range_m": [-.8, .8], "y_range_m": [-.5, .5],
                 "order": "y rows, x columns, ascending", "alignment": "gravity aligned, robot yaw"},
        "height": {"formula": "clip(base_z - terrain_z - offset, min, max)", "offset_m": HEIGHT_OFFSET, "clip": [-1., 1.]},
        "unknown": {"height": 0., "validity": 0., "confidence": 0.},
        "normalization": "teacher proprio scales retained; no running observation normalization",
    }
    if expected["profile"] == "v1":
        checks.update(history_steps=expected["history_steps"], policy_dt_s=POLICY_DT,
                      history="past noisy proprio only; oldest to newest; zero padded on reset; current proprio first")
    for key, value in checks.items():
        if actual.get(key) != value:
            raise ValueError("Model observation contract mismatch: " + key)
    return actual


def finite_vector(value, size, name):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError(name + " must be a finite vector of length " + str(size))
    return value


class ObservationPacker:
    """Call exactly once per policy tick, never once per LiDAR packet."""
    def __init__(self, profile, history_steps=None):
        self.spec = contract(profile, history_steps)
        self.reset()

    def reset(self, clear_watermark=True):
        self.history = deque([np.zeros(57, dtype=np.float32)
                              for _ in range(self.spec["history_steps"]-1)],
                             maxlen=self.spec["history_steps"]-1)
        if clear_watermark:
            self.last_step = None
            self.last_stamp = None

    def pack(self, proprio, terrain, step, stamp):
        p = finite_vector(proprio, 57, "proprio")
        t = np.asarray(terrain, dtype=np.float64)
        if t.shape != (3, N) or not np.isfinite(t).all():
            raise ValueError("terrain must be finite [height, validity, confidence], shape (3,187)")
        if not np.all((t[1] == 0) | (t[1] == 1)) or np.any(np.abs(t[0]) > 1) or np.any((t[2] < 0) | (t[2] > 1)):
            raise ValueError("Invalid terrain channel ranges")
        if np.any(t[0, t[1] == 0] != 0) or np.any(t[2, t[1] == 0] != 0):
            raise ValueError("Unknown cells require h=m=c=0")
        if not np.isfinite(stamp) or type(step) is not int or step < 0:
            raise ValueError("Invalid policy tick")
        if self.last_step is not None:
            if step <= self.last_step or stamp <= self.last_stamp:
                self.reset(clear_watermark=False)
                raise ValueError("Repeated/reordered policy tick")
            if step != self.last_step+1 or abs(stamp-self.last_stamp-POLICY_DT) > .005:
                self.reset(clear_watermark=False)
                self.last_step, self.last_stamp = step, stamp
                raise ValueError("Policy tick missing/repeated or not 20ms; history reset")
        out = np.concatenate((p, t.ravel(), *self.history)).astype(np.float32)
        self.history.append(p.astype(np.float32))
        self.last_step, self.last_stamp = step, stamp
        return out


JOINT_NAMES = tuple(f"{leg}_{joint}_joint" for leg in ("fl", "fr", "hl", "hr")
                    for joint in ("hipx", "hipy", "knee")) + tuple(f"{leg}_wheel_joint" for leg in ("fl", "fr", "hl", "hr"))


def make_proprio(gyro_body, gravity_body, command, joint_position, joint_velocity,
                 default_joint_position, previous_issued_action):
    """Inputs in SI and JOINT_NAMES order. default pose MUST come from the trained policy."""
    gyro = finite_vector(gyro_body, 3, "gyro")
    gravity = finite_vector(gravity_body, 3, "gravity")
    if abs(np.linalg.norm(gravity)-1) > .02:
        raise ValueError("Projected gravity must have unit norm, not m/s^2")
    command = finite_vector(command, 3, "command")
    pos = finite_vector(joint_position, 16, "joint_position") - finite_vector(default_joint_position, 16, "default pose")
    pos[12:] = 0.
    vel = finite_vector(joint_velocity, 16, "joint_velocity")
    act = finite_vector(previous_issued_action, 16, "previous issued action")
    return np.concatenate((np.clip(gyro, -100, 100)*.25, gravity, command,
                           np.clip(pos, -100, 100), np.clip(vel, -100, 100)*.05,
                           np.clip(act, -100, 100))).astype(np.float32)
