"""Pure CPU mapping between training joint coordinates and the reference M20 SDK.

This module prepares values only. It does not create a ROS command publisher.
SDK interface getters/setters already apply direction/offset; use the corrected
APIs directly, and never apply raw conversion a second time.
"""
import json
from pathlib import Path
import numpy as np
from .contract import JOINT_NAMES, finite_vector


def load_robot_config(path):
    cfg = json.loads(Path(path).read_text())
    joint = cfg["joints"]
    if tuple(joint["policy_order"]) != JOINT_NAMES or len(joint["sdk_order"]) != 16 or set(joint["sdk_order"]) != set(JOINT_NAMES):
        raise ValueError("Joint order/mapping mismatch")
    for name in ("sdk_direction", "sdk_initial_offset_deg", "default_position_policy_rad",
                 "action_scale_policy", "kp_sdk", "kd_sdk"):
        finite_vector(joint[name], 16, name)
    if not np.all(np.isin(joint["sdk_direction"], [-1, 1])):
        raise ValueError("SDK directions must be +/-1")
    if np.any(np.asarray(joint["action_scale_policy"]) <= 0) or np.any(np.asarray(joint["kp_sdk"]) < 0) or np.any(np.asarray(joint["kd_sdk"]) < 0):
        raise ValueError("Invalid action scale or PD gains")
    for name, shape in (("normalized_action_clip", (2,)), ("processed_target_clip_policy", (16, 2))):
        bounds = np.asarray(cfg["policy"][name])
        if bounds.shape != shape or not np.isfinite(bounds).all() or np.any(bounds[..., 0] >= bounds[..., 1]):
            raise ValueError("Invalid clipping bounds: " + name)
    for name in ("joint_target_position_clip_enabled", "wheel_target_velocity_clip_enabled"):
        if name in cfg["policy"] and type(cfg["policy"][name]) is not bool:
            raise ValueError(name + " must be a boolean")
    return cfg


def indices(cfg):
    policy, sdk = cfg["joints"]["policy_order"], cfg["joints"]["sdk_order"]
    return np.array([sdk.index(n) for n in policy]), np.array([policy.index(n) for n in sdk])


def raw_state_to_policy(raw_q_sdk, raw_dq_sdk, cfg, effective_offset_deg=None):
    joint = cfg["joints"]
    offset = finite_vector(joint["sdk_initial_offset_deg"] if effective_offset_deg is None else effective_offset_deg, 16, "offset")
    direction = np.asarray(joint["sdk_direction"])
    q = finite_vector(raw_q_sdk, 16, "raw position")*direction+np.deg2rad(offset)
    dq = finite_vector(raw_dq_sdk, 16, "raw velocity")*direction
    policy_from_sdk, _ = indices(cfg)
    return q[policy_from_sdk], dq[policy_from_sdk]


def corrected_state_to_policy(q_sdk, dq_sdk, cfg):
    policy_from_sdk, _ = indices(cfg)
    return finite_vector(q_sdk, 16, "corrected position")[policy_from_sdk], finite_vector(dq_sdk, 16, "corrected velocity")[policy_from_sdk]


def action_to_sdk_matrix(action, cfg):
    """16x5 [kp,q,kd,dq,tau_ff] for M20Interface.SetJointCommand.

    Motor position/rate hard-limit checks and physical control arbitration must
    happen in the actual controller before applying this prepared matrix.
    """
    a = finite_vector(action, 16, "policy action")
    lo, hi = cfg["policy"]["normalized_action_clip"]
    a = np.clip(a, lo, hi)
    j = cfg["joints"]
    targets = np.asarray(j["default_position_policy_rad"])+a*np.asarray(j["action_scale_policy"])
    processed_clip = np.asarray(cfg["policy"]["processed_target_clip_policy"])
    apply_clip = np.array([cfg["policy"].get("joint_target_position_clip_enabled", True)]*12 +
                          [cfg["policy"].get("wheel_target_velocity_clip_enabled", True)]*4)
    targets = np.where(apply_clip, np.clip(targets, processed_clip[:, 0], processed_clip[:, 1]), targets)
    _, sdk_from_policy = indices(cfg)
    targets = targets[sdk_from_policy]
    out = np.zeros((16, 5), dtype=np.float32)
    out[:, 0], out[:, 2] = j["kp_sdk"], j["kd_sdk"]
    for i, name in enumerate(j["sdk_order"]):
        out[i, 3 if "wheel" in name else 1] = targets[i]
    return out


def sdk_matrix_to_raw(matrix, cfg, effective_offset_deg=None):
    """For direct drdds serialization only; do not use before SDK SetJointCommand."""
    result = np.asarray(matrix, dtype=np.float64).copy()
    if result.shape != (16, 5) or not np.isfinite(result).all():
        raise ValueError("Expected finite SDK matrix [16,5]")
    j = cfg["joints"]
    offset = finite_vector(j["sdk_initial_offset_deg"] if effective_offset_deg is None else effective_offset_deg, 16, "offset")
    d = np.asarray(j["sdk_direction"])
    result[:, 1] = (result[:, 1]-np.deg2rad(offset))*d
    result[:, 3] *= d
    result[:, 4] *= d
    return result
