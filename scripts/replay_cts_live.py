#!/usr/bin/env python3
"""Prepare or verify isolated synthetic replay of the actual C++ CTS runner."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def prepare(path, config, scenario="nominal"):
    cfg = json.loads(config.read_text())
    binding = dict(source_id="offline_fixture", hardware_serial="offline_fixture",
                   world_frame=cfg["frames"]["world"], binding_sha256="a" * 64,
                   publisher_node="offline_fixture")
    frames = []
    for tick in range(12):
        yaw = tick * .01
        feedback = dict(schema="m20_cts_landing_feedback.v1", joint_order="sdk16",
                        leg_order="FL_FR_HL_HR", jacobian_columns="policy_leg_joints",
                        kinematics_valid=True, **{k: v for k, v in binding.items() if k != "publisher_node"},
                        session=1, sequence=tick + 1, capture_monotonic_ns=1,
                        joint_position_sdk_rad=[0.] * 16, joint_velocity_sdk_rad_s=[0.] * 16,
                        contact_support_force_N=[0., 40., 40., 40.], contact_valid=[True] * 4,
                        contact_capture_monotonic_ns=[1] * 4,
                        support_normal_velocity_world_m_s=[-.4, 0., 0., 0.],
                        support_position_world_m=[.5, 0., .025] * 4,
                        point_jacobian_world_m_per_rad=(np.eye(3) * .15).ravel().tolist() * 4,
                        base_position_world_m=[0., 0., .6], base_yaw_world_rad=yaw,
                        terrain_query=dict(valid=True, session=1, sequence=1, stamp_ros_s=1.,
                                           position_world_m=[0., 0., .6], yaw_world_rad=yaw))
        frames.append(dict(yaw=yaw, yaw_rate=.03, height=[.1] * 187,
                           raw_command=[0., 0., 0.] if tick < 4 else [.4, 0., 0.],
                           feedback=feedback))
        if tick >= 8 and scenario == "expired_source":
            frames[-1].update(terrain_session=2, query_age_s=.2)
        elif tick >= 8 and scenario == "inference_expiry":
            frames[-1].update(query_age_s=.11, inference_delay_s=.015)
    document = dict(scope="synthetic offline fixture only", scenario=scenario, binding=binding, frames=frames)
    if cfg["aos"].get("cts_execution_mode", "landing") == "nominal_pd":
        del document["binding"]
        for frame in frames:
            del frame["feedback"]
    path.write_text(json.dumps(document) + "\n")


def verify(report, model_dir):
    from m20_heightmap.cts import CTSInference
    text = report.read_text().strip().splitlines()
    rows = json.loads(text[-1])
    if rows["hardware_commands"] != 0 or rows["command_publishers"] != 0:
        raise ValueError("Replay scope was not read-only")
    frames = rows["frames"]
    if len(frames) != 12:
        raise ValueError("Expected the fixed twelve replay frames")
    model = CTSInference(model_dir / "policy.onnx", model_dir / "observation_contract.json")
    errors = []
    history = [np.zeros(57, dtype="f")] * 3
    raw_effective_differ = False
    modified_substeps = 0
    mode = rows.get("execution_mode", "landing")
    if mode not in ("landing", "nominal_pd"):
        raise ValueError("Unknown replay execution mode")
    nominal_pd = mode == "nominal_pd"
    expected_checks = {"out_of_bounds_target_rejected", "nonfinite_target_rejected",
                       "stale_body_rejected", "nonfinite_body_rejected", "missed_policy_tick_latches_hold",
                       "hold_preserves_published_target_and_pd", "hold_stops_wheels_and_feedforward",
                       "hold_does_not_advance_previous_action", "stale_body_rejected_after_hold"}
    if nominal_pd and (set(rows.get("checks", {})) != expected_checks or
                       any(value is not True for value in rows["checks"].values())):
        raise ValueError("Nominal PD boundary/HOLD checks missing or failed")
    settings = json.loads((model_dir / "observation_contract.json").read_text())["quiet_execution_control"]["settings"]
    for i, row in enumerate(frames):
        obs = np.asarray(row["obs"], dtype="f")
        raw = np.asarray(row["raw_command"], dtype="f")
        if obs.shape != (789,) or raw.shape != (3,):
            raise ValueError("Actual runner returned an incorrect input ABI")
        expected_raw = [0., 0., 0.] if i < 4 else [.4, 0., 0.]
        np.testing.assert_allclose(raw, expected_raw, atol=1e-7, rtol=0)
        np.testing.assert_allclose(obs[618:], np.concatenate(history), atol=2e-6, rtol=0)
        action = model(obs[None], raw[None])[0]
        actual = np.asarray(row["nominal_action"], dtype="f")
        np.testing.assert_allclose(actual, action, atol=2e-5, rtol=2e-5)
        errors.append(float(np.max(np.abs(action - actual))))
        steps = np.asarray(row["substeps"], dtype="f")
        nominal = np.asarray(row["nominal_command"], dtype="f")
        if steps.shape != (4, 16, 5) or not np.isfinite(steps).all():
            raise ValueError("Missing/nonfinite actual execution substeps")
        wheels = np.arange(16) % 4 == 3
        np.testing.assert_array_equal(steps[:, wheels, 3], np.tile(nominal[wheels, 3], (4, 1)))
        leg = ~wheels
        kp = steps[:, leg, 0] / nominal[leg, 0]
        if (kp < settings["landing_kp_scale"] - 2e-6).any() or (kp > 1. + 2e-6).any():
            raise ValueError("Landing gains compounded or exceeded saved limits")
        if (np.abs(steps[:, leg, 1] - nominal[leg, 1]) > settings["max_joint_correction_rad"] + 2e-6).any():
            raise ValueError("Landing correction exceeded saved bounds")
        modified_substeps += int(np.count_nonzero(np.any(np.abs(steps[:, leg, :3] - nominal[leg, :3]) > 1e-6, axis=(1, 2))))
        if nominal_pd:
            np.testing.assert_array_equal(steps, np.tile(nominal, (4, 1, 1)))
            np.testing.assert_array_equal(row["landing_counts"], np.zeros(6, dtype=int))
        elif row["landing_counts"][0] != (i + 1) * 4 * 4:
            raise ValueError("Live runner did not execute four real controller substeps per policy tick")
        if rows.get("scenario") in ("expired_source", "inference_expiry") and i >= 8:
            np.testing.assert_array_equal(obs[57:618], np.zeros(561))
        history = history[1:] + [obs[:57].copy()]
        raw_effective_differ |= bool(i < 4 and obs[8] != raw[2])
    if not raw_effective_differ:
        raise ValueError("Fixture did not exercise raw zero with effective yaw feedback")
    if not nominal_pd and (modified_substeps == 0 or frames[-1]["landing_counts"][1] == 0):
        raise ValueError("Live landing execution was a no-op; nominal-only replay rejected")
    if not nominal_pd and rows.get("scenario") == "expired_source" and frames[-1]["landing_counts"][2] <= frames[7]["landing_counts"][2]:
        raise ValueError("Expired new source did not invalidate the old terrain epoch")
    return dict(passed=True, frames=len(frames), execution_substeps=4 * len(frames),
                max_abs_error=max(errors), raw_zero_effective_yaw_nonzero=True,
                history_advances_per_policy_tick=1, hardware_commands=0, hardware_ready=False,
                modified_substeps=modified_substeps, landing_counts=frames[-1]["landing_counts"],
                execution_mode=rows.get("execution_mode", "landing"),
                checks=rows.get("checks", {}),
                scenario=rows.get("scenario", "nominal"),
                onnx_sha256=hashlib.sha256((model_dir / "policy.onnx").read_bytes()).hexdigest(),
                scope="actual C++ runner synthetic replay; no physical contact or actuator validation")


def main():
    ap = argparse.ArgumentParser()
    operation = ap.add_mutually_exclusive_group(required=True)
    operation.add_argument("--prepare", type=Path)
    operation.add_argument("--verify", type=Path)
    ap.add_argument("--config", type=Path, default=Path("runtime/heightmap_v1.local.json"))
    ap.add_argument("--model-dir", type=Path, default=Path("models/v1"))
    ap.add_argument("--scenario", choices=["nominal", "expired_source", "inference_expiry"], default="nominal")
    args = ap.parse_args()
    if args.prepare:
        prepare(args.prepare, args.config, args.scenario)
    else:
        print(json.dumps(verify(args.verify, args.model_dir), indent=2))


if __name__ == "__main__":
    main()

