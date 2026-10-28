"""Compare the production C++ feedback header to the saved training reference.

Run with the training Python environment; no simulator, ROS, GPU or motors.
M20_HEADING_EVIDENCE optionally saves the complete check summary.
"""
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="training torch required for feedback parity")

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT.parent / "rl_training"
REFERENCE = TRAINING / "source/rl_training/m20_student_v1/v3_5_feedback.py"
if not REFERENCE.is_file():
    REFERENCE = TRAINING / "source/rl_training/task_support/student_v3_5/v3_5_feedback.py"
CHECKPOINT = Path(os.environ.get("M20_V35_CHECKPOINT", str(
    TRAINING / "logs/rsl_rl/deeprobotics_m20_student_v3_5/"
    "2026-10-19_12-12-39_599339_student_v3_5/model_2000.pt")))
KEYS = ("translation_threshold", "yaw_threshold", "heading_hold_kp", "heading_hold_kd",
        "heading_hold_max_yaw_rate", "heading_hold_deadband", "stair_heading_hold_kp",
        "stair_heading_hold_kd", "stair_heading_hold_max_yaw_rate",
        "stair_heading_hold_deadband", "heading_hold_rate_enter")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def terrain(height=None, valid=None, confidence=None):
    obs = np.zeros(789, dtype=np.float32)
    obs[57:244] = (np.zeros((11, 17), dtype=np.float32) if height is None else height).ravel()
    obs[244:431] = (np.ones((11, 17), dtype=np.float32) if valid is None else valid).ravel()
    obs[431:618] = (np.ones((11, 17), dtype=np.float32) if confidence is None else confidence).ravel()
    return obs.tolist()


def operation(step=0, yaw=0., rate=0., raw=(.5, 0., 0.), obs=None):
    return dict(step=step, yaw=yaw, rate=rate, raw=list(raw),
                obs=terrain() if obs is None else obs)


def scenarios():
    stair_height = np.zeros((11, 17), dtype=np.float32)
    stair_height[:, 8:] = -.2
    stair = terrain(stair_height)
    cases = {
        "duplicate_step_terrain_change": [operation(), operation(1, .2, .05),
                                           operation(1, .2, .05, obs=stair)],
        "duplicate_first_step_reenters_from_before": [operation(8, .2), operation(8, 1.4),
                                                     operation(9, 1.5)],
        "wrap_both_directions": [operation(0, 3.12), operation(1, -3.12),
                                  {"reset": True}, operation(1, -3.12), operation(2, 3.12)],
        "idle_strafe_turn_transitions": [operation(raw=(0., 0., 0.)),
            operation(1, .1, .1, raw=(0., 0., 0.)),
            operation(2, .1, .1, raw=(0., .4, 0.)),
            operation(3, .2, -.1, raw=(.4, 0., .3)),
            operation(4, .3, .1), operation(5, .4, .1),
            operation(6, .6, 0., raw=(0., 0., 0.))],
        "reset_same_step_then_regressed_step": [operation(5, .2), operation(6, .4),
            {"reset": True}, operation(6, .7), operation(2, -.9), operation(3, -.8)],
        "thresholds_and_clamps": [operation(),
            operation(1, .001, .03), operation(2, .001, .030001),
            operation(3, 2., 2.), operation(4, -2., -2., obs=stair),
            operation(5, 0., raw=(.1, 0., .1)),
            operation(6, .1, raw=(.100001, .1, .1)),
            operation(7, .2, raw=(.2, .100001, 0.)),
            operation(8, .3, raw=(.06, .08, 0.))],
    }
    expected_stairs = {}

    def scan(name, height, expected, valid=None, confidence=None):
        obs = terrain(height, valid, confidence)
        cases[name] = [operation(obs=obs), operation(1, .2, .05, obs=obs)]
        expected_stairs[name] = expected

    scan("scan_flat", np.zeros((11, 17), dtype=np.float32), False)
    scan("scan_stair", stair_height, True)
    for col, expected in ((1, False), (3, False), (4, True), (15, True), (16, False)):
        h = np.zeros((11, 17), dtype=np.float32)
        h[:, col:] = -.2
        scan("scan_x_gate_" + str(col), h, expected)
    h = np.zeros((11, 17), dtype=np.float32)
    h[[0, 1, 9, 10], 8:] = -.2
    scan("scan_y_gate_outside", h, False)
    h = np.zeros((11, 17), dtype=np.float32)
    h[2:4, 8:] = -.2
    valid = np.zeros((11, 17), dtype=np.float32)
    valid[2:4] = 1.
    scan("scan_two_inside_rows", h, True, valid)
    valid[3] = 0.
    scan("scan_one_inside_row", h, False, valid)
    valid[:] = .5
    scan("scan_validity_half_excluded", stair_height, False, valid)
    scan("scan_confidence_zero_excluded", stair_height, False,
         confidence=np.zeros((11, 17), dtype=np.float32))
    h = np.zeros((11, 17), dtype=np.float32)
    valid = np.zeros((11, 17), dtype=np.float32)
    valid[2:4, 7:9] = 1.
    h[3, 8] = -.2
    scan("scan_lower_median_even", h, False, valid)
    h[2, 8] = -.2
    scan("scan_lower_median_edge_positive", h, True, valid)
    for height, expected, name in ((np.float32(.06), False, "exact"),
        (np.nextafter(np.float32(.06), np.float32(np.inf)), True, "above")):
        h = np.zeros((11, 17), dtype=np.float32)
        h[:, 8:] = -height
        scan("scan_strict_06_" + name, h, expected)
    h = np.zeros((11, 17), dtype=np.float32)
    h[:, 8:] = -1.e6
    scan("scan_million_excluded", h, False)

    rng = np.random.default_rng(42019)
    ops = []
    for tick in range(320):
        height = np.round(rng.normal(0., .15, (11, 17)), 2).astype(np.float32)
        valid = (rng.random((11, 17)) > .25).astype(np.float32)
        confidence = rng.choice([0., .01, .8], (11, 17)).astype(np.float32)
        raw = ((.5, 0., 0.), (0., 0., 0.), (0., .3, 0.), (.4, 0., .2))[tick % 4]
        ops.append(operation(tick, float(rng.uniform(-math.pi, math.pi)),
                             float(rng.uniform(-.8, .8)), raw,
                             terrain(height, valid, confidence)))
        if tick % 23 == 0:
            ops.append(operation(tick, ops[-1]["yaw"], ops[-1]["rate"], raw))
        if tick % 71 == 0:
            ops.append({"reset": True})
    cases["random_finite_sensor_dropout"] = ops
    return cases, expected_stairs


def reference_results(module, metadata, operations):
    feedback = module.SensorHeadingFeedback(1, "cpu", metadata["tuning"])
    results = []
    for op in operations:
        if op.get("reset"):
            feedback.reset()
            results.append(dict(mode=int(feedback.mode[0]), target=float(feedback.target[0])))
            continue
        obs = torch.tensor(op["obs"], dtype=torch.float32)
        channels = (obs[None, 57:244], obs[None, 244:431], obs[None, 431:618])
        raw = torch.tensor([op["raw"]], dtype=torch.float32)
        result = feedback(raw, torch.tensor([op["yaw"]], dtype=torch.float32),
                          torch.tensor([op["rate"]], dtype=torch.float32), channels, op["step"])
        height, valid, confidence = channels
        x = torch.arange(-8, 9, dtype=height.dtype) * .1
        y = torch.arange(-5, 6, dtype=height.dtype) * .1
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack((xx.flatten(), yy.flatten()), -1)
        z = torch.where((valid > .5) & (confidence > 0.), -height, torch.nan)
        hits = torch.cat((grid[None], z[..., None]), -1)
        stairs = bool(module.stair_scan_mask(hits, grid)[0])
        results.append(dict(command=result[0].tolist(), mode=int(feedback.mode[0]),
                            target=float(feedback.target[0]), stairs=stairs))
    return results


def test_production_cpp_sensor_heading_feedback_matches_training(tmp_path):
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location("training_v35_feedback", REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    metadata = saved["student_v1"]["provenance"]["command_feedback"]
    assert metadata["version"] == "legacy_v1_v3_5"
    assert saved["student_v1"]["obs_dim"] == 789
    del saved
    executable = tmp_path / "heading_feedback_probe"
    sdk = ROOT / "src/m20_student_sdk"
    subprocess.run(["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-pedantic",
                    "-I" + str(sdk / "plugin"), "-I" + str(sdk / "vendor/include/utils"),
                    str(ROOT / "tests/heading_feedback_probe.cpp"), "-o", str(executable)],
                   check=True, capture_output=True, text=True, timeout=60)
    cases, expected_stairs = scenarios()
    requests = [dict(metadata=metadata, operations=ops) for ops in cases.values()]
    invalid_metadata_count = 0
    for key in KEYS:
        for value in ("missing", -1., "invalid", True):
            bad = copy.deepcopy(metadata)
            if value == "missing":
                del bad["tuning"][key]
            else:
                bad["tuning"][key] = value
            requests.append(dict(metadata=bad, operations=[]))
            invalid_metadata_count += 1
        for value in ("nan", "inf"):
            requests.append(dict(metadata=metadata, operations=[],
                                 metadata_injection=dict(key=key, value=value)))
            invalid_metadata_count += 1
    for version in ("legacy_v1", "", 5):
        bad = copy.deepcopy(metadata)
        bad["version"] = version
        requests.append(dict(metadata=bad, operations=[]))
        invalid_metadata_count += 1

    safety_ops = [operation(0, .1)]
    for invalid in ("raw_nan", "yaw_inf", "rate_nan", "obs_nan", "obs_inf", "obs_short"):
        bad = operation(1, .3)
        bad["invalid"] = invalid
        safety_ops.append(bad)
    safety_ops.append(operation(1, .3))
    requests.append(dict(metadata=metadata, operations=safety_ops))
    result = subprocess.run([str(executable)], input=json.dumps(requests, allow_nan=False),
                            check=True, capture_output=True, text=True, timeout=15)
    observed = json.loads(result.stdout)
    max_error = 0.
    compared = 0
    bypass_differences = 0
    for index, (name, operations) in enumerate(cases.items()):
        actual = observed[index]["results"]
        expected = reference_results(module, metadata, operations)
        assert len(actual) == len(expected), name
        for tick, (a, e, op) in enumerate(zip(actual, expected, operations)):
            assert "error" not in a, (name, tick, a)
            assert a["mode"] == e["mode"], (name, tick, a, e)
            assert abs(a["target"] - e["target"]) <= 1.e-7, (name, tick, a, e)
            if op.get("reset"):
                continue
            assert a["stairs"] == e["stairs"], (name, tick, a, e)
            error = float(np.max(np.abs(np.array(a["command"]) - e["command"])))
            max_error = max(max_error, error)
            assert error <= 2.e-6, (name, tick, a, e, error)
            if name in expected_stairs:
                assert a["stairs"] is expected_stairs[name], (name, a)
            bypass_differences += abs(e["command"][2] - op["raw"][2]) > 1.e-4
            compared += 1
    # Negative control: the copied runner's old raw-command bypass must fail
    # the same reference measurement on multiple deterministic cases.
    assert bypass_differences >= 10
    offset = len(cases)
    assert all("error" in item for item in observed[offset:offset + invalid_metadata_count])
    safety = observed[-1]["results"]
    assert all("error" in item for item in safety[1:-1])
    assert all(item["mode"] == safety[0]["mode"] and item["target"] == safety[0]["target"]
               for item in safety[1:-1])
    safety_reference = reference_results(module, metadata, [safety_ops[0], safety_ops[-1]])[-1]
    np.testing.assert_allclose(safety[-1]["command"], safety_reference["command"], rtol=0., atol=2.e-6)
    record = {
        "status": "passed", "device": "cpu", "torch_threads": torch.get_num_threads(),
        "checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha256(CHECKPOINT),
        "metadata_path": "student_v1.provenance.command_feedback", "feedback_version": metadata["version"],
        "reference": str(REFERENCE), "reference_sha256": sha256(REFERENCE),
        "production_header_sha256": sha256(sdk / "plugin/student_heading_feedback.hpp"),
        "probe_sha256": sha256(ROOT / "tests/heading_feedback_probe.cpp"),
        "observation_dim": 789, "cases": list(cases), "compared_policy_calls": compared,
        "max_command_abs_error": max_error, "absolute_tolerance": 2.e-6,
        "invalid_metadata_rejected": invalid_metadata_count,
        "nonfinite_or_bad_shape_inputs_rejected_without_state_change": len(safety_ops) - 2,
        "negative_control_raw_bypass_different_calls": int(bypass_differences),
        "negative_control_lower_median_not_mean": "scan_lower_median_even=false; mean would detect edge",
        "physical_actions": False,
    }
    destination = os.environ.get("M20_HEADING_EVIDENCE")
    if destination:
        Path(destination).write_text(json.dumps(record, indent=2) + "\n")

