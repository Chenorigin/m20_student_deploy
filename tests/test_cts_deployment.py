"""CTS dual-input policy and SDK boundary regressions. No robot connections."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from m20_heightmap.contract import contract, validate_model_contract
from m20_heightmap.cts import (FAMILY, VERSION, INPUTS, canonical_hash,
                               require_hardware_runtime, CTSInference, front_mask, validate_mapper)

ROOT = Path(__file__).resolve().parents[1]


def script_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def metadata():
    value = json.loads((ROOT / "models/v1/observation_contract.json").read_text())
    value.update(deployment_family=FAMILY, cts_version=VERSION, model_inputs=copy.deepcopy(INPUTS),
                 raw_command_semantics="requested_command_before_heading_feedback",
                 actor_scope="student_encoder_and_shared_decoder_nominal16",
                 standalone_robot_controller=False)
    value["command_feedback"] = {"version": "legacy_v1_v3_5", "tuning": {
        "translation_threshold": .1, "yaw_threshold": .1, "heading_hold_kp": .8,
        "heading_hold_kd": .4, "heading_hold_max_yaw_rate": .25, "heading_hold_deadband": .008,
        "stair_heading_hold_kp": 1.6, "stair_heading_hold_kd": .7,
        "stair_heading_hold_max_yaw_rate": .5, "stair_heading_hold_deadband": .008,
        "heading_hold_rate_enter": .03}}
    value["perception"] = {"front_only": True, "front_sensor_offset_m": [.32028, 0., -.013],
                           "lidar_horizontal_fov_deg": 135., "front_blind_distance_m": 0.}
    control = {"schema": "stair_quiet_v1.landing_control.v1", "settings": {"mode": "combined"},
               "action_semantics": "nominal_16d_before_execution_control_and_actuator_delay"}
    value["quiet_execution_control"] = control
    value["quiet_execution_control_sha256"] = canonical_hash(control)
    return value


def test_cts_observation_abi_and_control_digest(tmp_path, metadata):
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(metadata))
    validate_model_contract(path, contract("v1", 4))
    metadata["quiet_execution_control"]["settings"]["mode"] = "off"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="landing"):
        validate_model_contract(path, contract("v1", 4))


@pytest.mark.parametrize("change", ["inputs", "history", "raw", "controller", "family"])
def test_incompatible_cts_contract_is_rejected(tmp_path, metadata, change):
    if change == "inputs":
        del metadata["model_inputs"]["raw_command"]
    elif change == "history":
        metadata["history_steps"] = 1
    elif change == "raw":
        metadata["raw_command_semantics"] = "from_effective_command"
    elif change == "controller":
        metadata["standalone_robot_controller"] = True
    else:
        metadata["deployment_family"] = "m20_student_v3_5"
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        validate_model_contract(path, contract("v1", 4))


@pytest.mark.parametrize("marker", [{"deployment_family": FAMILY}, {"cts_version": VERSION},
    {"quiet_execution_control": None}, {"training_provenance": {"quiet_execution_control": {}}},
    {"cts": {"quiet_execution_control": None}},
    {"teacher_manifest": {"noise_refinement": {"control_contract": {}}}}])
def test_offline_acceptance_cannot_enable_hardware(marker):
    with pytest.raises(ValueError, match="Hardware startup is blocked"):
        require_hardware_runtime(marker)
    require_hardware_runtime({"deployment_family": "m20_student_v3_5"})


def test_explicit_nominal_pd_admits_only_complete_cts(metadata):
    cfg = {"aos": {"cts_execution_mode": "nominal_pd"}}
    assert require_hardware_runtime(metadata, cfg) == "nominal_pd"
    bad = copy.deepcopy(metadata)
    del bad["model_inputs"]["raw_command"]
    with pytest.raises(ValueError, match="contract mismatch"):
        require_hardware_runtime(bad, cfg)
    bad = copy.deepcopy(metadata)
    bad["deployment_family"] = "other_quiet"
    with pytest.raises(ValueError, match="Hardware startup is blocked"):
        require_hardware_runtime(bad, cfg)
    for value in ("landing", "invalid", None, False, [], {}):
        with pytest.raises(ValueError):
            require_hardware_runtime(metadata, {"aos": {"cts_execution_mode": value}})
    with pytest.raises(ValueError, match="Hardware startup is blocked"):
        require_hardware_runtime(metadata, {"aos": {}})


def test_mapper_rejects_sensor_or_quality_drift(metadata):
    p = metadata["perception"]
    quality = {"max_map_age_s": .5, "confidence_reference_std": .05,
               "confidence_age_tau_s": .5, "variance_growth_per_s": .001,
               "confidence_step_ambiguity_weight": 0.}
    p.update(quality)
    cfg = {"runtime": {"mapping_region": "front"}, "estimator": quality.copy(),
           "visibility": {"horizontal_fov_deg": p["lidar_horizontal_fov_deg"],
                          "sensors": [{"position_base_m": p["front_sensor_offset_m"], "yaw_rad": 0}, {}]}}
    validate_mapper(cfg, metadata)
    original = front_mask(metadata)
    metadata["perception"]["front_blind_distance_m"] = .4
    assert np.count_nonzero(front_mask(metadata)) < np.count_nonzero(original)
    cfg["visibility"]["horizontal_fov_deg"] = 360
    with pytest.raises(ValueError, match="geometry"):
        validate_mapper(cfg, metadata)


@pytest.fixture(scope="module")
def sensor_probe(tmp_path_factory):
    output = tmp_path_factory.mktemp("cts_sensor") / "probe"
    subprocess.run(["g++", "-std=c++17", "-O1", "-I" + str(ROOT / "src/m20_student_sdk/plugin"),
                    "-I" + str(ROOT / "src/m20_student_sdk/vendor/include/utils"),
                    str(ROOT / "src/m20_student_sdk/plugin/cts_sensor_probe.cpp"), "-o", str(output)],
                   capture_output=True, text=True, check=True, timeout=60)
    return output


def test_sensor_heading_front_mask_and_history(sensor_probe, metadata, tmp_path):
    torch = pytest.importorskip("torch")
    reference = script_module("reference_training_feedback")
    feedback = reference.SensorHeadingFeedback(1, "cpu", metadata["command_feedback"]["tuning"])
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(metadata))
    rng = np.random.default_rng(42)
    frames, expected = [], []
    history = [np.zeros(57)] * 3
    yy, xx = np.meshgrid(np.arange(-5, 6) * .1, np.arange(-8, 9) * .1, indexing="ij")
    dx, dy = xx.ravel() - .32028, yy.ravel()
    mask = (dx > 0) & (np.abs(np.arctan2(dy, dx)) <= np.deg2rad(67.5) + 1e-7)
    for i in range(8):
        p = rng.normal(0, .1, 57).astype(np.float32)
        raw = np.array([.5, 0, 0] if i > 3 else [0, 0, 0], dtype=np.float32)
        terrain = np.concatenate((rng.uniform(-.3, .3, 187), np.ones(187), np.ones(187))).astype(np.float32)
        channels = terrain.reshape(3, 187).copy()
        channels[:, ~mask] = 0
        effective = feedback(torch.tensor(raw[None]), torch.tensor([i * .02]),
                             torch.tensor([p[2] / .25]), torch.tensor(channels[:, None]), i)
        frames.append(dict(proprio=p.tolist(), terrain=terrain.tolist(), raw_command=raw.tolist(), yaw=i * .02))
        p[6:9] = effective[0].numpy()
        expected.append(np.concatenate((p, channels.ravel(), *history)))
        history = [*history[1:], p.copy()]
    result = subprocess.run([str(sensor_probe), str(path)], input=json.dumps(frames),
                            capture_output=True, text=True, check=True, timeout=10)
    rows = json.loads(result.stdout)
    for i, row in enumerate(rows):
        np.testing.assert_allclose(row["obs"], expected[i], atol=2e-6, rtol=0)
        np.testing.assert_array_equal(row["raw_command"], frames[i]["raw_command"])
    assert rows[1]["obs"][8] != 0 and rows[1]["raw_command"][2] == 0


def test_check_model_demands_separate_raw_command(metadata, tmp_path):
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(metadata))
    result = subprocess.run([sys.executable, str(ROOT / "scripts/check_model.py"),
                             "--model", str(tmp_path / "missing.onnx"), "--contract", str(path),
                             "--profile", "v1"], capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert "requires --raw-command" in result.stderr


def test_dual_input_gate_and_embedded_control(metadata, tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    # A deliberately non-identity decoder exposes omission and raw/effective
    # confusion, unlike an initialization-only identity decoder test.
    class Complete(torch.nn.Module):
        def forward(self, obs, raw):
            latent = obs[:, 9:25]
            return latent + .25 * (raw.abs().amax(-1, keepdim=True) > 1e-4)
    model = tmp_path / "policy.pt"
    torch.jit.save(torch.jit.script(Complete()), str(model), _extra_files={
        "cts_contract.json": json.dumps(metadata),
        "quiet_execution_control.json": json.dumps(metadata["quiet_execution_control"])})
    sidecar = tmp_path / "contract.json"
    sidecar.write_text(json.dumps(metadata))
    script_module("export_cts_onnx").export_onnx(model, sidecar, tmp_path / "policy.onnx")
    session = CTSInference(tmp_path / "policy.onnx", sidecar)
    obs = np.zeros((1, 789), dtype=np.float32)
    obs[0, 8] = .3  # Effective yaw alone must not open the raw command gate.
    np.testing.assert_array_equal(session(obs, np.zeros((1, 3))), np.zeros((1, 16)))
    np.testing.assert_array_equal(session(obs, np.array([[.5, 0, 0]])), np.full((1, 16), .25))
    with pytest.raises(ValueError, match="raw_command"):
        session(obs, np.zeros(3))
    metadata["command_feedback"]["tuning"]["heading_hold_kp"] += .1
    sidecar.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="identity/control"):
        CTSInference(tmp_path / "policy.onnx", sidecar)


@pytest.mark.parametrize("version", ["m20_cts_action_latent_v2", "m20_cts_action_latent_v8"])
def test_supported_training_versions_preserve_execution_guards(metadata, tmp_path, version):
    metadata["cts_version"] = version
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(metadata))
    validate_model_contract(path, contract("v1", 4))
    assert require_hardware_runtime(metadata, {"aos": {"cts_execution_mode": "nominal_pd"}}) == "nominal_pd"
    with pytest.raises(ValueError, match="Hardware startup is blocked"):
        require_hardware_runtime(metadata)
    metadata["model_inputs"]["raw_command"] = [1, 4]
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="contract mismatch"):
        validate_model_contract(path, contract("v1", 4))


@pytest.mark.parametrize("version", ["m20_cts_action_latent_v9", "m20_cts_action_latent_v1", None])
def test_unvalidated_training_versions_are_rejected(metadata, tmp_path, version):
    metadata["cts_version"] = version
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="contract mismatch"):
        validate_model_contract(path, contract("v1", 4))
