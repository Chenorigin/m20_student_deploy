"""CTS nominal policy ABI. No robot interfaces or training dependencies.

Raw command3 is sampled before sensor heading feedback. It cannot be recovered
from policy[6:9], including when the requested command was zero.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

FAMILY = "m20_cts_student"
VERSION = "m20_cts_action_latent_v2"
# v8 changes training curriculum; the validated nominal inference ABI is unchanged.
SUPPORTED_VERSIONS = (VERSION, "m20_cts_action_latent_v8")
INPUTS = {"obs": [1, 789], "raw_command": [1, 3]}
HARDWARE_REASON = ("CTS requires mandatory StairQuiet landing control at every 5ms substep; "
                   "the SDK has no validated contact/actuator binding. Hardware startup is blocked. "
                   "Use offline_cts.py or the training deploy_cts_student.py simulation runtime.")


def canonical_hash(value):
    # Match task_support/stair_quiet_v1/landing.py::contract_hash exactly.
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_cts_contract(actual):
    from .contract import contract
    if (actual.get("deployment_family") != FAMILY or actual.get("cts_version") not in SUPPORTED_VERSIONS
            or actual.get("model_inputs") != INPUTS
            or actual.get("raw_command_semantics") != "requested_command_before_heading_feedback"
            or actual.get("actor_scope") != "student_encoder_and_shared_decoder_nominal16"
            or actual.get("standalone_robot_controller") is not False
            or actual.get("history_steps") != contract("v1", 4)["history_steps"]):
        raise ValueError("CTS model input/execution contract mismatch")
    control = actual.get("quiet_execution_control", {})
    if (not isinstance(control, dict) or control.get("schema") != "stair_quiet_v1.landing_control.v1"
            or control.get("action_semantics") != "nominal_16d_before_execution_control_and_actuator_delay"
            or not isinstance(control.get("settings"), dict)
            or not control["settings"]
            or canonical_hash(control) != actual.get("quiet_execution_control_sha256")):
        raise ValueError("CTS mandatory landing control missing or changed")
    feedback = actual.get("command_feedback", {})
    if feedback.get("version") != "legacy_v1_v3_5" or not isinstance(feedback.get("tuning"), dict):
        raise ValueError("CTS command feedback missing")
    for key in ("translation_threshold", "yaw_threshold", "heading_hold_kp", "heading_hold_kd",
                "heading_hold_max_yaw_rate", "heading_hold_deadband", "stair_heading_hold_kp",
                "stair_heading_hold_kd", "stair_heading_hold_max_yaw_rate",
                "stair_heading_hold_deadband", "heading_hold_rate_enter"):
        value = feedback["tuning"].get(key)
        if type(value) not in (float, int) or not np.isfinite(value) or value < 0:
            raise ValueError("CTS invalid feedback parameter: " + key)
    perception = actual.get("perception", {})
    if perception.get("front_only") is not True:
        raise ValueError("CTS requires the saved front-only perception contract")
    front_mask(actual)
    return actual


def front_mask(actual):
    from .contract import GRID
    p = actual["perception"]
    sensor = np.asarray(p.get("front_sensor_offset_m"), dtype=float)
    fov, blind = p.get("lidar_horizontal_fov_deg"), p.get("front_blind_distance_m")
    if (sensor.shape != (3,) or not np.isfinite(sensor).all()
            or type(fov) not in (float, int) or not np.isfinite(fov) or not 0 < fov <= 360
            or type(blind) not in (float, int) or not np.isfinite(blind) or blind < 0):
        raise ValueError("CTS invalid front sensor contract")
    delta = GRID - sensor[:2]
    return ((delta[:, 0] > 0) & (np.linalg.norm(delta, axis=1) > blind + 1e-6)
            & (np.abs(np.arctan2(delta[:, 1], delta[:, 0])) <= np.deg2rad(fov * .5) + 1e-7))


def configured_contract(cfg):
    path = cfg.get("aos", {}).get("model_contract")
    if not path or not Path(path).is_file():
        return None
    actual = json.loads(Path(path).read_text())
    if actual.get("deployment_family") != FAMILY and "cts_version" not in actual:
        return None
    from .contract import contract, validate_model_contract
    validate_model_contract(path, contract(cfg["profile"], cfg["history_steps"]))
    validate_mapper(cfg, actual)
    execution_mode(cfg)
    return actual


def validate_mapper(cfg, actual):
    p, visibility = actual["perception"], cfg.get("visibility", {})
    sensors = visibility.get("sensors", [])
    if (cfg["runtime"].get("mapping_region") != "front" or len(sensors) != 2
            or visibility.get("horizontal_fov_deg") != p["lidar_horizontal_fov_deg"]
            or sensors[0].get("yaw_rad") != 0
            or sensors[0].get("position_base_m") != p["front_sensor_offset_m"]):
        raise ValueError("CTS model and calibrated front mapper geometry differ")
    for name, key in (("max_map_age_s", "max_map_age_s"),
                      ("confidence_reference_std", "confidence_reference_std"),
                      ("confidence_age_tau_s", "confidence_age_tau_s"),
                      ("variance_growth_per_s", "variance_growth_per_s"),
                      ("confidence_step_ambiguity_weight", "confidence_step_ambiguity_weight")):
        if p.get(key) is None or cfg["estimator"].get(name) != p[key]:
            raise ValueError("CTS mapper quality semantics differ: " + name)


def execution_mode(cfg):
    mode = cfg.get("aos", {}).get("cts_execution_mode", "landing")
    if mode not in ("landing", "nominal_pd"):
        raise ValueError("cts_execution_mode must be landing or nominal_pd")
    return mode


def require_hardware_runtime(actual, runtime_config=None):
    # Marker presence, not truthiness: null or stripped CTS fields must not
    # silently turn a nominal actor into a legacy complete controller.
    if not isinstance(actual, dict):
        raise ValueError("Missing model execution contract")
    if runtime_config is not None:
        mode = execution_mode(runtime_config)
        if mode == "nominal_pd" and actual.get("deployment_family") == FAMILY:
            validate_cts_contract(actual)
            return mode
    if (actual.get("deployment_family") == FAMILY or "cts_version" in actual
            or "quiet_execution_control" in actual or "control_contract" in actual):
        raise ValueError(HARDWARE_REASON)
    for value in actual.values():
        if isinstance(value, dict):
            require_hardware_runtime(value)


def validate_session(session, actual, contract_path):
    validate_cts_contract(actual)
    inputs = session.get_inputs()
    if ([(i.name, i.shape, i.type) for i in inputs] !=
            [(name, shape, "tensor(float)") for name, shape in INPUTS.items()]):
        raise ValueError("CTS ONNX must expose static float32 obs789 and raw_command3")
    if ([(o.name, o.shape, o.type) for o in session.get_outputs()] !=
            [("actions", [1, 16], "tensor(float)")]):
        raise ValueError("CTS ONNX must return static float32 actions16")
    validate_embedded(session.get_modelmeta().custom_metadata_map, actual, contract_path)


def validate_embedded(embedded, actual, contract_path):
    expected = {"deployment_family": FAMILY, "contract_sha256": sha(contract_path),
                "quiet_execution_control_sha256": actual["quiet_execution_control_sha256"]}
    if any(embedded.get(k) != v for k, v in expected.items()):
        raise ValueError("CTS embedded model identity/control mismatch")
    for name in ("command_feedback", "quiet_execution_control"):
        if json.loads(embedded.get(name, "null")) != actual[name]:
            raise ValueError("CTS embedded metadata mismatch: " + name)


class CTSInference:
    """CPU-only complete student+decoder, returning nominal actions only."""
    def __init__(self, model, contract_path):
        import onnxruntime as ort
        from .contract import contract, validate_model_contract
        actual = validate_model_contract(contract_path, contract("v1", 4))
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.add_session_config_entry("session.disable_prepacking", "1")
        self.session = ort.InferenceSession(str(model), sess_options=options,
                                           providers=["CPUExecutionProvider"])
        validate_session(self.session, actual, contract_path)

    def __call__(self, policy, raw_command):
        feed = {"obs": np.asarray(policy, dtype=np.float32),
                "raw_command": np.asarray(raw_command, dtype=np.float32)}
        for name, shape in INPUTS.items():
            if list(feed[name].shape) != shape or not np.isfinite(feed[name]).all():
                raise ValueError("Invalid CTS input: " + name)
        result = self.session.run(["actions"], feed)[0]
        if result.shape != (1, 16) or not np.isfinite(result).all():
            raise ValueError("Invalid CTS nominal action")
        return result
