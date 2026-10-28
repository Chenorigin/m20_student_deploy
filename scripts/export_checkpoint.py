#!/usr/bin/env python3
"""Export the deterministic student in a distillation checkpoint, on CPU only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import copy
import math

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src/m20_heightmap"))
from m20_heightmap.contract import contract, validate_model_contract


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def student_actor(saved, profile):
    # This SDK does not implement StairQuiet execution control. Keep this
    # check local: deployed export tools cannot depend on a training checkout.
    provenance = saved.get("student_" + profile, {}).get("provenance", {})
    if ("quiet_execution_control" in saved or "quiet_execution_control" in provenance
            or "control_contract" in saved.get("m20_training_manifest", {}).get("noise_refinement", {})
            or "control_contract" in provenance.get("teacher_manifest", {}).get("noise_refinement", {})):
        raise ValueError("StairQuiet requires its execution controller; the existing SDK export is unsupported")
    meta = saved.get("student_"+profile)
    if not isinstance(meta, dict) or not isinstance(saved.get("student_state_dict"), dict):
        raise ValueError("Expected the selected student version and student_state_dict")
    provenance = meta.get("provenance", {})
    actual = provenance.get("contract", {})
    steps = actual.get("history_steps", 1)
    spec = contract(profile, steps)
    cfg = provenance.get("training_config", {}).get("student", {})
    if cfg.get("obs_normalization") is not False or cfg.get("activation") != "elu":
        raise ValueError("Only unnormalized ELU student actors are supported")
    hidden = cfg.get("hidden_dims")
    if not isinstance(hidden, list) or not hidden or any(type(v) is not int or v <= 0 for v in hidden):
        raise ValueError("Invalid saved hidden dimensions")
    if actual.get("student_observation_dim") != spec["input_dim"]:
        raise ValueError("Checkpoint history/observation dimensions disagree")
    feedback = provenance.get("command_feedback", {})
    if profile != "v1" or steps != 4 or feedback.get("version") != "legacy_v1_v3_5":
        raise ValueError("This isolated deployment requires the student-v3.5 feedback contract")
    names = ("translation_threshold", "yaw_threshold", "heading_hold_kp", "heading_hold_kd",
             "heading_hold_max_yaw_rate", "heading_hold_deadband", "stair_heading_hold_kp",
             "stair_heading_hold_kd", "stair_heading_hold_max_yaw_rate",
             "stair_heading_hold_deadband", "heading_hold_rate_enter")
    tuning = feedback.get("tuning", {})
    for name in names:
        value = tuning.get(name)
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError("Missing/invalid feedback parameter: " + name)
    actual = copy.deepcopy(actual)
    actual["deployment_family"] = "m20_student_v3_5"
    actual["command_feedback"] = {"version": feedback["version"],
                                  "tuning": {name: tuning[name] for name in names}}
    actual["perception"] = copy.deepcopy(provenance["perception"])
    sizes = [spec["input_dim"], *hidden, 16]
    modules = []
    for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
        modules.append(torch.nn.Linear(a, b))
        if i < len(sizes)-2:
            modules.append(torch.nn.ELU())
    actor = torch.nn.Sequential(*modules)
    state = saved["student_state_dict"]
    if any(not k.startswith("mlp.") and k != "distribution.log_std_param" for k in state):
        raise ValueError("Unexpected actor state key")
    if any(not torch.isfinite(tensor).all() for tensor in state.values()):
        raise ValueError("Nonfinite actor weights")
    actor.load_state_dict({k[4:]: v for k, v in state.items() if k.startswith("mlp.")}, strict=True)
    duplicate = saved.get("actor_state_dict")
    if duplicate is not None and (set(duplicate) != set(state) or
            any(not torch.equal(duplicate[key], state[key]) for key in state)):
        raise ValueError("Conflicting actor_state_dict and student_state_dict")
    actor.eval().requires_grad_(False)
    return actor, actual, provenance, spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--profile", choices=("v0", "v1"), required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--env-yaml", type=Path, help="defaults to env.yaml beside the checkpoint")
    ap.add_argument("--training", type=Path, help="CTS only: training checkout for its canonical loader")
    args = ap.parse_args()
    torch.set_num_threads(1)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "cts" in saved:
        if args.profile != "v1" or args.training is None:
            raise ValueError("CTS requires --profile v1 --training /path/to/rl_training")
        from export_cts_checkpoint import export
        print(json.dumps(export(args.checkpoint,args.training,args.output),indent=2))
        return
    actor, actual, provenance, spec = student_actor(saved, args.profile)
    from training_parameters import derive
    env_path = args.env_yaml or args.checkpoint.parent/"env.yaml"
    training_parameters = derive(env_path, provenance["training_config"]["clip_actions"])
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError("Export output must be a new/empty directory; keep previous releases")
    (args.output/"robot_training_contract.json").write_text(json.dumps(training_parameters, indent=2)+"\n")
    sidecar = args.output/"observation_contract.json"
    sidecar.write_text(json.dumps(actual, indent=2)+"\n")
    validate_model_contract(sidecar, spec)
    exported = torch.jit.script(actor)
    torch.manual_seed(42)
    test_input = torch.randn(128, spec["input_dim"])
    with torch.inference_mode():
        expected = actor(test_input)
        observed = exported(test_input)
    if expected.shape != (128, 16) or not torch.isfinite(expected).all():
        raise ValueError("Invalid deterministic actor output")
    torch.testing.assert_close(observed, expected, rtol=1e-6, atol=1e-6)
    model_path = args.output/"policy.pt"
    torch.jit.save(exported, str(model_path))
    reloaded = torch.jit.load(str(model_path), map_location="cpu").eval()
    with torch.inference_mode():
        after = reloaded(test_input)
    torch.testing.assert_close(after, expected, rtol=1e-6, atol=1e-6)
    record = {"profile": args.profile, "checkpoint_path": str(args.checkpoint.resolve()),
              "checkpoint_sha256": digest(args.checkpoint), "policy_sha256": digest(model_path),
              "contract_sha256": digest(sidecar), "source_state_key": "student_state_dict",
              "input_dim": spec["input_dim"], "history_steps": spec["history_steps"],
              "action_dim": 16, "training_provenance": provenance,
              "deployment_family": "m20_student_v3_5", "iteration": saved.get("iter"),
              "hidden_dims": provenance["training_config"]["student"]["hidden_dims"],
              "command_feedback": actual["command_feedback"],
              "env_yaml_sha256": digest(env_path),
              "robot_training_contract_sha256": digest(args.output/"robot_training_contract.json"),
              "export_check": {"samples": 128, "max_abs_error": float((after-expected).abs().max()),
                               "device": "cpu", "optimizer_teacher_replay_exported": False}}
    (args.output/"deployment.json").write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps({k: record[k] for k in ("profile", "input_dim", "history_steps", "checkpoint_sha256", "export_check")}, indent=2))


if __name__ == "__main__":
    main()
