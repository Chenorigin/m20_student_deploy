#!/usr/bin/env python3
"""Export the canonical CTS student+decoder on CPU; never enables hardware.

Only export needs the training checkout. Board inference uses the self-contained
ONNX bundle and never opens teacher sources, retention data or optimizers.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/m20_heightmap"))
from m20_heightmap.contract import contract, validate_model_contract
from m20_heightmap.cts import FAMILY, INPUTS, canonical_hash, sha


def load_actor(checkpoint, training, policy_kind="cts"):
    # Use the production loader's metadata/state validation, not a reconstructed
    # approximation or the initialization student_state_dict.
    sys.path.insert(0, str(training / "source/rl_training"))
    sys.path.insert(0, str(training / "deploy/deploy_mujoco"))
    if policy_kind == "student-finetune":
        from deploy_cts_student_finetune import load_policy
    elif policy_kind == "cts":
        from deploy_cts_student import load_policy
    else:
        raise ValueError("Unknown policy kind")
    actor, control = load_policy(checkpoint)
    meta = deepcopy(actor.student_provenance)
    if policy_kind == "student-finetune":
        import torch
        from task_support.cts_student_finetune.configuration import BASELINE_SHA256, TASK, DEPLOY_VERSIONS
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        provenance = actor.finetune_provenance
        if (meta["version"] != "m20_cts_action_latent_v24"
                or provenance.get("task") != TASK
                or provenance.get("version") != saved["version"]
                or saved["version"] not in DEPLOY_VERSIONS
                or saved["student_ppo"].get("version") != saved["version"]
                or saved["student_ppo"].get("config") != provenance.get("config")
                or not isinstance(provenance.get("config"), dict)
                or provenance["config"].get("version") != saved["version"]
                or type(saved["student_ppo"].get("num_updates")) is not int
                or saved["student_ppo"]["num_updates"] < 0
                or provenance.get("baseline_sha256") != BASELINE_SHA256
                or provenance.get("reference_limit_interruption")):
            raise ValueError("Invalid or diagnostic student fine-tune provenance")
        meta["student_finetune"] = {
            "task": TASK, "version": saved["version"], "baseline_sha256": BASELINE_SHA256,
            "metadata_sha256": canonical_hash(provenance),
            "num_updates": saved["student_ppo"]["num_updates"],
            "source_state_key": "student_ppo.models.student + baseline frozen decoder"}
    return actor.policy, meta, control


def export(checkpoint, training, output, policy_kind="cts"):
    import torch
    from export_cts_onnx import export_onnx
    torch.set_num_threads(1)
    checkpoint, training, output = map(lambda p: Path(p).resolve(), (checkpoint, training, output))
    if output.exists() and any(output.iterdir()):
        raise ValueError("CTS export requires a new/empty model directory")
    checkpoint_sha = sha(checkpoint)
    actor, meta, control = load_actor(checkpoint, training, policy_kind)
    actual = deepcopy(meta["contract"])
    actual.update(deployment_family=FAMILY, cts_version=meta["version"], model_inputs=INPUTS,
                  raw_command_semantics="requested_command_before_heading_feedback",
                  actor_scope="student_encoder_and_shared_decoder_nominal16",
                  standalone_robot_controller=False,
                  command_feedback=deepcopy(meta["command_feedback"]),
                  perception=deepcopy(meta["perception"]), quiet_execution_control=control,
                  quiet_execution_control_sha256=canonical_hash(control))
    if 'joint_limits' in meta['config']:
        actual['software_joint_limits'] = deepcopy(meta['config']['joint_limits'])
        actual['software_joint_limits_sha256'] = canonical_hash(actual['software_joint_limits'])
    if "student_finetune" in meta:
        actual["student_finetune"] = deepcopy(meta["student_finetune"])
    if actual["quiet_execution_control_sha256"] != meta["quiet_execution_control_sha256"]:
        raise ValueError("Saved CTS execution digest mismatch")
    output.mkdir(parents=True, exist_ok=True)
    sidecar = output / "observation_contract.json"
    sidecar.write_text(json.dumps(actual, indent=2, allow_nan=False) + "\n")
    validate_model_contract(sidecar, contract("v1", 4))
    actor.eval().requires_grad_(False)
    scripted = torch.jit.script(actor)
    generator = torch.Generator().manual_seed(42)
    x = torch.randn(128, 789, generator=generator)
    raw = torch.randn(128, 3, generator=generator)
    raw[::2] = 0  # Effective yaw can remain nonzero in x while raw is zero.
    with torch.inference_mode():
        expected = actor(x, raw)
        torch.testing.assert_close(scripted(x, raw), expected, atol=1e-6, rtol=1e-6)
    path = output / "policy.pt"
    torch.jit.save(scripted, str(path), _extra_files={
        "quiet_execution_control.json": json.dumps(control, sort_keys=True),
        "cts_contract.json": json.dumps(actual, sort_keys=True)})
    reloaded = torch.jit.load(str(path), map_location="cpu").eval()
    with torch.inference_mode():
        observed = reloaded(x, raw)
        torch.testing.assert_close(observed, expected, atol=1e-6, rtol=1e-6)
    if sha(checkpoint) != checkpoint_sha:
        raise ValueError("Checkpoint changed during export")
    record = {"deployment_family": FAMILY, "checkpoint_path": str(checkpoint),
              "checkpoint_sha256": checkpoint_sha, "policy_sha256": sha(path),
              "contract_sha256": sha(sidecar), "model_inputs": INPUTS,
              "source_state_key": (meta["student_finetune"]["source_state_key"] if "student_finetune" in meta
                                   else "algorithm.models_state_dict.student + decoder"),
              "policy_kind": policy_kind,
              "quiet_execution_control_sha256": actual["quiet_execution_control_sha256"],
              "standalone_robot_controller": False, "hardware_commands": 0,
              "export_check": {"samples": 128, "max_abs_error": float((observed-expected).abs().max()),
                               "device": "cpu"}}
    if "student_finetune" in meta:
        record["student_finetune"] = deepcopy(meta["student_finetune"])
    (output / "deployment.json").write_text(json.dumps(record, indent=2) + "\n")
    export_onnx(path, sidecar, output / "policy.onnx")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--training", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--policy-kind", choices=["cts", "student-finetune"], default="cts")
    args = parser.parse_args()
    print(json.dumps(export(args.checkpoint, args.training, args.output, args.policy_kind), indent=2))


if __name__ == "__main__":
    main()
