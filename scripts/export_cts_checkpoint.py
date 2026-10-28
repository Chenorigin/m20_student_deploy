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


def load_actor(checkpoint, training):
    # Use the production loader's metadata/state validation, not a reconstructed
    # approximation or the initialization student_state_dict.
    sys.path.insert(0, str(training / "source/rl_training"))
    sys.path.insert(0, str(training / "deploy/deploy_mujoco"))
    from deploy_cts_student import load_policy
    actor, control = load_policy(checkpoint)
    return actor.policy, actor.student_provenance, control


def export(checkpoint, training, output):
    import torch
    from export_cts_onnx import export_onnx
    torch.set_num_threads(1)
    checkpoint, training, output = map(lambda p: Path(p).resolve(), (checkpoint, training, output))
    if output.exists() and any(output.iterdir()):
        raise ValueError("CTS export requires a new/empty model directory")
    checkpoint_sha = sha(checkpoint)
    actor, meta, control = load_actor(checkpoint, training)
    actual = deepcopy(meta["contract"])
    actual.update(deployment_family=FAMILY, cts_version=meta["version"], model_inputs=INPUTS,
                  raw_command_semantics="requested_command_before_heading_feedback",
                  actor_scope="student_encoder_and_shared_decoder_nominal16",
                  standalone_robot_controller=False,
                  command_feedback=deepcopy(meta["command_feedback"]),
                  perception=deepcopy(meta["perception"]), quiet_execution_control=control,
                  quiet_execution_control_sha256=canonical_hash(control))
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
              "source_state_key": "algorithm.models_state_dict.student + decoder",
              "quiet_execution_control_sha256": actual["quiet_execution_control_sha256"],
              "standalone_robot_controller": False, "hardware_commands": 0,
              "export_check": {"samples": 128, "max_abs_error": float((observed-expected).abs().max()),
                               "device": "cpu"}}
    (output / "deployment.json").write_text(json.dumps(record, indent=2) + "\n")
    export_onnx(path, sidecar, output / "policy.onnx")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--training", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.checkpoint, args.training, args.output), indent=2))


if __name__ == "__main__":
    main()
