#!/usr/bin/env python3
"""Validate a self-contained CTS nominal bundle, without training or hardware."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/m20_heightmap"))
from m20_heightmap.contract import contract, validate_model_contract
from m20_heightmap.cts import FAMILY, INPUTS, sha, CTSInference


def validate(model_dir, check_model=True):
    model_dir = Path(model_dir)
    sidecar = model_dir / "observation_contract.json"
    actual = validate_model_contract(sidecar, contract("v1", 4))
    if actual.get("deployment_family") != FAMILY:
        raise ValueError("Not a CTS bundle")
    deployed = json.loads((model_dir / "deployment.json").read_text())
    onnx = json.loads((model_dir / "policy.onnx.json").read_text())
    for record in (deployed, onnx):
        if (record.get("deployment_family") != FAMILY or record.get("model_inputs") != INPUTS
                or record.get("contract_sha256") != sha(sidecar)):
            raise ValueError("CTS model/contract digest or input schema mismatch")
    if (deployed["policy_sha256"] != sha(model_dir / "policy.pt")
            or onnx["torchscript_sha256"] != deployed["policy_sha256"]
            or onnx["policy_sha256"] != sha(model_dir / "policy.onnx")
            or deployed.get("quiet_execution_control_sha256") != actual["quiet_execution_control_sha256"]
            or deployed.get("standalone_robot_controller") is not False):
        raise ValueError("CTS model identity/execution provenance mismatch")
    for name in ("input.f32", "expected.f32"):
        if onnx.get("fixture_sha256", {}).get(name) != sha(model_dir / name):
            raise ValueError("CTS fixture changed: " + name)
    if check_model:
        CTSInference(model_dir / "policy.onnx", sidecar)
    return {"passed": True, "deployment_family": FAMILY, "model_inputs": INPUTS,
            "checkpoint_sha256": deployed["checkpoint_sha256"],
            "onnx_sha256": onnx["policy_sha256"], "hardware_ready": False, "hardware_commands": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/v1")
    parser.add_argument("--metadata-only", action="store_true", help="GOS: validate bundled digests without Python ORT")
    args = parser.parse_args()
    print(json.dumps(validate(args.model_dir, check_model=not args.metadata_only), indent=2))


if __name__ == "__main__":
    main()
