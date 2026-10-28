#!/usr/bin/env python3
"""Convert an embedded CTS TorchScript contract to a two-input nominal ONNX."""
import argparse
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/m20_heightmap"))
from m20_heightmap.contract import contract, validate_model_contract
from m20_heightmap.cts import FAMILY, INPUTS, sha, CTSInference, front_mask


def export_onnx(model, sidecar, output):
    import numpy as np
    import onnx
    import torch
    torch.set_num_threads(1)
    output = Path(output)
    receipt = output.with_suffix(".onnx.json")
    if output.exists() or receipt.exists():
        raise ValueError("CTS ONNX output must be new")
    actual = validate_model_contract(sidecar, contract("v1", 4))
    if actual.get("deployment_family") != FAMILY:
        raise ValueError("Expected a CTS contract")
    extras = {"quiet_execution_control.json": "", "cts_contract.json": ""}
    actor = torch.jit.load(str(model), map_location="cpu", _extra_files=extras).eval()
    if (json.loads(extras["cts_contract.json"]) != actual or
            json.loads(extras["quiet_execution_control.json"]) != actual["quiet_execution_control"]):
        raise ValueError("CTS TorchScript embedded/sidecar contracts differ")
    stream = io.BytesIO()
    torch.onnx.export(actor, (torch.zeros(1, 789), torch.zeros(1, 3)), stream,
                      input_names=list(INPUTS), output_names=["actions"], opset_version=17,
                      dynamo=False, do_constant_folding=True)
    graph = onnx.load_model_from_string(stream.getvalue())
    onnx.helper.set_model_props(graph, {
        "deployment_family": FAMILY, "contract_sha256": sha(sidecar),
        "quiet_execution_control_sha256": actual["quiet_execution_control_sha256"],
        "command_feedback": json.dumps(actual["command_feedback"], sort_keys=True),
        "quiet_execution_control": json.dumps(actual["quiet_execution_control"], sort_keys=True)})
    onnx.checker.check_model(graph)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(graph.SerializeToString())
    session = CTSInference(output, sidecar)
    rng = np.random.default_rng(42)
    samples, labels = [], []
    worst = 0.
    for index in range(128):
        x = rng.normal(0, .2, (1, 789)).astype(np.float32)
        x[:, 5] = -1
        x[:, 57:244] = rng.uniform(-.3, .3, (1, 187))
        mask = rng.integers(0, 2, (1, 187))
        mask[:, ~front_mask(actual)] = 0
        if index % 4 == 0:
            mask[:] = 0
        x[:, 244:431] = mask
        x[:, 57:244] *= mask
        x[:, 431:618] = rng.uniform(0, 1, (1, 187)) * mask
        raw = rng.uniform(-1, 1, (1, 3)).astype(np.float32)
        if index % 2 == 0:
            raw[:] = 0
        # Include gate boundaries and effective/raw disagreement explicitly.
        if index < 6:
            raw[:] = (0, 1.e-5, 1.e-4, 1.0001e-4, -1.e-4, -.1)[index]
        with torch.inference_mode():
            expected = actor(torch.from_numpy(x), torch.from_numpy(raw)).numpy()
        observed = session(x, raw)
        np.testing.assert_allclose(observed, expected, atol=2e-5, rtol=2e-5)
        worst = max(worst, float(np.abs(observed - expected).max()))
        samples.append(np.concatenate((x[0], raw[0])))
        labels.append(expected[0])
    # Explicit packed fixture format for the C++ offline probe, not a policy
    # observation or a network ABI. Raw commands always come from a separate source.
    np.asarray(samples, dtype="<f4").tofile(output.parent / "input.f32")
    np.asarray(labels, dtype="<f4").tofile(output.parent / "expected.f32")
    record = {"deployment_family": FAMILY, "model_inputs": INPUTS, "output_shape": [1, 16],
              "policy_sha256": sha(output), "torchscript_sha256": sha(model),
              "contract_sha256": sha(sidecar), "linear_precision": "float32",
              "fixture_sha256": {n: sha(output.parent / n) for n in ("input.f32", "expected.f32")},
              "fixture_columns": 792, "parity_samples": 128,
              "parity_atol": 2e-5, "parity_rtol": 2e-5, "max_abs_error": worst,
              "scope": "CPU nominal policy parity only; mandatory landing hardware binding absent"}
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(export_onnx(args.model, args.contract, args.output), indent=2))


if __name__ == "__main__":
    main()
