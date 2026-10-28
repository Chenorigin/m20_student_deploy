#!/usr/bin/env python3
"""Convert a validated student TorchScript export to a static single-input CPU ONNX."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import torch
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src/m20_heightmap"))
from m20_heightmap.contract import contract, validate_model_contract
from onnx_linear_precision import set_linear_precision


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--profile", choices=("v0", "v1"), required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--linear-precision", choices=("float64", "float32"), default="float64",
                    help="Linear compute precision; float32 produces a numerical comparison model")
    args = ap.parse_args()
    record_path = args.output.with_suffix(".onnx.json")
    if args.output.exists() or record_path.exists():
        raise ValueError("Use a new ONNX output path; do not overwrite an existing release")
    actual = json.loads(args.contract.read_text())
    if actual.get("deployment_family") == "m20_cts_student":
        if args.profile != "v1":raise ValueError("CTS requires profile v1")
        from export_cts_onnx import export_onnx
        print(json.dumps(export_onnx(args.model,args.contract,args.output),indent=2))
        return
    if "quiet_execution_control" in actual:
        raise ValueError("StairQuiet execution control is not implemented by the existing SDK")
    spec = contract(args.profile, actual.get("history_steps", 1))
    validate_model_contract(args.contract, spec)
    torch.set_num_threads(1)
    extra_files = {"quiet_execution_control.json": ""}
    actor = torch.jit.load(str(args.model), map_location="cpu", _extra_files=extra_files).eval()
    if extra_files["quiet_execution_control.json"]:
        raise ValueError("StairQuiet nominal actor cannot be exported without its execution controller")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample = torch.zeros(1, spec["input_dim"])
    buffer = io.BytesIO()
    torch.onnx.export(actor, sample, buffer, input_names=["obs"], output_names=["actions"],
                      opset_version=17, dynamo=False, do_constant_folding=True)
    graph = onnx.load_model_from_string(buffer.getvalue())
    properties = {item.key: item.value for item in graph.metadata_props}
    required_properties = {"deployment_family": "m20_student_v3_5",
                  "command_feedback": json.dumps(actual["command_feedback"], sort_keys=True, separators=(",", ":")),
                  "contract_sha256": hashlib.sha256(args.contract.read_bytes()).hexdigest()}
    for name, value in required_properties.items():
        if name in properties and properties[name] != value:
            raise ValueError("Conflicting source ONNX metadata: " + name)
        properties[name] = value
    onnx.helper.set_model_props(graph, properties)
    graph, precision, geometry = set_linear_precision(graph, args.linear_precision)
    graph_bytes = graph.SerializeToString()
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 1, 1
    options.add_session_config_entry("session.disable_prepacking", "1")
    session = ort.InferenceSession(graph_bytes, sess_options=options, providers=["CPUExecutionProvider"])
    if (len(session.get_inputs()) != 1 or session.get_inputs()[0].shape != [1, spec["input_dim"]]
            or session.get_inputs()[0].type != "tensor(float)"):
        raise ValueError("ONNX input must be static [1,student_dim]")
    if (len(session.get_outputs()) != 1 or session.get_outputs()[0].shape != [1, 16]
            or session.get_outputs()[0].type != "tensor(float)"):
        raise ValueError("ONNX output must be static [1,16]")
    rng = np.random.default_rng(42)
    worst = 0.
    for _ in range(32):
        x = rng.normal(size=(1, spec["input_dim"])).astype(np.float32)
        with torch.inference_mode():
            expected = actor(torch.from_numpy(x)).numpy()
        observed = session.run(["actions"], {"obs": x})[0]
        if not np.isfinite(observed).all():
            raise ValueError("ONNX returns nonfinite actions")
        np.testing.assert_allclose(observed, expected, atol=2e-5, rtol=2e-5)
        worst = max(worst, float(np.max(np.abs(expected-observed))))
    record = {"profile": args.profile, "input_shape": [1, spec["input_dim"]], "output_shape": [1, 16],
              "input_name": "obs", "output_name": "actions", "opset": 17, "providers": session.get_providers(),
              "policy_sha256": hashlib.sha256(graph_bytes).hexdigest(),
              "torchscript_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
              "source_model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
              "source_model_format": "torchscript",
              "source_onnx_sha256": precision["source_onnx_sha256"],
              "linear_precision": args.linear_precision, "precision": precision,
              "hidden_dims": geometry["hidden_dims"],
              "contract_sha256": hashlib.sha256(args.contract.read_bytes()).hexdigest(),
              "robot_training_contract_sha256": hashlib.sha256((args.contract.parent/"robot_training_contract.json").read_bytes()).hexdigest(),
              "parity_samples": 32, "parity_atol": 2e-5, "parity_rtol": 2e-5,
              "max_abs_error": worst, "scope": "CPU numerical export parity, not robot closed-loop"}
    with args.output.open("xb") as stream:
        stream.write(graph_bytes)
    with record_path.open("x") as stream:
        stream.write(json.dumps(record, indent=2)+"\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
