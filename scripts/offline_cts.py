#!/usr/bin/env python3
"""Run fixed CTS parity cases via CPU ORT or the real C++ board probe."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/m20_heightmap"))
from m20_heightmap.cts import CTSInference, validate_embedded
from validate_cts_bundle import validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/v1")
    parser.add_argument("--probe", type=Path, help="ARM: installed student_onnx_probe; no Python ORT required")
    args = parser.parse_args()
    model = args.model_dir.resolve()
    report = validate(model, check_model=args.probe is None)
    inputs = np.fromfile(model / "input.f32", dtype="<f4").reshape(-1, 792)
    expected = np.fromfile(model / "expected.f32", dtype="<f4").reshape(-1, 16)
    if not len(inputs) or len(inputs) != len(expected) or not np.isfinite(inputs).all():
        raise ValueError("Empty, truncated or invalid CTS fixtures")
    if args.probe:
        result = subprocess.run([str(args.probe.resolve()), str(model / "policy.onnx")],
                                capture_output=True, text=True, check=True, timeout=30)
        sidecar = model / "observation_contract.json"
        validate_embedded(json.loads(result.stdout), json.loads(sidecar.read_text()), sidecar)
        scratch = ROOT / "runtime"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="cts_offline_", dir=scratch) as directory:
            output = Path(directory) / "actions.f32"
            result = subprocess.run([str(args.probe.resolve()), str(model / "policy.onnx"),
                                     str(model / "input.f32"), str(output)],
                                    capture_output=True, text=True, check=False, timeout=60)
            if result.returncode:
                raise RuntimeError("CTS probe failed: " + result.stderr)
            report["probe"] = json.loads(result.stdout)
            actual = np.fromfile(output, dtype="<f4").reshape(-1, 16)
    else:
        session = CTSInference(model / "policy.onnx", model / "observation_contract.json")
        actual = np.concatenate([session(row[None, :789], row[None, 789:]) for row in inputs])
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError("CTS output shape/finiteness mismatch")
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
    report.update(status="passed", frames=len(inputs), max_abs_error=float(np.abs(actual - expected).max()),
                  scope="nominal student+decoder parity only; no physical control acceptance")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
