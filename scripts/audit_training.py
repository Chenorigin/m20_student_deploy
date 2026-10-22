#!/usr/bin/env python3
"""Read local training sources without Isaac. Optional CPU Torch reference parity."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src/m20_heightmap"))
from m20_heightmap.contract import contract, validate_model_contract
from m20_heightmap.estimator import EstimatorConfig, confidence, edge_variance


def defaults(path, class_name):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return {n.target.id: ast.literal_eval(n.value) for n in cls.body if isinstance(n, ast.AnnAssign)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training", type=Path, default=ROOT.parent/"rl_training")
    ap.add_argument("--output", type=Path, default=ROOT/"evidence/training_audit.json")
    ap.add_argument("--torch-parity", action="store_true")
    args = ap.parse_args()
    source = args.training/"source/rl_training"
    sys.path.insert(0, str(source))
    from m20_student_v0.contract import observation_contract as v0
    from m20_student_v1.contract import observation_contract as v1
    contracts = {"v0": v0(), "v1_history4": v1(4)}
    for name, actual in contracts.items():
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
            json.dump(actual, f); f.flush()
            validate_model_contract(f.name, contract("v0" if name == "v0" else "v1"))
    paths = [source/p for p in ("m20_student_v0/contract.py", "m20_student_v1/contract.py",
                               "m20_student_v0/perception.py", "m20_student_v1/perception.py", "m20_student_v1/configuration.py")]
    result = {"sources": {str(p.relative_to(args.training)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              "contracts": contracts,
              "v0_training_defaults": defaults(source/"m20_student_v0/perception.py", "StudentV0PerceptionCfg"),
              "v1_training_defaults": defaults(source/"m20_student_v1/configuration.py", "StudentV1PerceptionCfg"),
              "contract_validation": "passed", "torch_parity": "skipped: pass --torch-parity in training Python"}
    if args.torch_parity:
        import torch
        from m20_student_v0.perception import confidence_from_quality as c0, StudentV0PerceptionCfg
        from m20_student_v1.perception import confidence_from_quality as c1, pose_height_variance, grid_points, quaternion_rotation
        from m20_student_v1.configuration import StudentV1PerceptionCfg
        from m20_heightmap.geometry import rotation
        torch.set_num_threads(1)
        rng = np.random.default_rng(17)
        valid = rng.random(187) > .2
        var, age = rng.uniform(0, .02, 187), rng.uniform(0, .4, 187)
        ours = confidence(valid, var, age, EstimatorConfig())
        errors = {}
        for name, fun, cfg in (("v0", c0, StudentV0PerceptionCfg()), ("v1", c1, StudentV1PerceptionCfg())):
            expected = fun(torch.tensor(valid), torch.tensor(var), torch.tensor(age), cfg).numpy()
            errors[name+"_confidence_max_error"] = float(np.max(np.abs(ours-expected)))
            np.testing.assert_allclose(ours, expected, atol=1e-12)
        z = rng.normal(0, .1, (1, 187))
        expected = pose_height_variance(torch.tensor(z), torch.tensor(valid[None]), grid_points("cpu").double(),
                                        torch.tensor([.015]), torch.tensor([.01]), torch.tensor([0.])).numpy()[0]
        ours = edge_variance(z[0], valid, np.full(187,.015**2), np.full(187,.01**2))
        errors["v1_edge_variance_max_error"] = float(np.max(np.abs(ours-expected)))
        np.testing.assert_allclose(ours, expected, atol=1e-8)
        q = rng.normal(size=(200,4)); q /= np.linalg.norm(q,axis=1)[:,None]
        expected = quaternion_rotation(torch.tensor(q[:,[3,0,1,2]])).numpy()
        actual = np.stack([rotation(v) for v in q])
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        errors["rotation_max_error"] = float(np.max(np.abs(actual-expected)))
        result["torch_parity"] = errors
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"output": str(args.output), "contract_validation": result["contract_validation"], "torch_parity": result["torch_parity"]}, indent=2))


if __name__ == "__main__":
    main()
