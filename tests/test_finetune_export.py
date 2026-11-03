"""Finetune export admission and actual policy parity. No robot connections."""
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from export_cts_checkpoint import export, load_actor


def test_unknown_policy_kind_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown policy kind"):
        load_actor(tmp_path / "missing.pt", tmp_path, "unknown")


@pytest.fixture
def live_checkpoint():
    training = os.environ.get("M20_TEST_TRAINING")
    checkpoint = os.environ.get("M20_TEST_FINETUNE_CHECKPOINT")
    if not training or not checkpoint:
        pytest.skip("Set explicit M20_TEST_TRAINING and M20_TEST_FINETUNE_CHECKPOINT for real checkpoint tests")
    import torch
    torch.set_num_threads(1)
    return Path(training), Path(checkpoint), torch.load(checkpoint, map_location="cpu", weights_only=True)


def test_exports_finetuned_weights_instead_of_baseline(tmp_path, live_checkpoint):
    import numpy as np
    import torch
    from m20_heightmap.cts import CTSInference
    training, checkpoint, saved = live_checkpoint
    original, _, _ = load_actor(checkpoint, training, "student-finetune")
    # Deliberately different actor, preserving the frozen decoder and ABI.
    key = next(k for k in saved["student_ppo"]["models"] if k.startswith("student.mlp.") and k.endswith("weight"))
    saved["student_ppo"]["models"][key] = saved["student_ppo"]["models"][key] + .1
    saved["student_ppo"]["num_updates"] = 1
    changed = tmp_path / "changed.pt"
    torch.save(saved, changed)
    actor, _, _ = load_actor(changed, training, "student-finetune")
    x = torch.randn(1, 789, generator=torch.Generator().manual_seed(42))
    raw = torch.tensor([[.3, 0., 0.]])
    with torch.inference_mode():
        expected = actor(x, raw).numpy()
        baseline = original(x, raw).numpy()
    assert np.max(np.abs(expected - baseline)) > 1e-4
    report = export(changed, training, tmp_path / "bundle", "student-finetune")
    observed = CTSInference(tmp_path / "bundle/policy.onnx", tmp_path / "bundle/observation_contract.json")(x.numpy(), raw.numpy())
    np.testing.assert_allclose(observed, expected, atol=2e-5, rtol=2e-5)
    assert report["student_finetune"]["num_updates"] == 1
    assert report["source_state_key"].startswith("student_ppo.models")


@pytest.mark.parametrize("change", ["diagnostic", "version", "ppo_version", "config_version", "counter", "baseline", "decoder"])
def test_rejects_corrupt_or_diagnostic_finetune(tmp_path, live_checkpoint, change):
    import torch
    training, _, saved = live_checkpoint
    if change == "diagnostic":
        saved["metadata"]["reference_limit_interruption"] = True
    elif change == "version":
        saved["version"] = "unknown"
    elif change == "ppo_version":
        saved["student_ppo"]["version"] = "unknown"
    elif change == "counter":
        saved["student_ppo"]["num_updates"] = -1
    elif change == "config_version":
        saved["metadata"]["config"]["version"] = "unknown"
        saved["student_ppo"]["config"]["version"] = "unknown"
    elif change == "baseline":
        saved["metadata"]["baseline_sha256"] = "0" * 64
    else:
        key = next(k for k in saved["student_ppo"]["models"] if k.startswith("decoder.") and k.endswith("weight"))
        saved["student_ppo"]["models"][key] = saved["student_ppo"]["models"][key] + .1
    path = tmp_path / "invalid.pt"
    torch.save(saved, path)
    with pytest.raises(ValueError):
        export(path, training, tmp_path / "bundle", "student-finetune")
    assert not (tmp_path / "bundle").exists()
