"""Complete new actor export, strict new contracts, and original CTS isolation."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from export_cts_idle_checkpoint import export, load_actor
from export_cts_checkpoint import export as export_cts
from validate_cts_bundle import validate
from m20_heightmap.cts import CTSInference, canonical_hash, validate_cts_contract
from m20_heightmap.cts_idle import SCOPE


@pytest.fixture(scope='module')
def candidate(tmp_path_factory):
    torch = pytest.importorskip('torch')
    torch.set_num_threads(1)
    training = Path(os.environ.get('CTS_IDLE_TRAINING_ROOT', str(ROOT.parent / 'rl_training')))
    value = os.environ.get('CTS_IDLE_TEST_CHECKPOINT')
    if not value or not Path(value).is_file():
        pytest.skip('Set CTS_IDLE_TEST_CHECKPOINT to a saved idle checkpoint for real export tests')
    checkpoint = Path(value)
    output = tmp_path_factory.mktemp('idle_export') / 'bundle'
    export(checkpoint, training, output)
    actor, meta, _ = load_actor(checkpoint, training)
    return output, actor, checkpoint, training


def test_complete_self_contained_bundle(candidate):
    output, actor, _, _ = candidate
    assert validate(output)['passed']
    assert validate(output, check_model=False)['passed']
    assert {p.name for p in output.iterdir()} == {'policy.pt', 'policy.onnx', 'observation_contract.json',
        'deployment.json', 'policy.onnx.json', 'input.f32', 'expected.f32'}
    sidecar = json.loads((output / 'observation_contract.json').read_text())
    assert sidecar['actor_scope'] == SCOPE
    assert sidecar['idle_finetune']['num_updates'] >= 0


def test_onnx_captures_nonzero_idle_head_and_raw_gate(candidate):
    import torch
    output, actor, _, _ = candidate
    runtime = CTSInference(output / 'policy.onnx', output / 'observation_contract.json')
    inputs = np.fromfile(output / 'input.f32', dtype='<f4').reshape(-1, 792)
    idle_difference = 0.
    for index, row in enumerate(inputs):
        obs = row[None, :789].copy()
        obs[0, 8] = .4  # Effective yaw must not disable a raw stop request.
        raw = row[None, 789:]
        with torch.no_grad():
            expected = actor(torch.from_numpy(obs), torch.from_numpy(raw)).numpy()
            base = actor.reference(torch.from_numpy(obs), torch.from_numpy(raw)).numpy()
        actual = runtime(obs, raw)
        np.testing.assert_allclose(actual, expected, atol=2.e-5, rtol=2.e-5)
        if np.max(np.abs(raw)) > 1.e-4:
            np.testing.assert_array_equal(expected, base)
        else:
            idle_difference = max(idle_difference, float(np.abs(expected - base).max()))
    # Nonzero trained checkpoint required: zero head alone cannot catch omission.
    assert idle_difference > 1.e-6


@pytest.mark.parametrize('change', ['scope', 'task', 'version', 'baseline', 'gate', 'bounds', 'nan', 'missing', 'digest', 'counter'])
def test_new_identity_tampering_rejected(candidate, change):
    output, _, _, _ = candidate
    actual = json.loads((output / 'observation_contract.json').read_text())
    item = actual['idle_finetune']
    if change == 'scope': actual['actor_scope'] = 'student_encoder_and_shared_decoder_nominal16'
    elif change == 'task': item['task'] = 'other'
    elif change == 'version': actual['cts_version'] = 'm20_cts_action_latent_v17'
    elif change == 'baseline': item['baseline_sha256'] = '0' * 64
    elif change == 'gate': item['idle_command_epsilon'] = .01
    elif change == 'bounds': item['residual_bounds'] = [1.] * 15
    elif change == 'nan': item['residual_bounds'][0] = float('nan')
    elif change == 'missing': del actual['idle_finetune']
    elif change == 'digest': item['head_sha256'] = 'invalid'
    elif change == 'counter': item['num_updates'] = True
    with pytest.raises(ValueError):
        validate_cts_contract(actual)


def test_ordinary_cts_loader_does_not_accept_idle_checkpoint(candidate, tmp_path):
    _, _, checkpoint, training = candidate
    with pytest.raises((ValueError, KeyError)):
        export_cts(checkpoint, training, tmp_path / 'wrong_export')


def test_bundle_provenance_is_bound_to_contract(candidate, tmp_path):
    import shutil
    output, _, _, _ = candidate
    clone = tmp_path / 'tampered'
    shutil.copytree(output, clone)
    path = clone / 'deployment.json'
    record = json.loads(path.read_text())
    record['idle_finetune']['head_sha256'] = '1' * 64
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='provenance'):
        validate(clone, check_model=False)
