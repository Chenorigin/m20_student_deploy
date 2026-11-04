"""Complete new actor export, strict new contracts, and original CTS isolation."""
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from export_cts_stair_noise_checkpoint import export, load_actor
from export_cts_checkpoint import export as export_cts
from validate_cts_bundle import validate
from m20_heightmap.cts import CTSInference, validate_cts_contract
from m20_heightmap.cts_stair_noise import SCOPE


@pytest.fixture(scope='module')
def candidate(tmp_path_factory):
    torch = pytest.importorskip('torch')
    torch.set_num_threads(1)
    training = Path(os.environ.get('CTS_STAIR_NOISE_TRAINING_ROOT', str(ROOT.parent / 'rl_training')))
    value = os.environ.get('CTS_STAIR_NOISE_TEST_CHECKPOINT')
    if not value or not Path(value).is_file():
        pytest.skip('Set CTS_STAIR_NOISE_TEST_CHECKPOINT to a saved stair_noise checkpoint for real export tests')
    checkpoint = Path(value)
    output = tmp_path_factory.mktemp('stair_noise_export') / 'bundle'
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
    assert sidecar['stair_noise']['num_updates'] >= 0


def test_onnx_captures_nonzero_forward_head_and_raw_gate(candidate):
    import torch
    output, actor, _, _ = candidate
    runtime = CTSInference(output / 'policy.onnx', output / 'observation_contract.json')
    inputs = np.fromfile(output / 'input.f32', dtype='<f4').reshape(-1, 792)
    stair_noise_difference = 0.
    for index, row in enumerate(inputs):
        obs = row[None, :789].copy()
        obs[0, 6:9] = [.7, -.4, .4]  # Gate uses raw vx, never processed sensor commands.
        raw = row[None, 789:]
        with torch.no_grad():
            expected = actor(torch.from_numpy(obs), torch.from_numpy(raw)).numpy()
            base = actor.reference(torch.from_numpy(obs), torch.from_numpy(raw)).numpy()
        actual = runtime(obs, raw)
        np.testing.assert_allclose(actual, expected, atol=2.e-5, rtol=2.e-5)
        if raw[0, 0] <= 1.e-4:
            np.testing.assert_array_equal(expected, base)
        else:
            stair_noise_difference = max(stair_noise_difference, float(np.abs(expected - base).max()))
    # Nonzero trained checkpoint required: zero head alone cannot catch omission.
    assert stair_noise_difference > 1.e-6


@pytest.mark.parametrize('raw', [[0., .4, 0.], [0., 0., .4], [-.2, .1, .3],
                               [1.e-4, .4, .3], [1.0001e-4, .4, .3], [.4, .2, .3]])
def test_forward_boundary_and_mixed_commands(candidate, raw):
    import torch
    output, actor, _, _ = candidate
    runtime = CTSInference(output / 'policy.onnx', output / 'observation_contract.json')
    obs = np.fromfile(output / 'input.f32', dtype='<f4').reshape(-1, 792)[:1, :789].copy()
    obs[:, 6:9] = [.7, -.5, -.3]
    command = np.array([raw], dtype=np.float32)
    with torch.no_grad():
        expected = actor(torch.from_numpy(obs), torch.from_numpy(command)).numpy()
        base = actor.reference(torch.from_numpy(obs), torch.from_numpy(command)).numpy()
    np.testing.assert_allclose(runtime(obs, command), expected, atol=2e-5, rtol=2e-5)
    if raw[0] <= 1.e-4:
        np.testing.assert_array_equal(expected, base)


@pytest.mark.parametrize('change', ['scope', 'task', 'version', 'baseline', 'gate', 'bounds', 'nan', 'missing', 'digest', 'counter', 'mixed'])
def test_new_identity_tampering_rejected(candidate, change):
    output, _, _, _ = candidate
    actual = json.loads((output / 'observation_contract.json').read_text())
    item = actual['stair_noise']
    if change == 'scope': actual['actor_scope'] = 'student_encoder_and_shared_decoder_nominal16'
    elif change == 'task': item['task'] = 'other'
    elif change == 'version': actual['cts_version'] = 'm20_cts_action_latent_v17'
    elif change == 'baseline': item['baseline_sha256'] = '0' * 64
    elif change == 'gate': item['forward_command_epsilon'] = .01
    elif change == 'bounds': item['residual_bounds'] = [1.] * 15
    elif change == 'nan': item['residual_bounds'][0] = float('nan')
    elif change == 'missing': del actual['stair_noise']
    elif change == 'digest': item['head_sha256'] = 'invalid'
    elif change == 'counter': item['num_updates'] = True
    elif change == 'mixed': actual['idle_finetune'] = {}
    with pytest.raises(ValueError):
        validate_cts_contract(actual)


def test_ordinary_cts_loader_does_not_accept_stair_noise_checkpoint(candidate, tmp_path):
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
    record['stair_noise']['head_sha256'] = '1' * 64
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='provenance'):
        validate(clone, check_model=False)


def test_one_command_selects_complete_bundle_without_ssh(candidate, capsys):
    from start_cts_stair_noise import main
    output, _, _, training = candidate
    assert main(['--training', str(training), '--bundle', str(output),
                 '--aos', 'test-aos', '--gos', 'test-gos', '--dry-run']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['argv'][0] == 'start'
    assert result['argv'][-2:] == ['--bundle', str(output)]
    assert result['ssh_connections'] == result['hardware_commands'] == 0


def test_one_command_stop_needs_neither_bundle_nor_gos(candidate, capsys):
    from start_cts_stair_noise import main
    _, _, _, training = candidate
    assert main(['stop', '--training', str(training), '--aos', 'test-aos', '--dry-run']) == 0
    result = json.loads(capsys.readouterr().out)
    assert '--bundle' not in result['argv'] and '--gos' not in result['argv']


def test_one_command_missing_bundle_fails_before_connection(candidate):
    from start_cts_stair_noise import main
    output, _, _, training = candidate
    with pytest.raises(FileNotFoundError):
        main(['--training', str(training), '--bundle', str(output/'missing'),
              '--aos', 'test-aos', '--gos', 'test-gos', '--dry-run'])
