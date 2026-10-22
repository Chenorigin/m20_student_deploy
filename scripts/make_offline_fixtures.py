#!/usr/bin/env python3
"""Create fixed CPU-only ARM parity fixtures from the selected export."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    torch.set_num_threads(1)
    fixtures = ROOT/'evidence/offline_models'
    if fixtures.exists():
        raise ValueError('keep the existing fixed fixtures; use a new release')
    (fixtures/'v1').mkdir(parents=True)
    actor = torch.jit.load(str(ROOT/'models/v1/policy.pt'),map_location='cpu').eval()
    rng = np.random.default_rng(42)
    random = rng.normal(size=(32,789)).astype('float32')
    x = np.zeros((128,789),dtype='float32')
    x[:,:57] = rng.normal(0,.2,(128,57)); x[:,5] = -1.
    x[:,57:244] = rng.uniform(-.3,.3,(128,187))
    x[:,244:431] = rng.integers(0,2,(128,187))
    x[:,431:618] = rng.uniform(0,1,(128,187))*x[:,244:431]
    x[:,57:244] *= x[:,244:431]
    x[:,618:] = rng.normal(0,.2,(128,171))
    x = np.concatenate([random,x])
    with torch.inference_mode():
        y = np.concatenate([actor(torch.from_numpy(a[None])).numpy() for a in x])
    x.tofile(fixtures/'v1/input.f32'); y.tofile(fixtures/'v1/expected.f32')
    historical = ROOT/'evidence/historical_observations.f32'
    if historical.is_file():
        h=np.fromfile(historical,dtype='float32').reshape(-1,789)
        output=fixtures/'historical_v3';output.mkdir()
        with torch.inference_mode():
            labels=np.concatenate([actor(torch.from_numpy(a[None])).numpy() for a in h])
        historical.rename(output/'input.f32');labels.tofile(output/'expected.f32')
    tests = load('heading_scenarios',ROOT/'tests/test_heading_feedback.py')
    reference = load('heading_reference',ROOT/'scripts/reference_training_feedback.py')
    metadata = json.loads((ROOT/'models/v1/observation_contract.json').read_text())['command_feedback']
    cases,_ = tests.scenarios()
    requests = [dict(metadata=metadata,operations=ops) for ops in cases.values()]
    expected = [dict(results=tests.reference_results(reference,metadata,ops)) for ops in cases.values()]
    (fixtures/'heading_input.json').write_text(json.dumps(requests,allow_nan=False))
    (fixtures/'heading_expected.json').write_text(json.dumps(expected,allow_nan=False))
    (fixtures/'manifest.json').write_text(json.dumps(dict(onnx_frames=len(x),heading_scenarios=len(cases),hardware_commands=0),indent=2)+'\n')
    print('Offline CPU fixtures: '+str(fixtures))


if __name__ == '__main__':
    main()
