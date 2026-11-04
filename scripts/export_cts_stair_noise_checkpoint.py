#!/usr/bin/env python3
"""Export and verify the whole stair-noise actor through the existing CTS ABI."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import torch
from torch import nn

from export_cts_checkpoint import export_actor_bundle
from export_cts_idle_checkpoint import head_digest
from m20_heightmap.cts import canonical_hash, sha
from m20_heightmap.cts_stair_noise import SOURCE


class ExportActor(nn.Module):
    """Scriptable residual composition; exported weights are the canonical actor's.

    The training actor uses a Python global gate constant, which TorchScript
    cannot capture. Keep the gate as an attribute without editing live training.
    load_actor verifies this composition against the actual canonical policy.
    """
    def __init__(self, actor, epsilon):
        super().__init__()
        self.reference = actor.reference
        self.head = actor.head
        self.register_buffer('bounds', actor.bounds.detach().clone())
        self.forward_command_epsilon = epsilon

    def forward(self, policy, raw_command):
        base = self.reference(policy, raw_command)
        residual = torch.tanh(self.head(torch.cat((policy, raw_command, base), -1))) * self.bounds
        return torch.where((raw_command[:, 0] > self.forward_command_epsilon)[:, None], base + residual, base)


def load_actor(checkpoint, training):
    sys.path.insert(0, str(training / 'source/rl_training'))
    sys.path.insert(0, str(training / 'deploy/deploy_mujoco'))
    from deploy_cts_stair_noise import load_policy
    from task_support.cts_stair_noise.configuration import FORWARD_EPSILON
    canonical, control = load_policy(checkpoint)
    actor = ExportActor(canonical.policy, FORWARD_EPSILON).eval().requires_grad_(False)
    generator = torch.Generator().manual_seed(73)
    inputs = torch.randn(256, 789, generator=generator)
    commands = torch.randn(256, 3, generator=generator)
    commands[:8] = torch.tensor([[0., 0., 0.], [0., .3, 0.], [0., 0., .3],
        [-.3, 0., 0.], [1.e-4, 0., .3], [1.0001e-4, 0., .3], [.3, .2, .1], [.3, 0., 0.]])
    with torch.inference_mode():
        expected = canonical.policy(inputs, commands)
        observed = actor(inputs, commands)
        torch.testing.assert_close(observed, expected, atol=0., rtol=0.)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    metadata = deepcopy(canonical.student_provenance)
    provenance, cfg = canonical.finetune_provenance, canonical.stair_noise_config
    metadata['stair_noise'] = dict(
        task=provenance['task'], version=provenance['version'],
        baseline_sha256=provenance['baseline_sha256'], checkpoint_sha256=sha(checkpoint),
        head_sha256=head_digest(canonical.policy.head.state_dict()),
        metadata_sha256=canonical_hash(provenance), config_sha256=canonical_hash(cfg),
        num_updates=saved['stair_noise_ppo']['num_updates'],
        forward_command_epsilon=FORWARD_EPSILON,
        residual_bounds=deepcopy(cfg['residual_bounds']), source_state_key=SOURCE)
    return actor, metadata, control


def export(checkpoint, training, output):
    return export_actor_bundle(checkpoint, training, output, load_actor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--training', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.checkpoint, args.training, args.output), indent=2))


if __name__ == '__main__':
    main()
