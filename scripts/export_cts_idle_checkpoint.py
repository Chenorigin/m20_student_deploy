#!/usr/bin/env python3
"""Export the complete frozen-v24 idle actor using the existing CTS pipeline."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

from export_cts_checkpoint import export_actor_bundle
from m20_heightmap.cts import canonical_hash, sha
from m20_heightmap.cts_idle import SOURCE


def head_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(',', ':')).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_actor(checkpoint, training):
    sys.path.insert(0, str(training / 'source/rl_training'))
    sys.path.insert(0, str(training / 'deploy/deploy_mujoco'))
    from deploy_cts_idle_finetune import load_policy
    actor, control = load_policy(checkpoint)
    # Canonical loader verifies baseline SHA, frozen tensors, saved head and
    # configuration. Export the actual complete actor, never reconstruct it.
    import torch
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    metadata = deepcopy(actor.student_provenance)
    provenance = actor.finetune_provenance
    cfg = actor.idle_finetune_config
    metadata['idle_finetune'] = dict(
        task=provenance['task'], version=provenance['version'],
        baseline_sha256=provenance['baseline_sha256'], checkpoint_sha256=sha(checkpoint),
        head_sha256=head_digest(actor.policy.head.state_dict()),
        metadata_sha256=canonical_hash(provenance), config_sha256=canonical_hash(cfg),
        num_updates=saved['idle_ppo']['num_updates'],
        idle_command_epsilon=cfg['idle_command_epsilon'],
        residual_bounds=deepcopy(cfg['residual_bounds']), source_state_key=SOURCE)
    return actor.policy, metadata, control


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
