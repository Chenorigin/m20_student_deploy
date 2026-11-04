#!/usr/bin/env python3
"""Export the complete v2 stair actor; never starts robot control."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import torch
from export_cts_checkpoint import export_actor_bundle
from export_cts_idle_checkpoint import head_digest
from m20_heightmap.cts import canonical_hash, sha
from m20_heightmap.cts_stair_finetune import SOURCE, VERSION

def load_actor(checkpoint, training):
    sys.path.insert(0, str(training / 'source/rl_training'))
    sys.path.insert(0, str(training / 'deploy/deploy_mujoco'))
    from deploy_cts_stair_finetune import load_policy
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved.get('version') != VERSION:
        raise ValueError('This deployment adapter accepts stair-finetune v2 only')
    canonical, control = load_policy(checkpoint)
    provenance = canonical.stair_finetune_provenance
    state = saved['ascent_ppo']
    metadata = deepcopy(canonical.student_provenance)
    metadata['stair_finetune'] = dict(
        task=provenance['task'], version=provenance['version'],
        baseline_sha256=provenance['baseline_sha256'], checkpoint_sha256=sha(checkpoint),
        head_sha256=head_digest(canonical.policy.head.state_dict()),
        metadata_sha256=canonical_hash(provenance), config_sha256=canonical_hash(state['config']),
        num_updates=state['num_updates'], forward_command_epsilon=canonical.policy.epsilon,
        source_state_key=SOURCE)
    return canonical.policy, metadata, control

def export(checkpoint, training, output):
    return export_actor_bundle(checkpoint, training, output, load_actor)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'training', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.checkpoint, args.training, args.output), indent=2))

if __name__ == '__main__':
    main()
