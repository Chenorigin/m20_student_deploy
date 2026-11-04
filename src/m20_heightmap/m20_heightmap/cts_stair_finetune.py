"""Board-local identity for the complete frozen-v24 and forward student actor."""
import math
import re

SCOPE = 'frozen_v24_with_forward_student_nominal16'
VERSION = 'm20_cts_stair_finetune_v2'
TASK = 'Rough-Deeprobotics-M20-CTS-Stair-Finetune'
BASELINE_SHA256 = '8e268e6852ce0c64c98df730ea7e6d473dc689a89f394ed60ef1408c291babc6'
SOURCE = 'ascent_ppo.reference + ascent_ppo.forward_head'


def validate_stair_finetune_contract(actual):
    item = actual.get('stair_finetune')
    keys = {'task', 'version', 'baseline_sha256', 'checkpoint_sha256', 'head_sha256',
            'metadata_sha256', 'config_sha256', 'num_updates', 'forward_command_epsilon',
            'source_state_key'}
    if (actual.get('actor_scope') != SCOPE
            or actual.get('cts_version') != 'm20_cts_action_latent_v24'
            or not isinstance(item, dict) or set(item) != keys
            or item['task'] != TASK or item['version'] != VERSION
            or item['baseline_sha256'] != BASELINE_SHA256 or item['source_state_key'] != SOURCE
            or type(item['num_updates']) is not int or item['num_updates'] < 0
            or type(item['forward_command_epsilon']) is not float
            or item['forward_command_epsilon'] != 1.e-4):
        raise ValueError('Invalid CTS stair-finetune identity or forward gate')
    for key in ('baseline_sha256', 'checkpoint_sha256', 'head_sha256', 'metadata_sha256', 'config_sha256'):
        if not isinstance(item[key], str) or not re.fullmatch('[0-9a-f]{64}', item[key]):
            raise ValueError('Invalid CTS stair-finetune digest: ' + key)
    return item
