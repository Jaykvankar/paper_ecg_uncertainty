"""Locked paired-experiment settings. Diagnosis is never an input."""
from pathlib import Path

import yaml

PRIMARY_OUTCOME = 'delta_classwise_ece'
UNCERTAINTY_SCORE = 'delta_mean_binary_entropy'
PRIMARY_LEVEL = 'creased'
ARMS = ('standardized', 'raw_mv')


def load_protocol(path='configs/paired.yaml'):
    protocol = yaml.safe_load(Path(path).read_text())
    if protocol.get('diagnosis_as_input') is not False:
        raise ValueError('Diagnosis is an outcome, never an input')
    if protocol.get('primary_outcome') != PRIMARY_OUTCOME:
        raise ValueError(f'Primary outcome must be {PRIMARY_OUTCOME}')
    if protocol.get('uncertainty_score') != UNCERTAINTY_SCORE:
        raise ValueError(f'Uncertainty score must be {UNCERTAINTY_SCORE}')
    if protocol.get('primary_level') != PRIMARY_LEVEL:
        raise ValueError(f'Primary page must be {PRIMARY_LEVEL}')
    if list(protocol.get('arms', [])) != list(ARMS):
        raise ValueError('Arms must be standardized, then raw_mv')
    resolved = {}
    for name in ARMS:
        config_path = Path(protocol['arm_configs'][name])
        arm = yaml.safe_load(config_path.read_text())
        if arm.get('device') != 'cuda':
            raise ValueError(f'{name} must set device: cuda')
        resolved[name] = {
            'seed': arm['seed'], 'width': arm['width'],
            'blocks_per_stage': arm['blocks_per_stage'],
            'standardize': arm['standardize'], 'device': arm['device'],
            'config': str(config_path).replace('\\', '/'),
            'results': protocol['arm_results'][name],
        }
    if resolved['standardized']['standardize'] is not True:
        raise ValueError('The standardized arm must z-score each lead')
    if resolved['raw_mv']['standardize'] is not False:
        raise ValueError('The raw millivolt arm must keep voltage')
    shared = ('seed', 'width', 'blocks_per_stage')
    if any(resolved['standardized'][key] != resolved['raw_mv'][key] for key in shared):
        raise ValueError('The two arms must share seed and architecture')
    protocol['resolved'] = resolved
    return protocol
