"""Locked paired-experiment settings. Diagnosis is never an input."""
from pathlib import Path

import yaml

PRIMARY_OUTCOME = 'delta_classwise_ece'
UNCERTAINTY_SCORE = 'delta_mean_binary_entropy'
PRIMARY_LEVEL = 'creased'
ARMS = ('standardized', 'raw_mv')
CORRECTOR_FEATURES = ('logit', 'traced_lead_std_mv', 'ink_gap_fraction')


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
    _check_scale_control(protocol)
    _check_corrector(protocol)
    _check_real_scan(protocol)
    protocol['resolved'] = resolved
    return protocol


def _check_scale_control(protocol):
    block = protocol.get('scale_control')
    if not isinstance(block, dict):
        raise ValueError('The oracle scale check is required')
    if block.get('kind') != 'oracle_affine_match' or block.get('uses_clean_waveform') is not True:
        raise ValueError('Scale control must stay an oracle that uses the clean waveform')


def _check_corrector(protocol):
    block = protocol.get('corrector')
    if not isinstance(block, dict):
        raise ValueError('The trace-only corrector block is required')
    if block.get('arm') != 'raw_mv':
        raise ValueError('The corrector is fit on the raw millivolt arm')
    if block.get('fit_split') != 'validation':
        raise ValueError('The corrector fit split must be validation')
    if block.get('apply_split') != 'test':
        raise ValueError('The corrector is applied once on the test split')
    features = tuple(block.get('features') or [])
    for name in features:
        if 'clean' in str(name).lower():
            raise ValueError('A corrector feature must not use the clean waveform')
    if features != CORRECTOR_FEATURES:
        raise ValueError('Corrector features must be the trace-only set')
    if block.get('target_class') != 'CD':
        raise ValueError('The corrector success class is CD')
    if block.get('target_role') != 'secondary':
        raise ValueError('The CD corrector target stays a secondary check')
    limit = block.get('auroc_drop_limit')
    if not isinstance(limit, (int, float)) or limit < 0:
        raise ValueError('The AUROC drop limit must be a non-negative number')


def _check_real_scan(protocol):
    block = protocol.get('real_scan')
    if not isinstance(block, dict):
        raise ValueError('The real color-scan block is required')
    if block.get('source') != 'D0':
        raise ValueError('The external image test uses the PTB-XL color scans only')
    if block.get('score_split') != 'test':
        raise ValueError('Real scans are scored on the test split')
    if float(block.get('correlation_threshold', 0)) < 0.8:
        raise ValueError('The real-scan gate stays at correlation 0.80')
    excluded = list(block.get('excluded') or [])
    for name in ('emory', 'akershus', 'phone', 'mold'):
        if name not in excluded:
            raise ValueError(f'The real-scan claim must exclude {name}')
