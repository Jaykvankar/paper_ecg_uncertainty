"""Score the pre-specified width-64 models on the saved creased fold-10 traces.

The locked width-32 JSON is not read for writing and is not replaced.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .intervals import bootstrap_arm_contrast
from .models import ECGResNet
from .pages import page_dir
from .paired import arm_summary, json_ready, load_ready_model, prepare_batch
from .protocol import load_protocol
from .train import atomic_json, predict, resolve_device

RUNS = {
    'standardized': Path('results/stage2_wide'),
    'raw_mv': Path('results/stage2_wide_raw'),
}
OUTPUT = Path('results/sensitivity_wide/summary.json')
LOCKED_CONTRAST = Path('results/paired_pages/creased/test/arm_contrast.json')


def agrees_with_locked(contrast, locked_point, target):
    """The interval excludes zero and shares the locked contrast's sign."""
    low = float(contrast['low'][target])
    high = float(contrast['high'][target])
    point = float(contrast['point'][target])
    excludes_zero = low > 0 or high < 0
    same_sign = point * float(locked_point) > 0
    return bool(excludes_zero and same_sign)


def score(level='creased', split='test'):
    for path in RUNS.values():
        summary = json.loads((path / 'training_summary.json').read_text())
        if summary['status'] != 'ready_for_clean_test':
            raise ValueError(f'{path} did not pass the validation gate')
        rule = summary.get('agreement_rule')
        stored = json.loads((path / 'run_config.json').read_text())['training']
        if stored.get('agreement_rule') != 'interval_excludes_zero_and_same_sign_as_locked_contrast':
            raise ValueError('The agreement rule must be the one fixed before training')
        if stored.get('width') != 64:
            raise ValueError('This sensitivity is the width-64 network only')
        if rule is not None and rule != stored['agreement_rule']:
            raise ValueError('Training summary and config disagree on the agreement rule')
    protocol = load_protocol()
    destination = page_dir(level, split)
    saved = np.load(destination / 'traces.npz', allow_pickle=False)
    classes = [str(name) for name in saved['classes']]
    labels = saved['labels']
    patients = saved['patient_id']
    probabilities = {}
    arms = {}
    for arm, run_dir in RUNS.items():
        state, config, _ = load_ready_model(run_dir)
        if bool(config.get('standardize', True)) != (arm == 'standardized'):
            raise ValueError(f'{arm} checkpoint does not match its standardize setting')
        device = resolve_device(config.get('device', 'cuda'), smoke=False)
        model = ECGResNet(**state['model_args']).to(device)
        model.load_state_dict(state['model'])
        epsilon = config['normalization_epsilon']
        clean = prepare_batch(saved['clean_mv'], config['standardize'], epsilon)
        paper = prepare_batch(saved['traced_mv'], config['standardize'], epsilon)
        y_clean, p_clean = predict(
            model, DataLoader(TensorDataset(torch.from_numpy(clean), torch.from_numpy(labels)),
                              batch_size=config['batch_size']), device)
        _, p_paper = predict(
            model, DataLoader(TensorDataset(torch.from_numpy(paper), torch.from_numpy(labels)),
                              batch_size=config['batch_size']), device)
        arms[arm] = arm_summary(
            protocol, arm, level, split, y_clean, p_clean, p_paper, patients, classes, config,
            saved['snr_db'], str(run_dir), str(saved['digitizer_check']))
        arms[arm]['role'] = 'sensitivity'
        arms[arm]['width'] = 64
        probabilities[f'{arm}_clean'] = p_clean
        probabilities[f'{arm}_paper'] = p_paper
    contrast = bootstrap_arm_contrast(
        labels, probabilities['raw_mv_clean'], probabilities['raw_mv_paper'],
        probabilities['standardized_clean'], probabilities['standardized_paper'],
        classes, patients, 15, protocol['bootstrap']['resamples'], protocol['bootstrap']['seed'],
        tuple(protocol['bootstrap']['percentiles']))
    locked = json.loads(LOCKED_CONTRAST.read_text())
    target = 'CD'
    result = json_ready({
        'role': 'sensitivity',
        'width': 64,
        'primary_unchanged': True,
        'agreement_rule': 'interval_excludes_zero_and_same_sign_as_locked_contrast',
        'agreement_class': target,
        'agrees': agrees_with_locked(contrast, locked['point'][target], target),
        'locked_contrast_point': locked['point'],
        'contrast': contrast,
        'standardized': arms['standardized'],
        'raw_mv': arms['raw_mv'],
    })
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(OUTPUT, result)
    return result


def main():
    print(json.dumps(score(), indent=2), flush=True)


if __name__ == '__main__':
    main()
