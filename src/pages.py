"""Saved creased-page traces shared by both frozen models."""
import json
from pathlib import Path

import numpy as np

LOCKED_RESULTS = {
    'standardized': Path('results/paired_standardized.json'),
    'raw_mv': Path('results/paired_raw_mv.json'),
}


def page_dir(level, split):
    if split not in ('test', 'validation'):
        raise ValueError('The page split must be test or validation')
    return Path('results/paired_pages') / level / split


def assert_matches_locked(result, locked):
    """Raise when a rerun disagrees with the locked creased-page JSON.

    This function does not write that JSON.
    """
    arm = result['arm']
    for name, value in result['delta_classwise_ece'].items():
        previous = float(locked['delta_classwise_ece'][name])
        if not np.isfinite(value) or abs(float(value) - previous) > 1e-5:
            raise RuntimeError(
                f'{arm} {name} delta ECE {value} disagrees with the locked value {previous}. '
                'The locked JSON was not modified.')


def load_locked(arm):
    path = LOCKED_RESULTS[arm]
    if not path.exists():
        raise FileNotFoundError(f'Missing locked paired result: {path}')
    return json.loads(path.read_text())


def affine_match(traced, clean, epsilon=1e-6):
    """Map each traced lead onto the clean lead mean and standard deviation.

    This is an oracle. It reads the clean waveform and is not a corrector feature.
    """
    traced = np.asarray(traced, dtype=np.float64)
    clean = np.asarray(clean, dtype=np.float64)
    if traced.shape != clean.shape or traced.ndim != 3:
        raise ValueError('Expected matching (records, leads, time) arrays')
    traced_mean = traced.mean(axis=-1, keepdims=True)
    clean_mean = clean.mean(axis=-1, keepdims=True)
    traced_std = np.maximum(traced.std(axis=-1, keepdims=True), epsilon)
    clean_std = np.maximum(clean.std(axis=-1, keepdims=True), epsilon)
    return ((traced - traced_mean) / traced_std * clean_std + clean_mean).astype(np.float32)


def lead_std_mv(traced):
    traced = np.asarray(traced, dtype=np.float64)
    return traced.std(axis=-1).mean(axis=-1)


def probability_logit(probability):
    clipped = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))
