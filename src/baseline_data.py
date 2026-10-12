"""Stage 2 input handling; does not download or rewrite Stage 1 outputs."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from .data import build_labels, load_signal


def baseline_frame(data_config):
    frame = build_labels(Path(data_config['data_dir']), data_config)
    stored = pd.read_csv(Path(data_config['results_dir']) / 'record_labels.csv', index_col='ecg_id')
    pd.testing.assert_frame_equal(frame[stored.columns], stored)
    # Missing diagnostic annotations are not known all-negative targets.
    return frame.loc[frame.has_diagnostic_label].copy()


def normalize(signal, epsilon=1e-6):
    signal = np.asarray(signal, dtype=np.float32)
    if not np.isfinite(signal).all():
        raise ValueError('Nonfinite clean signal')
    return (signal - signal.mean(axis=-1, keepdims=True)) / np.maximum(
        signal.std(axis=-1, keepdims=True), epsilon)


def load_split(frame, split, data_config, training_config, limit=None):
    part = frame.loc[frame.split == split]
    if limit:
        part = part.sample(n=min(limit, len(part)), random_state=training_config['seed']).sort_index()
    array = np.empty((len(part), 12, 1000), dtype=np.float32)
    root = Path(data_config['data_dir'])

    def read(filename):
        return normalize(load_signal(root, filename), training_config['normalization_epsilon'])

    with ThreadPoolExecutor(max_workers=training_config['load_workers']) as pool:
        for n, value in enumerate(pool.map(read, part.filename_lr)):
            array[n] = value
    labels = part[data_config['classes']].to_numpy(dtype=np.float32)
    print(f'Loaded {split}: {len(part)} labeled records', flush=True)
    return TensorDataset(torch.from_numpy(array), torch.from_numpy(labels)), part.index.to_numpy()
