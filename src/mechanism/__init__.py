"""Mechanisms of tracing-induced calibration error."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.intervals import bootstrap_classwise_delta_ece, classwise_delta
from src.models import ECGResNet
from src.pages import affine_match, page_dir
from src.paired import arm_summary, load_ready_model, prepare_batch
from src.protocol import load_protocol
from src.train import atomic_json, predict, resolve_device


def reliability_bins(labels, probability, classes, bins=15):
    labels = np.asarray(labels)
    probability = np.asarray(probability)
    diagram = {}
    for index, name in enumerate(classes):
        outcome = labels[:, index]
        score = probability[:, index]
        assignments = np.minimum((score * bins).astype(int), bins - 1)
        rows = []
        for bin_index in range(bins):
            mask = assignments == bin_index
            rows.append({
                'bin': bin_index,
                'count': int(mask.sum()),
                'confidence': None if not mask.any() else float(score[mask].mean()),
                'accuracy': None if not mask.any() else float(outcome[mask].mean()),
            })
        diagram[name] = rows
    return diagram


def snr_by_superclass(labels, snr, classes):
    """Mean lead SNR for records with and without each superclass."""
    labels = np.asarray(labels)
    snr = np.asarray(snr, dtype=np.float64)
    finite = np.isfinite(snr)
    report = {}
    for index, name in enumerate(classes):
        present = finite & (labels[:, index] >= 0.5)
        absent = finite & (labels[:, index] < 0.5)
        report[name] = {
            'with_label_mean_snr_db': None if not present.any() else float(snr[present].mean()),
            'with_label_records': int(present.sum()),
            'without_label_mean_snr_db': None if not absent.any() else float(snr[absent].mean()),
            'without_label_records': int(absent.sum()),
        }
    return report


def score_oracle(protocol=None, split='test', level=None):
    protocol = protocol or load_protocol()
    if protocol['scale_control']['kind'] != 'oracle_affine_match':
        raise ValueError('Scale control must stay an oracle')
    level = level or protocol['primary_level']
    store = page_dir(level, split) / 'traces.npz'
    if not store.exists():
        raise FileNotFoundError(f'Render pages before the oracle scale check: {store}')
    saved = np.load(store, allow_pickle=False)
    matched = affine_match(saved['traced_mv'], saved['clean_mv'])
    classes = [str(name) for name in saved['classes']]
    labels = saved['labels']
    patients = saved['patient_id']
    output = Path('results/mechanism')
    output.mkdir(parents=True, exist_ok=True)
    by_class = snr_by_superclass(labels, saved['snr_db'], classes)
    atomic_json(output / 'snr_by_class.json', {'split': split, 'level': level, 'by_class': by_class})
    written = {'snr_by_class': str(output / 'snr_by_class.json')}
    for arm in protocol['arms']:
        spec = protocol['resolved'][arm]
        state, config, _ = load_ready_model(spec['results'])
        device = resolve_device(config.get('device', 'cuda'), smoke=False)
        model = ECGResNet(**state['model_args']).to(device)
        model.load_state_dict(state['model'])
        epsilon = config['normalization_epsilon']
        clean = prepare_batch(saved['clean_mv'], spec['standardize'], epsilon)
        paper = prepare_batch(matched, spec['standardize'], epsilon)
        y_clean, p_clean = predict(
            model, DataLoader(TensorDataset(torch.from_numpy(clean), torch.from_numpy(labels)),
                              batch_size=config['batch_size']), device)
        _, p_paper = predict(
            model, DataLoader(TensorDataset(torch.from_numpy(paper), torch.from_numpy(labels)),
                              batch_size=config['batch_size']), device)
        summary = arm_summary(
            protocol, arm, level, split, y_clean, p_clean, p_paper, patients, classes, config,
            saved['snr_db'], str(spec['results']), str(saved['digitizer_check']))
        summary['kind'] = 'oracle_affine_match'
        summary['uses_clean_waveform'] = True
        summary['snr_by_class'] = by_class
        summary['reliability'] = {
            'clean': reliability_bins(y_clean, p_clean, classes, config['ece_bins']),
            'oracle': reliability_bins(y_clean, p_paper, classes, config['ece_bins']),
        }
        destination = output / f'{arm}.json'
        atomic_json(destination, summary)
        written[arm] = str(destination)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', default='configs/paired.yaml')
    parser.add_argument('--split', choices=['test', 'validation'], default='test')
    args = parser.parse_args()
    written = score_oracle(load_protocol(args.protocol), args.split)
    print(json.dumps(written, indent=2), flush=True)


if __name__ == '__main__':
    main()

