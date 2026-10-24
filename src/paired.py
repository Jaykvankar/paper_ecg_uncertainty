"""Score a frozen classifier on clean and reconstructed waveforms.

Fold 10 is scored once per arm, and only after the training-record trace check passes.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, TensorDataset

from .baseline_data import baseline_frame, prepare_signal
from .data import load_signal
from .digitize import recover_leads
from .images import render_record
from .intervals import bootstrap_classwise_delta_ece, classwise_delta
from .models import ECGResNet
from .protocol import load_protocol
from .snr import leadwise_snr
from .train import atomic_json, predict, provenance, resolve_device, seed_everything
from .uncertainty import mean_binary_entropy


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def load_ready_model(run_dir):
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / 'training_summary.json').read_text())
    if summary['status'] != 'ready_for_clean_test':
        raise ValueError('The checkpoint is smoke-only, incomplete, or not past the validation gate')
    state = torch.load(run_dir / 'best.pt', map_location='cpu', weights_only=False)
    if state['signature'] != summary['signature'] or state['run_config']['smoke']:
        raise ValueError('Checkpoint does not match the completed training run')
    config = state['run_config']['training']
    data_config = state['run_config']['data']
    if provenance(data_config) != state['run_config']['source_hashes']:
        raise ValueError('Source or data hashes differ from the checkpoint. Train again in a new directory.')
    return state, config, data_config


def require_digitizer_check(protocol):
    path = Path(protocol['digitizer_check'])
    if not path.exists():
        raise RuntimeError('Run python -m src.digitize --check before any paired fold-10 score.')
    check = json.loads(path.read_text())
    if check.get('status') != 'passed' or check.get('split') != 'train':
        raise RuntimeError('The training-record trace check has not passed. Fold 10 stays untouched.')
    if check.get('minimum_correlation', 0) < protocol['correlation_threshold']:
        raise RuntimeError('Stored trace correlations are below the locked threshold.')
    return check


def score_arm(protocol, arm, run_dir=None, limit=None, level=None):
    check = require_digitizer_check(protocol)
    level = level or protocol['primary_level']
    if level not in protocol['levels']:
        raise ValueError(f'{level} is not a locked page level')
    spec = protocol['resolved'][arm]
    run_dir = Path(run_dir or spec['results'])
    state, config, data_config = load_ready_model(run_dir)
    if bool(config.get('standardize', True)) != bool(spec['standardize']):
        raise ValueError(f'{arm} checkpoint does not match the locked standardize setting')
    seed_everything(config['seed'], config['cpu_threads'])
    frame = baseline_frame(data_config)
    part = frame.loc[frame.split == 'test']
    if limit:
        part = part.sample(n=min(limit, len(part)), random_state=config['seed']).sort_index()
    device = resolve_device('cpu' if limit else config.get('device', 'cuda'), smoke=False)
    model = ECGResNet(**state['model_args']).to(device)
    model.load_state_dict(state['model'])
    root = Path(data_config['data_dir'])
    clean, recovered, snr_rows = [], [], []
    image_root = run_dir / 'paired_images' / level
    for filename in part.filename_lr:
        signal = load_signal(root, filename)
        png, layout_path = render_record(root / filename, image_root / Path(filename).name, level=level, seed=config['seed'])
        layout = json.loads(layout_path.read_text())
        page = recover_leads(np.asarray(Image.open(png)), layout)
        clean.append(prepare_signal(signal, spec['standardize'], config['normalization_epsilon']))
        recovered.append(prepare_signal(page, spec['standardize'], config['normalization_epsilon']))
        snr_rows.append(leadwise_snr(signal, page, data_config['sampling_rate']))
    clean = np.stack(clean)
    recovered = np.stack(recovered)
    labels = part[data_config['classes']].to_numpy(dtype=np.float32)
    loader_clean = DataLoader(TensorDataset(torch.from_numpy(clean), torch.from_numpy(labels)),
                              batch_size=config['batch_size'])
    loader_paper = DataLoader(TensorDataset(torch.from_numpy(recovered), torch.from_numpy(labels)),
                              batch_size=config['batch_size'])
    y_clean, p_clean = predict(model, loader_clean, device)
    y_paper, p_paper = predict(model, loader_paper, device)
    if not np.array_equal(y_clean, y_paper):
        raise ValueError('Clean and paper labels diverged')
    delta = classwise_delta(y_clean, p_clean, p_paper, data_config['classes'], config['ece_bins'])
    entropy_clean = mean_binary_entropy(p_clean)
    entropy_paper = mean_binary_entropy(p_paper)
    patients = frame.loc[part.index, 'patient_id'].to_numpy()
    interval = bootstrap_classwise_delta_ece(
        y_clean, p_clean, p_paper, data_config['classes'], patients, config['ece_bins'],
        protocol['bootstrap']['resamples'], protocol['bootstrap']['seed'],
        tuple(protocol['bootstrap']['percentiles']))
    finite_snr = [row['mean_snr_db'] for row in snr_rows if np.isfinite(row['mean_snr_db'])]
    return json_ready({
        'status': 'engineering_subset_not_the_fold10_result' if limit else 'paired_scored',
        'arm': arm, 'level': level, 'primary_level': protocol['primary_level'],
        'primary_outcome': protocol['primary_outcome'],
        'records': int(len(part)), 'limit': limit,
        'delta_classwise_ece': delta['delta_classwise_ece'],
        'delta_classwise_auroc': delta['delta_classwise_auroc'],
        'delta_mean_binary_entropy': {
            name: float(entropy_paper[i] - entropy_clean[i]) for i, name in enumerate(data_config['classes'])
        },
        'patient_bootstrap': interval,
        'mean_snr_db': float(np.mean(finite_snr)) if finite_snr else None,
        'digitizer_check': check['record'],
        'claim_scope': protocol['claim_scope'],
        'run_dir': str(run_dir),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', default='configs/paired.yaml')
    parser.add_argument('--arm', choices=['standardized', 'raw_mv'], required=True)
    parser.add_argument('--run-dir', default=None)
    parser.add_argument('--level', default=None)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    result = score_arm(protocol, args.arm, args.run_dir, args.limit, args.level)
    output = Path(args.output or f"results/paired_{args.arm}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
