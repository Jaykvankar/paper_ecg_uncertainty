"""Evaluate a completed Stage 2 checkpoint on the untouched labeled fold 10."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .baseline_data import baseline_frame, load_split
from .data import digest
from .data import build_labels
from .metrics import classification_metrics
from .models import ECGResNet
from .train import atomic_json, predict, provenance, seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', default='results/stage2')
    args = parser.parse_args()
    out = Path(args.run_dir)
    summary = json.loads((out / 'training_summary.json').read_text())
    if summary['status'] != 'ready_for_clean_test':
        raise ValueError('Training is smoke-only, incomplete, or needs validation debugging; do not evaluate test yet')
    checkpoint = out / 'best.pt'
    # Only load checkpoints created by this project, not untrusted downloaded files.
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config, data_config = state['run_config']['training'], state['run_config']['data']
    if state['signature'] != summary['signature'] or state['run_config']['smoke']:
        raise ValueError('Checkpoint does not match the completed training run')
    if provenance(data_config) != state['run_config']['source_hashes']:
        raise ValueError('Data provenance differs from training')
    seed_everything(config['seed'], config['cpu_threads'])
    frame = baseline_frame(data_config)
    all_records = build_labels(Path(data_config['data_dir']), data_config)
    excluded = {split: int(((all_records.split == split) & ~all_records.has_diagnostic_label).sum())
                for split in data_config['splits']}
    test, ids = load_split(frame, 'test', data_config, config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ECGResNet(**state['model_args']).to(device)
    model.load_state_dict(state['model'])
    y, p = predict(model, DataLoader(test, batch_size=config['batch_size']), device)
    metrics = classification_metrics(y, p, data_config['classes'], config['ece_bins'])
    np.savez_compressed(out / 'clean_test_predictions.npz', ecg_ids=ids, labels=y, probabilities=p)
    result = {
        **metrics, 'split': 'fold 10, diagnostic-labeled records only',
        'classes': data_config['classes'], 'excluded_unlabeled_records': excluded,
        'checkpoint_epoch': state['epoch'] + 1, 'checkpoint_sha256': digest(checkpoint),
        'best_validation_macro_auroc': state['best_auc'], 'calibration': 'none; raw sigmoid probabilities',
        'run_config': state['run_config'],
        'status': 'needs_review' if metrics['macro_auroc'] is None or metrics['macro_auroc'] < config['validation_auc_review_threshold'] else 'baseline_measured',
    }
    atomic_json(out / 'clean_baseline.json', result)
    report = ('# Stage 2 clean-signal baseline\n\n'
              f"Status: {result['status']}. Checkpoint selected on fold 9 only.\n\n"
              f"Fold 10 records: {len(ids)}. Macro AUROC: {metrics['macro_auroc']}. "
              f"ECE: {metrics['ece']}. Brier: {metrics['brier']}.\n\n"
              'Per-class AUROC:\n\n' + '\n'.join(f'- {k}: {v}' for k, v in metrics['per_class_auroc'].items())
              + '\n\n## Honest limitations\n\n'
              '- One seed and compact architecture; no digitized-image evaluation yet.\n'
              f'- Unlabeled diagnostic records excluded: {excluded}.\n'
              '- Per-record per-lead standardization removes absolute voltage scale.\n'
              '- Labels follow the official SCP mapping, including zero-likelihood statements.\n'
              '- Calibration metrics are for uncalibrated probabilities; no confidence intervals yet.\n'
              '- This clean baseline alone does not test the digitization-degradation hypothesis.\n'
              '- A low test score requires documented investigation, not tuning against fold 10.\n')
    (out / 'REPORT.md').write_text(report)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
