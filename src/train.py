"""Train only on folds 1-8; select checkpoints only on fold 9."""
import argparse
import csv
import hashlib
from importlib.metadata import distributions
import json
import os
from pathlib import Path
import platform
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from .baseline_data import baseline_frame, load_split
from .data import digest
from .metrics import classification_metrics
from .models import ECGResNet


def seed_everything(seed, threads):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def resolve_device(preference, smoke):
    """Research runs use one NVIDIA GPU. Smoke tests and explicit cpu configs stay on CPU."""
    if smoke:
        preference = 'cpu'
    if preference == 'cpu':
        return torch.device('cpu')
    if preference != 'cuda':
        raise RuntimeError(
            f'Unsupported device {preference!r}. This project trains on NVIDIA CUDA '
            '(Google Colab or a 40-series laptop).')
    if not torch.cuda.is_available():
        raise RuntimeError(
            'No NVIDIA GPU is visible. Train on Google Colab or the 40-series laptop, '
            'or pass --smoke for a CPU check.')
    return torch.device('cuda')


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def atomic_checkpoint(path, value):
    temporary = path.with_suffix('.pt.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def provenance(data_config):
    root = Path(data_config['data_dir'])
    stage1 = json.loads((Path(data_config['results_dir']) / 'stage1.json').read_text())
    for name, expected in stage1['metadata_sha256'].items():
        if digest(root / name) != expected:
            raise ValueError(f'Stage 1 metadata changed: {name}')
    return {name: digest(root / name) for name in ['ptbxl_database.csv', 'scp_statements.csv']} | {
        'record_labels.csv': digest(Path(data_config['results_dir']) / 'record_labels.csv'),
    } | {f'src/{name}': digest(Path(__file__).parent / name) for name in
         ['models.py', 'baseline_data.py', 'metrics.py', 'train.py', 'eval.py']}


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    labels, probabilities = [], []
    for x, y in loader:
        logits = model(x.to(device))
        probabilities.append(logits.sigmoid().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(probabilities)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/stage2.yaml')
    parser.add_argument('--data-config', default='configs/pilot.yaml')
    parser.add_argument('--output-dir', default='results/stage2')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='One epoch on 64 train/64 validation records; never a baseline')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    data_config = yaml.safe_load(Path(args.data_config).read_text())
    if not config['exclude_unlabeled']:
        raise ValueError('This pilot explicitly excludes unknown diagnostic targets')
    if args.smoke:
        config['epochs'] = 1
    if config['epochs'] < 1 or config['batch_size'] < 1:
        raise ValueError('epochs and batch_size must be positive')
    seed_everything(config['seed'], config['cpu_threads'])
    out = Path(args.output_dir)
    if args.smoke and out == Path('results/stage2'):
        out = Path('results/smoke_stage2')
    out.mkdir(parents=True, exist_ok=True)
    last_path = out / 'last.pt'
    if args.resume and not last_path.exists():
        raise FileNotFoundError(f'No resumable checkpoint: {last_path}')
    if not args.resume and any((out / name).exists() for name in ['last.pt', 'best.pt', 'run_config.json']):
        raise FileExistsError('Use --resume or a new output directory; an existing run will not be overwritten')
    source_hashes = provenance(data_config)
    run_config = {'training': config, 'data': data_config, 'smoke': args.smoke,
                  'source_hashes': source_hashes}
    signature = hashlib.sha256(json.dumps(run_config, sort_keys=True).encode()).hexdigest()
    state = torch.load(last_path, map_location='cpu', weights_only=False) if args.resume else None
    if state and state['signature'] != signature:
        raise ValueError('Resume config/data differ from checkpoint; use a new output directory')
    frame = baseline_frame(data_config)
    train, train_ids = load_split(frame, 'train', data_config, config, 64 if args.smoke else None)
    validation, validation_ids = load_split(frame, 'validation', data_config, config, 64 if args.smoke else None)
    device = resolve_device(config.get('device', 'cuda'), args.smoke)
    generator = torch.Generator().manual_seed(config['seed'])
    train_loader = DataLoader(train, batch_size=config['batch_size'], shuffle=True,
                              generator=generator, num_workers=0, pin_memory=device.type == 'cuda')
    val_loader = DataLoader(validation, batch_size=config['batch_size'], shuffle=False)
    model_args = {'classes': len(data_config['classes']), 'width': config['width'],
                  'blocks_per_stage': config['blocks_per_stage']}
    model = ECGResNet(**model_args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    amp = config['amp'] and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    criterion = torch.nn.BCEWithLogitsLoss()
    history, best, start = [], -1.0, 0
    if state:
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        random.setstate(state['rng_python'])
        np.random.set_state(state['rng_numpy'])
        torch.set_rng_state(state['rng_torch'])
        generator.set_state(state['rng_loader'])
        if device.type == 'cuda' and state['rng_cuda']:
            torch.cuda.set_rng_state_all(state['rng_cuda'])
        history, best, start = state['history'], state['best_auc'], state['epoch'] + 1
    else:
        atomic_json(out / 'run_config.json', run_config)
        atomic_json(out / 'record_ids.json', {'train': train_ids.tolist(), 'validation': validation_ids.tolist()})
    environment = {
        'python': platform.python_version(), 'torch': torch.__version__, 'device': str(device),
        'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
        'amp': amp, 'parameters': sum(p.numel() for p in model.parameters()),
        'packages': dict(sorted((d.metadata['Name'], d.version) for d in distributions())),
    }
    atomic_json(out / f'environment_epoch_{start:03d}.json', environment)
    print(f"Device: {device}; AMP: {amp}; parameters: {environment['parameters']}; smoke: {args.smoke}", flush=True)
    for epoch in range(start, config['epochs']):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                loss = criterion(model(x), y)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss; stopping')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(x)
        y, p = predict(model, val_loader, device)
        metrics = classification_metrics(y, p, data_config['classes'], config['ece_bins'])
        score = metrics['macro_auroc']
        if score is None:
            raise ValueError('Validation split lacks one or more classes; AUROC selection is undefined')
        improved = score > best
        best = max(best, score)
        history.append({'epoch': epoch + 1, 'train_bce': total_loss / len(train),
                        'validation_macro_auroc': score, 'validation_ece': metrics['ece'],
                        'validation_brier': metrics['brier'], 'learning_rate': optimizer.param_groups[0]['lr']})
        scheduler.step()
        state = {
            'epoch': epoch, 'model': model.state_dict(), 'model_args': model_args,
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
            'signature': signature, 'run_config': run_config, 'best_auc': best,
            'validation_metrics': metrics, 'history': history,
            'rng_python': random.getstate(), 'rng_numpy': np.random.get_state(),
            'rng_torch': torch.get_rng_state(), 'rng_loader': generator.get_state(),
            'rng_cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
        }
        if improved:
            atomic_checkpoint(out / 'best.pt', state)
            np.savez_compressed(out / 'validation_predictions.npz', ecg_ids=validation_ids, labels=y, probabilities=p)
        atomic_checkpoint(last_path, state)
        with (out / 'history.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        print(json.dumps(history[-1]), flush=True)
    status = 'smoke_only' if args.smoke else ('needs_review' if best < config['validation_auc_review_threshold'] else 'ready_for_clean_test')
    summary = {'status': status, 'epochs_completed': len(history), 'best_validation_macro_auroc': best,
               'selection_split': 'fold 9 only', 'test_evaluated': False,
               'review_threshold': config['validation_auc_review_threshold'], 'signature': signature}
    atomic_json(out / 'training_summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
