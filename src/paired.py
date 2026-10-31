"""Score a frozen classifier on clean and reconstructed waveforms.

Fold 10 is scored once per arm, and only after the training-record trace check passes.
A split rerun renders each creased page once and does not replace the locked JSON.
"""
import argparse
import hashlib
import importlib.util
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
from .intervals import bootstrap_arm_contrast, bootstrap_classwise_delta_ece, classwise_delta
from .models import ECGResNet
from .pages import assert_matches_locked, load_locked, page_dir
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


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _visible_layout(image, layout):
    """Keep lead boxes on the page. A crease can shrink the image after the box is stored."""
    height, width = np.asarray(image).shape[:2]
    visible = json.loads(json.dumps(layout))
    for entry in visible.get('leads', []):
        box = entry.get('lead_bounding_box')
        if not isinstance(box, dict):
            continue
        for key, point in box.items():
            y = min(max(float(point[0]), 0.0), height - 1)
            x = min(max(float(point[1]), 0.0), width - 1)
            box[key] = [y, x]
    return visible


def _existing_page(record_dir):
    if not record_dir.exists():
        return None
    pages = sorted(record_dir.glob('*-0.png'))
    if not pages or not pages[0].with_suffix('.json').exists():
        return None
    return pages[0], pages[0].with_suffix('.json')


def _write_traces(path, completed, part, classes, tracer, check):
    order = [name for name in part.filename_lr.tolist() if name in completed]
    subset = part.set_index('filename_lr', drop=False).loc[order]
    temporary = path.with_name(f'{path.stem}.writing.npz')
    np.savez_compressed(
        temporary,
        filename=np.asarray(order),
        patient_id=subset.patient_id.to_numpy(),
        labels=subset[classes].to_numpy(dtype=np.float32),
        clean_mv=np.stack([completed[name]['clean'] for name in order]),
        traced_mv=np.stack([completed[name]['traced'] for name in order]),
        ink_gap=np.asarray([completed[name]['gap'] for name in order], dtype=np.float32),
        snr_db=np.asarray([completed[name]['snr'] for name in order], dtype=np.float64),
        classes=np.asarray(classes),
        tracer=np.asarray(tracer),
        tracer_sha256=np.asarray(_file_sha256(tracer)),
        digitizer_check=np.asarray(check['record']),
    )
    temporary.replace(path)


def locked_recover():
    """Use the tracer that produced the locked JSON when the working copy differs."""
    current = Path('src/digitize.py')
    backup = Path('src/digitize_locked.py')
    if not backup.exists() or _file_sha256(backup) == _file_sha256(current):
        return recover_leads, current.as_posix()
    source = backup.read_text(encoding='utf-8')
    source = source.replace('from .', 'from src.')
    spec = importlib.util.spec_from_loader('src.digitize_locked', loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, str(backup), 'exec'), module.__dict__)
    return module.recover_leads, backup.as_posix()


def prepare_batch(signals, standardize, epsilon):
    signals = np.asarray(signals, dtype=np.float32)
    if not np.isfinite(signals).all():
        raise ValueError('Nonfinite signal')
    if not standardize:
        return signals.copy()
    mean = signals.mean(axis=-1, keepdims=True)
    deviation = np.maximum(signals.std(axis=-1, keepdims=True), epsilon)
    return ((signals - mean) / deviation).astype(np.float32)


def _arm_data(protocol):
    """Training and data settings stored with the frozen standardized checkpoint."""
    _state, config, data_config = load_ready_model(protocol['resolved']['standardized']['results'])
    return {'training': config, 'data': data_config}


def _split_frame(protocol, split):
    if split not in ('test', 'validation'):
        raise ValueError('The page split must be test or validation')
    arm_config = _arm_data(protocol)
    frame = baseline_frame(arm_config['data'])
    part = frame.loc[frame.split == split].sort_index()
    if len(part) == 0:
        raise ValueError(f'No labeled records in {split}')
    return part, arm_config


def render_pages(protocol, split, level=None):
    """Render each record once and store millivolt traces before either arm's preprocessing."""
    check = require_digitizer_check(protocol)
    level = level or protocol['primary_level']
    if level not in protocol['levels']:
        raise ValueError(f'{level} is not a locked page level')
    part, arm_config = _split_frame(protocol, split)
    training = arm_config['training']
    data_config = arm_config['data']
    destination = page_dir(level, split)
    destination.mkdir(parents=True, exist_ok=True)
    store = destination / 'traces.npz'
    partial = destination / 'traces.partial.npz'
    recover, tracer = locked_recover()
    classes = data_config['classes']
    if store.exists():
        saved = np.load(store, allow_pickle=False)
        if len(saved['filename']) == len(part) and saved['tracer'].item() == tracer:
            return destination
    seed_everything(training['seed'], training['cpu_threads'])
    root = Path(data_config['data_dir'])
    image_root = destination / 'images'
    completed = {}
    if partial.exists():
        saved = np.load(partial, allow_pickle=False)
        if saved['tracer'].item() == tracer:
            for index, name in enumerate(saved['filename']):
                completed[str(name)] = {
                    'clean': np.array(saved['clean_mv'][index]),
                    'traced': np.array(saved['traced_mv'][index]),
                    'gap': float(saved['ink_gap'][index]),
                    'snr': float(saved['snr_db'][index]),
                }
        saved.close()
    total = len(part)
    for count, filename in enumerate(part.filename_lr, 1):
        if filename in completed:
            if count % 50 == 0 or count == total:
                print(f'Rendered {level} {split} pages: {count}/{total}', flush=True)
            continue
        signal = load_signal(root, filename)
        record_dir = image_root / Path(filename).name
        existing = _existing_page(record_dir)
        if existing is None:
            png, layout_path = render_record(
                root / filename, record_dir, level=level, seed=training['seed'])
        else:
            png, layout_path = existing
        layout = _visible_layout(np.asarray(Image.open(png)), json.loads(layout_path.read_text()))
        image = np.asarray(Image.open(png))
        if recover is recover_leads:
            page, gap = recover_leads(image, layout, return_gap=True)
        else:
            page = recover(image, layout)
            _, gap = recover_leads(image, layout, return_gap=True)
        completed[filename] = {
            'clean': signal.astype(np.float32),
            'traced': np.asarray(page, dtype=np.float32),
            'gap': float(gap),
            'snr': float(leadwise_snr(signal, page, data_config['sampling_rate'])['mean_snr_db']),
        }
        if count % 25 == 0:
            _write_traces(partial, completed, part, classes, tracer, check)
        if count % 50 == 0 or count == total:
            print(f'Rendered {level} {split} pages: {count}/{total}', flush=True)
    _write_traces(store, completed, part, classes, tracer, check)
    if partial.exists():
        partial.unlink()
    return destination


def score_saved_pages(protocol, split, level=None):
    """Score both frozen models on one saved render. Fold 10 must match the locked JSON."""
    level = level or protocol['primary_level']
    destination = render_pages(protocol, split, level)
    saved = np.load(destination / 'traces.npz', allow_pickle=False)
    classes = [str(name) for name in saved['classes']]
    labels = saved['labels']
    patients = saved['patient_id']
    summaries = {}
    probabilities = {}
    for arm in protocol['arms']:
        spec = protocol['resolved'][arm]
        state, config, _ = load_ready_model(spec['results'])
        if bool(config.get('standardize', True)) != bool(spec['standardize']):
            raise ValueError(f'{arm} checkpoint does not match the locked standardize setting')
        device = resolve_device(config.get('device', 'cuda'), smoke=False)
        model = ECGResNet(**state['model_args']).to(device)
        model.load_state_dict(state['model'])
        epsilon = config['normalization_epsilon']
        clean = prepare_batch(saved['clean_mv'], spec['standardize'], epsilon)
        paper = prepare_batch(saved['traced_mv'], spec['standardize'], epsilon)
        loader_clean = DataLoader(
            TensorDataset(torch.from_numpy(clean), torch.from_numpy(labels)), batch_size=config['batch_size'])
        loader_paper = DataLoader(
            TensorDataset(torch.from_numpy(paper), torch.from_numpy(labels)), batch_size=config['batch_size'])
        y_clean, p_clean = predict(model, loader_clean, device)
        y_paper, p_paper = predict(model, loader_paper, device)
        if not np.array_equal(y_clean, y_paper):
            raise ValueError('Clean and paper labels diverged')
        summary = arm_summary(
            protocol, arm, level, split, y_clean, p_clean, p_paper, patients, classes, config,
            saved['snr_db'], str(spec['results']), str(saved['digitizer_check']))
        if split == 'test' and level == protocol['primary_level']:
            assert_matches_locked(summary, load_locked(arm))
            summary['matches_locked'] = True
        atomic_json(destination / f'{arm}.json', summary)
        probabilities[f'{arm}_clean'] = p_clean
        probabilities[f'{arm}_paper'] = p_paper
        summaries[arm] = summary
    np.savez_compressed(destination / 'probabilities.npz', labels=labels, patient_id=patients, **probabilities)
    if 'raw_mv_paper' in probabilities and 'standardized_paper' in probabilities:
        contrast = bootstrap_arm_contrast(
            labels, probabilities['raw_mv_clean'], probabilities['raw_mv_paper'],
            probabilities['standardized_clean'], probabilities['standardized_paper'],
            classes, patients, config['ece_bins'],
            protocol['bootstrap']['resamples'], protocol['bootstrap']['seed'],
            tuple(protocol['bootstrap']['percentiles']))
        contrast = json_ready(contrast)
        atomic_json(destination / 'arm_contrast.json', contrast)
        summaries['arm_contrast'] = contrast
    return summaries


def arm_summary(protocol, arm, level, split, y_clean, p_clean, p_paper, patients, classes, config,
                snr_db, run_dir, digitizer_check):
    delta = classwise_delta(y_clean, p_clean, p_paper, classes, config['ece_bins'])
    entropy_clean = mean_binary_entropy(p_clean)
    entropy_paper = mean_binary_entropy(p_paper)
    interval = bootstrap_classwise_delta_ece(
        y_clean, p_clean, p_paper, classes, patients, config['ece_bins'],
        protocol['bootstrap']['resamples'], protocol['bootstrap']['seed'],
        tuple(protocol['bootstrap']['percentiles']))
    finite_snr = snr_db[np.isfinite(snr_db)]
    return json_ready({
        'status': 'paired_scored',
        'arm': arm, 'level': level, 'split': split,
        'primary_level': protocol['primary_level'],
        'primary_outcome': protocol['primary_outcome'],
        'records': int(len(y_clean)), 'limit': None,
        'delta_classwise_ece': delta['delta_classwise_ece'],
        'delta_classwise_auroc': delta['delta_classwise_auroc'],
        'delta_mean_binary_entropy': {
            name: float(entropy_paper[i] - entropy_clean[i]) for i, name in enumerate(classes)
        },
        'paper_macro_auroc': delta['paper']['macro_auroc'],
        'patient_bootstrap': interval,
        'mean_snr_db': float(np.mean(finite_snr)) if len(finite_snr) else None,
        'digitizer_check': digitizer_check,
        'claim_scope': protocol['claim_scope'],
        'run_dir': run_dir,
    })


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
    parser.add_argument('--arm', choices=['standardized', 'raw_mv'], default=None)
    parser.add_argument('--split', choices=['test', 'validation'], default=None)
    parser.add_argument('--run-dir', default=None)
    parser.add_argument('--level', default=None)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    if args.split:
        if args.arm or args.limit or args.output:
            raise SystemExit('--split scores both arms from one render. Do not pass --arm, --limit, or --output.')
        summaries = score_saved_pages(protocol, args.split, args.level)
        print(json.dumps(summaries, indent=2), flush=True)
        return
    if args.arm is None:
        raise SystemExit('Pass --arm for one arm, or --split to render once and score both.')
    result = score_arm(protocol, args.arm, args.run_dir, args.limit, args.level)
    output = Path(args.output or f"results/paired_{args.arm}.json")
    locked = Path(f"results/paired_{args.arm}.json")
    if output.resolve() == locked.resolve() and locked.exists() and args.limit is None:
        assert_matches_locked(result, json.loads(locked.read_text()))
        print(json.dumps(result, indent=2), flush=True)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
