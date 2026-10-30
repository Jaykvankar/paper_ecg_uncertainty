"""External check on color scans of printed PTB-XL pages.

Fold 10 scans are scored only after a training-fold scan passes the correlation gate.
The corrector is not refit here.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, TensorDataset

from .baseline_data import baseline_frame
from .corrector import apply_probabilities, load_coefficients
from .data import LEADS, load_signal
from .digitize import lead_correlation, recover_leads
from .intervals import bootstrap_classwise_delta_ece, classwise_delta
from .models import ECGResNet
from .pages import lead_std_mv
from .paired import _arm_data, json_ready, load_ready_model, prepare_batch
from .protocol import load_protocol
from .snr import leadwise_snr
from .train import atomic_json, predict, resolve_device
from .uncertainty import mean_binary_entropy

GATE = Path('results/real_scan/gate.json')
OUTPUT = Path('results/real_scan')


def layout_from_bands(image):
    """One-column lead boxes from the page edges. Diagnosis is not read."""
    height, width = np.asarray(image).shape[:2]
    band = height / len(LEADS)
    leads = []
    for index, name in enumerate(LEADS):
        top = int(round(index * band))
        bottom = int(round((index + 1) * band)) - 1
        leads.append({
            'lead_name': name,
            'start_sample': 0,
            'end_sample': 1000,
            'lead_bounding_box': {
                '0': [top, 0], '1': [top, width - 1],
                '2': [bottom, width - 1], '3': [bottom, 0],
            },
        })
    return {'y_grid': max(band / 10.0, 1.0), 'leads': leads}


def _layout_for(image_path):
    sidecar = image_path.with_suffix('.json')
    if sidecar.exists():
        layout = json.loads(sidecar.read_text())
        if 'leads' in layout and 'y_grid' in layout:
            return layout
    return layout_from_bands(np.asarray(Image.open(image_path)))


def _record_key(filename):
    stem = Path(filename).stem
    if stem.endswith('_lr'):
        stem = stem[:-3]
    return stem.lstrip('0') or '0'


def index_images(root, variant_id):
    index = {}
    if not variant_id or not Path(root).exists():
        return index
    for path in Path(root).rglob('*'):
        if path.suffix.lower() not in {'.png', '.jpg', '.jpeg'}:
            continue
        if variant_id not in path.stem:
            continue
        tokens = ''.join(character if character.isdigit() else ' ' for character in path.stem).split()
        for token in tokens:
            index.setdefault(token.lstrip('0') or '0', path)
    return index


def _blocked(path, reason):
    result = {'status': 'blocked', 'reason': reason}
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, result)
    return result


def gate(protocol=None, gate_path=None):
    protocol = protocol or load_protocol()
    path = Path(gate_path or GATE)
    block = protocol['real_scan']
    if not block.get('variant_id'):
        return _blocked(path, 'color-scan variant_id is unset')
    root = Path(block['data_dir'])
    source = root / 'D0' if (root / 'D0').exists() else root
    if not source.exists():
        return _blocked(path, 'D0 images are not on disk')
    images = index_images(source, str(block['variant_id']))
    arm = _arm_data(protocol)
    frame = baseline_frame(arm['data'])
    train = frame.loc[frame.split == 'train'].sort_index()
    chosen = None
    image_path = None
    for _, row in train.iterrows():
        image_path = images.get(_record_key(row.filename_lr))
        if image_path is not None:
            chosen = row
            break
    if chosen is None:
        return _blocked(path, 'no training-fold color scan matched a PTB-XL record')
    image = np.asarray(Image.open(image_path))
    layout = _layout_for(image_path)
    try:
        recovered, _ = recover_leads(image, layout, return_gap=True)
        clean = load_signal(Path(arm['data']['data_dir']), chosen.filename_lr)
        scores = lead_correlation(clean, recovered)
    except (ValueError, OSError) as error:
        return _blocked(path, f'training-fold scan could not be traced: {error}')
    minimum = min(scores.values())
    threshold = float(block['correlation_threshold'])
    result = {
        'status': 'passed' if minimum >= threshold else 'blocked',
        'reason': None if minimum >= threshold else 'training-fold scan correlation is below 0.80',
        'record': chosen.filename_lr,
        'split': 'train',
        'image': str(image_path),
        'lead_correlation': scores,
        'minimum_correlation': minimum,
        'threshold': threshold,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, result)
    return result


def score(protocol=None, gate_path=None, output_dir=None):
    protocol = protocol or load_protocol()
    gate_file = Path(gate_path or GATE)
    destination = Path(output_dir or OUTPUT)
    fold10 = destination / 'fold10_raw_mv.json'
    if not gate_file.exists():
        return {'status': 'blocked', 'reason': 'gate file missing'}
    stored = json.loads(gate_file.read_text())
    if stored.get('status') != 'passed':
        return {'status': 'blocked', 'reason': stored.get('reason') or stored.get('status')}
    block = protocol['real_scan']
    root = Path(block['data_dir'])
    source = root / 'D0' if (root / 'D0').exists() else root
    images = index_images(source, str(block['variant_id']))
    arm_config = _arm_data(protocol)
    frame = baseline_frame(arm_config['data'])
    test = frame.loc[frame.split == 'test'].sort_index()
    rows, clean_mv, traced_mv, gaps = [], [], [], []
    root_data = Path(arm_config['data']['data_dir'])
    for _, row in test.iterrows():
        image_path = images.get(_record_key(row.filename_lr))
        if image_path is None:
            continue
        image = np.asarray(Image.open(image_path))
        recovered, gap = recover_leads(image, _layout_for(image_path), return_gap=True)
        clean_mv.append(load_signal(root_data, row.filename_lr).astype(np.float32))
        traced_mv.append(recovered)
        gaps.append(gap)
        rows.append(row)
    if not rows:
        return {'status': 'blocked', 'reason': 'no fold-10 color scans matched PTB-XL records'}
    part = frame.loc[[row.name for row in rows]]
    classes = list(arm_config['data']['classes'])
    labels = part[classes].to_numpy(dtype=np.float32)
    patients = part.patient_id.to_numpy()
    clean_mv = np.stack(clean_mv)
    traced_mv = np.stack(traced_mv)
    coefficients = load_coefficients()
    summaries = {}
    destination.mkdir(parents=True, exist_ok=True)
    for arm in protocol['arms']:
        spec = protocol['resolved'][arm]
        state, config, _ = load_ready_model(spec['results'])
        device = resolve_device(config.get('device', 'cuda'), smoke=False)
        model = ECGResNet(**state['model_args']).to(device)
        model.load_state_dict(state['model'])
        epsilon = config['normalization_epsilon']
        clean = prepare_batch(clean_mv, spec['standardize'], epsilon)
        paper = prepare_batch(traced_mv, spec['standardize'], epsilon)
        y_clean, p_clean = predict(
            model, DataLoader(TensorDataset(torch.from_numpy(clean), torch.from_numpy(labels)),
                              batch_size=config['batch_size']), device)
        _, p_paper = predict(
            model, DataLoader(TensorDataset(torch.from_numpy(paper), torch.from_numpy(labels)),
                              batch_size=config['batch_size']), device)
        delta = classwise_delta(y_clean, p_clean, p_paper, classes, config['ece_bins'])
        interval = bootstrap_classwise_delta_ece(
            y_clean, p_clean, p_paper, classes, patients, config['ece_bins'],
            protocol['bootstrap']['resamples'], protocol['bootstrap']['seed'],
            tuple(protocol['bootstrap']['percentiles']))
        summary = {
            'status': 'real_scan_scored',
            'arm': arm,
            'source': 'D0',
            'records': int(len(part)),
            'delta_classwise_ece': delta['delta_classwise_ece'],
            'delta_classwise_auroc': delta['delta_classwise_auroc'],
            'patient_bootstrap': interval,
            'paper_macro_auroc': delta['paper']['macro_auroc'],
        }
        if arm == 'raw_mv':
            corrected = apply_probabilities(
                coefficients, p_paper, lead_std_mv(traced_mv), np.asarray(gaps, dtype=np.float64))
            corrected_delta = classwise_delta(y_clean, p_clean, corrected, classes, config['ece_bins'])
            summary['corrected_delta_ece'] = corrected_delta['delta_classwise_ece']
            summary['corrected_bootstrap'] = bootstrap_classwise_delta_ece(
                y_clean, p_clean, corrected, classes, patients, config['ece_bins'],
                protocol['bootstrap']['resamples'], protocol['bootstrap']['seed'],
                tuple(protocol['bootstrap']['percentiles']))
            summary['corrector_fit_split'] = coefficients['fit_split']
            entropy_clean = mean_binary_entropy(p_clean)
            entropy_paper = mean_binary_entropy(corrected)
            summary['delta_mean_binary_entropy'] = {
                name: float(entropy_paper[index] - entropy_clean[index]) for index, name in enumerate(classes)
            }
        atomic_json(destination / f'fold10_{arm}.json', json_ready(summary))
        summaries[arm] = summary
    if not fold10.exists():
        raise RuntimeError('The raw-millivolt fold-10 scan file was not written')
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['gate', 'score'])
    parser.add_argument('--protocol', default='configs/paired.yaml')
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    result = gate(protocol) if args.command == 'gate' else score(protocol)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
