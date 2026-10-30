"""Compare the two existing tracers on fold 9, then score fold 10 only if a new one wins.

The locked paired JSON files are not rewritten. The clean waveform is used only for SNR.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import load_signal
from .digitize import lead_correlation, recover_leads, recover_leads_sequence
from .images import render_record
from .intervals import bootstrap_arm_contrast
from .models import ECGResNet
from .pages import page_dir
from .paired import arm_summary, json_ready, load_ready_model, prepare_batch
from .protocol import load_protocol
from .sensitivity import agrees_with_locked
from .snr import leadwise_snr
from .train import atomic_json, predict, resolve_device

CHOICE = Path('results/tracer_sensitivity/validation_choice.json')
OUTPUT = Path('results/tracer_sensitivity/summary.json')
LOCKED_CONTRAST = Path('results/paired_pages/creased/test/arm_contrast.json')
LOCKED_SNR = 9.06
RUNS = {'standardized': Path('results/stage2'), 'raw_mv': Path('results/stage2_raw')}


def _page(split, filename):
    record_dir = page_dir('creased', split) / 'images' / Path(filename).name
    pages = sorted(record_dir.glob('*-0.png'))
    if not pages or not pages[0].with_suffix('.json').exists():
        raise FileNotFoundError(f'Missing creased page for {filename}')
    return pages[0], pages[0].with_suffix('.json')


def _trace(path, layout, method):
    image = np.asarray(Image.open(path))
    if method == 'dark_run':
        return recover_leads(image, layout)
    if method == 'image_to_sequence':
        return recover_leads_sequence(image, layout)
    raise ValueError(method)


def _snr(clean, traced):
    row = leadwise_snr(np.asarray(clean), np.asarray(traced), 100)
    return float(row['mean_snr_db'])


def compare_validation():
    """Mean SNR of both tracers on saved creased fold-9 pages."""
    saved = np.load(page_dir('creased', 'validation') / 'traces.npz', allow_pickle=False)
    filenames = [str(name) for name in saved['filename']]
    clean = saved['clean_mv']
    partial = CHOICE.with_name('validation_snr.npz')
    dark = np.full(len(filenames), np.nan, dtype=np.float64)
    sequence = np.full(len(filenames), np.nan, dtype=np.float64)
    done = np.zeros(len(filenames), dtype=bool)
    if partial.exists():
        stored = np.load(partial, allow_pickle=False)
        if len(stored['dark']) == len(filenames):
            dark = stored['dark']
            sequence = stored['sequence']
            done = stored['done']
        stored.close()
    dark_failures = []
    sequence_failures = []
    for index, filename in enumerate(filenames):
        if done[index]:
            if (index + 1) % 25 == 0 or index + 1 == len(filenames):
                print(f'Validation tracer comparison: {index + 1}/{len(filenames)}', flush=True)
            continue
        png, layout_path = _page('validation', filename)
        layout = json.loads(layout_path.read_text())
        try:
            dark[index] = _snr(clean[index], _trace(png, layout, 'dark_run'))
        except (ValueError, OSError) as error:
            dark[index] = np.nan
            dark_failures.append({'filename': filename, 'error': str(error)})
        try:
            sequence[index] = _snr(clean[index], _trace(png, layout, 'image_to_sequence'))
        except (ValueError, OSError) as error:
            sequence[index] = np.nan
            sequence_failures.append({'filename': filename, 'error': str(error)})
        done[index] = True
        if (index + 1) % 25 == 0 or index + 1 == len(filenames):
            partial.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(partial, dark=dark, sequence=sequence, done=done)
            print(f'Validation tracer comparison: {index + 1}/{len(filenames)}', flush=True)
    both = np.isfinite(dark) & np.isfinite(sequence)
    result = {
        'split': 'validation',
        'records': int(len(filenames)),
        'records_traced_by_both': int(both.sum()),
        'dark_run_mean_snr_db': None if not np.isfinite(dark).any() else float(np.nanmean(dark)),
        'image_to_sequence_mean_snr_db': None if not np.isfinite(sequence).any() else float(np.nanmean(sequence)),
        'dark_run_failures': int((~np.isfinite(dark)).sum()),
        'image_to_sequence_failures': int((~np.isfinite(sequence)).sum()),
    }
    CHOICE.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(CHOICE, result)
    return result


def _training_record_correlation(method):
    protocol = load_protocol()
    data = json.loads(Path('results/digitizer_check.json').read_text())
    if data.get('record') != 'records100/00000/00001_lr':
        raise ValueError('The training-record check must stay on 00001_lr')
    import yaml
    pilot = yaml.safe_load(Path('configs/pilot.yaml').read_text())
    record = Path(pilot['data_dir']) / data['record']
    png, layout_path = render_record(
        record, Path('results/tracer_sensitivity/training_page'), level='printed',
        seed=int(protocol['resolved']['standardized']['seed']))
    layout = json.loads(layout_path.read_text())
    recovered = _trace(png, layout, method)
    clean = load_signal(Path(pilot['data_dir']), data['record'])
    scores = lead_correlation(clean, recovered)
    minimum = float(min(scores.values()))
    return {
        'record': data['record'], 'level': 'printed', 'method': method,
        'lead_correlation': scores, 'minimum_correlation': minimum,
        'passed': minimum >= 0.80,
    }


def choose():
    comparison = compare_validation()
    dark = comparison['dark_run_mean_snr_db']
    sequence = comparison['image_to_sequence_mean_snr_db']
    sequence_better = (
        sequence is not None and dark is not None and sequence > dark
        and comparison['image_to_sequence_failures'] == 0)
    chosen = 'dark_run'
    reason = 'The dark-run reader stays. Fold 9 did not prefer image_to_sequence.'
    training = None
    if sequence_better:
        training = _training_record_correlation('image_to_sequence')
        if training['passed']:
            chosen = 'image_to_sequence'
            reason = 'image_to_sequence has the higher fold-9 SNR and passes the training-record gate.'
        else:
            reason = 'image_to_sequence has the higher fold-9 SNR but the training record correlates below 0.80.'
    elif comparison['image_to_sequence_failures']:
        reason = 'image_to_sequence failed on at least one fold-9 page. The dark-run reader stays.'
    choice = {**comparison, 'chosen': chosen, 'reason': reason, 'training_record': training}
    atomic_json(CHOICE, choice)
    return choice


def _score_test(method):
    protocol = load_protocol()
    saved = np.load(page_dir('creased', 'test') / 'traces.npz', allow_pickle=False)
    filenames = [str(name) for name in saved['filename']]
    labels = saved['labels']
    patients = saved['patient_id']
    classes = [str(name) for name in saved['classes']]
    clean_mv = saved['clean_mv']
    traced = []
    snr = []
    for index, filename in enumerate(filenames):
        png, layout_path = _page('test', filename)
        page = _trace(png, json.loads(layout_path.read_text()), method)
        traced.append(page)
        snr.append(_snr(clean_mv[index], page))
        if (index + 1) % 50 == 0 or index + 1 == len(filenames):
            print(f'Test tracer {method}: {index + 1}/{len(filenames)}', flush=True)
    traced = np.stack(traced).astype(np.float32)
    snr = np.asarray(snr, dtype=np.float64)
    probabilities = {}
    arms = {}
    for arm, run_dir in RUNS.items():
        state, config, _ = load_ready_model(run_dir)
        if int(config['width']) != 32:
            raise ValueError('Fold 10 is rescored with the locked width-32 models only')
        device = resolve_device(config.get('device', 'cuda'), smoke=False)
        model = ECGResNet(**state['model_args']).to(device)
        model.load_state_dict(state['model'])
        epsilon = config['normalization_epsilon']
        clean = prepare_batch(clean_mv, config['standardize'], epsilon)
        paper = prepare_batch(traced, config['standardize'], epsilon)
        y_clean, p_clean = predict(
            model, DataLoader(TensorDataset(torch.from_numpy(clean), torch.from_numpy(labels)),
                              batch_size=config['batch_size']), device)
        _, p_paper = predict(
            model, DataLoader(TensorDataset(torch.from_numpy(paper), torch.from_numpy(labels)),
                              batch_size=config['batch_size']), device)
        arms[arm] = arm_summary(
            protocol, arm, 'creased', 'test', y_clean, p_clean, p_paper, patients, classes, config,
            snr, str(run_dir), 'records100/00000/00001_lr')
        probabilities[f'{arm}_clean'] = p_clean
        probabilities[f'{arm}_paper'] = p_paper
    contrast = bootstrap_arm_contrast(
        labels, probabilities['raw_mv_clean'], probabilities['raw_mv_paper'],
        probabilities['standardized_clean'], probabilities['standardized_paper'],
        classes, patients, 15, protocol['bootstrap']['resamples'], protocol['bootstrap']['seed'],
        tuple(protocol['bootstrap']['percentiles']))
    locked = json.loads(LOCKED_CONTRAST.read_text())
    finite = snr[np.isfinite(snr)]
    mean_snr = float(np.mean(finite)) if len(finite) else None
    agrees = agrees_with_locked(contrast, locked['point']['CD'], 'CD')
    return {
        'fold10_retraced': True,
        'method': method,
        'mean_snr_db': mean_snr,
        'snr_above_locked': bool(mean_snr is not None and mean_snr > LOCKED_SNR),
        'agrees_with_locked_contrast': agrees,
        'succeeded': bool(mean_snr is not None and mean_snr > LOCKED_SNR and agrees),
        'contrast': contrast,
        'standardized': arms['standardized'],
        'raw_mv': arms['raw_mv'],
    }


def summarize():
    choice = json.loads(CHOICE.read_text()) if CHOICE.exists() else choose()
    if choice.get('chosen') != 'image_to_sequence':
        summary = {
            'fold10_retraced': False,
            'method': 'dark_run',
            'mean_snr_db': LOCKED_SNR,
            'snr_above_locked': False,
            'succeeded': False,
            'reason': choice.get('reason'),
            'validation': {key: choice[key] for key in choice if key != 'training_record'},
        }
    else:
        summary = _score_test('image_to_sequence')
        summary['reason'] = choice.get('reason')
        summary['validation'] = {key: choice[key] for key in choice if key != 'training_record'}
    summary['primary_unchanged'] = True
    summary['locked_mean_snr_db'] = LOCKED_SNR
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(OUTPUT, json_ready(summary))
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['choose', 'summarize'])
    args = parser.parse_args()
    result = choose() if args.command == 'choose' else summarize()
    print(json.dumps({key: result[key] for key in result if key not in {'standardized', 'raw_mv', 'contrast'}}, indent=2), flush=True)


if __name__ == '__main__':
    main()
