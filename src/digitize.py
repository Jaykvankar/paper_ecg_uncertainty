"""Recover 12 leads from a rendered page.

Lead boxes only locate the trace. Values come from the column max-finder
used by ECG-Image-Kit image_to_sequence. Stored plotted pixels are ignored.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

from .data import LEADS, load_signal
from .train import atomic_json

MODEL_LEADS = LEADS


def canon_lead(name):
    if name is None:
        return None
    key = ''.join(str(name).upper().split())
    if key.startswith('FULL'):
        return None
    return key if key in MODEL_LEADS else None


def image_to_sequence(mask):
    """MATLAB image_to_sequence max-finder: 0 is the bottom row, and the first maximum wins."""
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError('image_to_sequence expects one grayscale page')
    height, width = mask.shape
    values = np.empty(width, dtype=np.float64)
    present = np.zeros(width, dtype=bool)
    for column in range(width):
        stripe = mask[:, column]
        if stripe.max() <= 0:
            values[column] = np.nan
            continue
        max_idx = int(np.argmax(stripe)) + 1
        values[column] = height - max_idx
        present[column] = True
    if not present.any():
        raise ValueError('No ink in the lead crop')
    index = np.arange(width)
    values[~present] = np.interp(index[~present], index[present], values[present])
    return values


def _box_limits(box):
    points = list(box.values()) if isinstance(box, dict) else list(box)
    rows = [float(point[0]) for point in points]
    cols = [float(point[1]) for point in points]
    return int(np.floor(min(rows))), int(np.ceil(max(rows))), int(np.floor(min(cols))), int(np.ceil(max(cols)))


def _luminance(image):
    array = np.asarray(image)
    if array.ndim == 2:
        return array.astype(np.float64)
    return (0.2989 * array[..., 0] + 0.5870 * array[..., 1] + 0.1140 * array[..., 2]).astype(np.float64)


def _dark_runs(stripe, tolerance=12):
    cutoff = float(stripe.min()) + tolerance
    index = np.flatnonzero(stripe <= cutoff)
    if len(index) == 0:
        return []
    runs = []
    start = previous = index[0]
    for value in index[1:]:
        if value == previous + 1:
            previous = value
        else:
            runs.append(0.5 * (start + previous))
            start = previous = value
    runs.append(0.5 * (start + previous))
    return runs


def _resample(values, n_samples):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == n_samples:
        return values
    source = np.linspace(0.0, 1.0, len(values))
    return np.interp(np.linspace(0.0, 1.0, n_samples), source, values)


def recover_leads(image, layout, n_samples=1000, return_gap=False):
    """Read each lead with the column max-finder inside its box.

    A tall QRS can enter the next lead's box. Rows that belong to only one box
    estimate that lead's baseline, and each column keeps the dark run nearest
    that baseline. Plotted-pixel coordinates are never read.
    """
    if 'y_grid' not in layout:
        raise ValueError('The page layout has no voltage scale')
    gray = _luminance(image)
    entries = []
    for entry in layout['leads']:
        entry = {key: value for key, value in entry.items() if key != 'plotted_pixels'}
        if 'lead_bounding_box' not in entry:
            continue
        entries.append((entry, _box_limits(entry['lead_bounding_box'])))
    millivolts_per_pixel = 0.5 / float(layout['y_grid'])
    found = {}
    gaps = []
    for entry, span in entries:
        name = canon_lead(entry.get('lead_name'))
        if name is None or name in found:
            continue
        top, bottom, left, right = span

        # Clip the bounding box to the actual image dimensions.
        image_height, image_width = gray.shape[:2]
        top = max(0, top)
        bottom = min(image_height - 1, bottom)
        left = max(0, left)
        right = min(image_width - 1, right)

        if bottom <= top or right <= left:
            raise ValueError(f'{name} has an empty box')

        exclusive = np.ones(bottom - top + 1, dtype=bool)
        for _, other in entries:
            if other == span:
                continue
            if other[1] < top or other[0] > bottom:
                continue
            exclusive[max(top, other[0]) - top:min(bottom, other[1]) - top + 1] = False
        if int(exclusive.sum()) < 5:
            exclusive[:] = True
        crop = gray[top:bottom + 1, left:right + 1]
        seeds = []
        for column in range(crop.shape[1]):
            stripe = crop[:, column].copy()
            stripe[~exclusive] = 255
            runs = _dark_runs(stripe)
            if runs:
                seeds.append(float(np.median(runs)))
        if not seeds:
            raise ValueError(f'No ink for {name}')
        reference = float(np.median(seeds))
        rows = np.empty(crop.shape[1], dtype=np.float64)
        empty = 0
        for column in range(crop.shape[1]):
            runs = _dark_runs(crop[:, column])
            if not runs:
                empty += 1
                rows[column] = reference
            else:
                rows[column] = min(runs, key=lambda row: abs(row - reference))
        gaps.append(empty / crop.shape[1])
        image_row = top + rows
        centered = (np.median(image_row) - image_row) * millivolts_per_pixel
        start, end = int(entry['start_sample']), int(entry['end_sample'])
        if end <= start:
            raise ValueError(f'{name} has an empty time span')
        piece = _resample(centered, end - start).astype(np.float32)
        full = np.full(n_samples, np.nan, dtype=np.float32)
        full[start:end] = piece
        if not np.isfinite(full).all():
            raise ValueError(f'{name} was not plotted across the full record')
        found[name] = full
    missing = [name for name in MODEL_LEADS if name not in found]
    if missing:
        raise ValueError(f'Missing leads: {missing}')
    waveform = np.stack([found[name] for name in MODEL_LEADS]).astype(np.float32)
    if return_gap:
        return waveform, float(np.mean(gaps))
    return waveform


def recover_leads_sequence(image, layout, n_samples=1000):
    """Same lead boxes and voltage scale as recover_leads.

    The column position comes from image_to_sequence. Page ink is dark, so each
    crop is inverted before that max-finder. The clean waveform is not read.
    """
    if 'y_grid' not in layout:
        raise ValueError('The page layout has no voltage scale')
    gray = _luminance(image)
    entries = []
    for entry in layout['leads']:
        entry = {key: value for key, value in entry.items() if key != 'plotted_pixels'}
        if 'lead_bounding_box' not in entry:
            continue
        entries.append((entry, _box_limits(entry['lead_bounding_box'])))
    millivolts_per_pixel = 0.5 / float(layout['y_grid'])
    found = {}
    image_height, image_width = gray.shape[:2]
    for entry, span in entries:
        name = canon_lead(entry.get('lead_name'))
        if name is None or name in found:
            continue
        top, bottom, left, right = span
        top = max(0, top)
        bottom = min(image_height - 1, bottom)
        left = max(0, left)
        right = min(image_width - 1, right)
        if bottom <= top or right <= left:
            raise ValueError(f'{name} has an empty box')
        crop = gray[top:bottom + 1, left:right + 1]
        ink = crop.max() - crop
        mask = ink.copy()
        for _, other in entries:
            if other == span:
                continue
            if other[1] < top or other[0] > bottom:
                continue
            row_start = max(top, other[0]) - top
            row_stop = min(bottom, other[1]) - top + 1
            mask[row_start:row_stop, :] = 0
        if mask.max() <= 0:
            mask = ink
        sequence = image_to_sequence(mask)
        row_from_top = (crop.shape[0] - 1) - sequence
        image_row = top + row_from_top
        centered = (np.median(image_row) - image_row) * millivolts_per_pixel
        start, end = int(entry['start_sample']), int(entry['end_sample'])
        if end <= start:
            raise ValueError(f'{name} has an empty time span')
        piece = _resample(centered, end - start).astype(np.float32)
        full = np.full(n_samples, np.nan, dtype=np.float32)
        full[start:end] = piece
        if not np.isfinite(full).all():
            raise ValueError(f'{name} was not plotted across the full record')
        found[name] = full
    missing = [name for name in MODEL_LEADS if name not in found]
    if missing:
        raise ValueError(f'Missing leads: {missing}')
    return np.stack([found[name] for name in MODEL_LEADS]).astype(np.float32)


def lead_correlation(clean, recovered):
    scores = {}
    for name, left, right in zip(MODEL_LEADS, clean, recovered):
        left = np.asarray(left, dtype=np.float64)
        right = np.asarray(right, dtype=np.float64)
        left = left - left.mean()
        right = right - right.mean()
        scale = np.linalg.norm(left) * np.linalg.norm(right)
        scores[name] = float(np.dot(left, right) / scale) if scale else 0.0
    return scores


def check_training_record(protocol_path='configs/paired.yaml', data_config='configs/pilot.yaml',
                          output=None):
    from .images import render_record
    from .protocol import load_protocol

    protocol = load_protocol(protocol_path)
    data = yaml.safe_load(Path(data_config).read_text())
    labels = Path(data['results_dir']) / 'record_labels.csv'
    header = labels.read_text().splitlines()[0].split(',')
    row = None
    for line in labels.read_text().splitlines()[1:]:
        fields = line.split(',')
        if fields[0] == '1':
            row = dict(zip(header, fields))
            break
    if row is None or row['split'] != 'train' or row['filename_lr'] != 'records100/00000/00001_lr':
        raise ValueError('The digitizer check record must stay the training record 00001_lr')
    record = Path(data['data_dir']) / row['filename_lr']
    page_dir = Path('.artifacts/digitizer_check')
    png, layout_path = render_record(record, page_dir, level='printed', seed=int(protocol['resolved']['standardized']['seed']))
    layout = json.loads(layout_path.read_text())
    for lead in layout['leads']:
        lead.pop('plotted_pixels', None)
    image = np.asarray(Image.open(png))
    recovered = recover_leads(image, layout)
    clean = load_signal(Path(data['data_dir']), row['filename_lr'])
    scores = lead_correlation(clean, recovered)
    minimum = min(scores.values())
    threshold = float(protocol['correlation_threshold'])
    result = {
        'status': 'passed' if minimum >= threshold else 'failed',
        'record': row['filename_lr'], 'split': row['split'], 'strat_fold': int(float(row['strat_fold'])),
        'level': 'printed', 'lead_correlation': scores, 'minimum_correlation': minimum,
        'threshold': threshold, 'page': str(png),
        'digitizer': protocol['digitizer'],
        'note': 'Layout boxes locate leads. The max-finder supplies the waveform. Fold 10 was not scored.',
    }
    destination = Path(output or protocol['digitizer_check'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(destination, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--protocol', default='configs/paired.yaml')
    args = parser.parse_args()
    if not args.check:
        raise SystemExit('Pass --check to trace training record 00001_lr. This does not score fold 10.')
    result = check_training_record(args.protocol)
    print(json.dumps(result, indent=2), flush=True)
    if result['status'] != 'passed':
        raise SystemExit('Digitizer check failed. Do not score fold 10.')


if __name__ == '__main__':
    main()
