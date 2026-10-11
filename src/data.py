"""Stage 1: download, validate and label PTB-XL 100 Hz records."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
from pathlib import Path
import threading

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import wfdb
import yaml

LEADS = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
_local = threading.local()


def session():
    if not hasattr(_local, 'session'):
        _local.session = requests.Session()
        _local.session.mount('https://', HTTPAdapter(max_retries=Retry(
            total=4, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))
    return _local.session


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(root, base, name, checksums=None):
    target = root / name
    expected = checksums[name] if checksums is not None else None
    if target.exists() and (expected is None or digest(target) == expected):
        return target
    response = session().get(base + name, timeout=(15, 90))
    response.raise_for_status()
    payload = response.content
    if expected is not None and hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f'SHA256 mismatch: {name}')
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + '.part')
    temporary.write_bytes(payload)
    temporary.replace(target)
    return target


def build_labels(root, config):
    frame = pd.read_csv(root / 'ptbxl_database.csv', index_col='ecg_id')
    statements = pd.read_csv(root / 'scp_statements.csv', index_col=0)
    mapping = statements.loc[statements.diagnostic == 1, 'diagnostic_class'].dropna().to_dict()
    classes = config['classes']
    assert set(mapping.values()) <= set(classes), 'Unexpected diagnostic superclass'
    assert frame.index.is_unique and frame.patient_id.notna().all()
    assert frame.strat_fold.isin(range(1, 11)).all()
    frame['split'] = ''
    for split, folds in config['splits'].items():
        selected = frame.strat_fold.isin(folds)
        assert (frame.loc[selected, 'split'] == '').all(), 'Overlapping fold assignments'
        frame.loc[selected, 'split'] = split
    assert (frame.split != '').all(), 'Unassigned fold'
    patients = {s: set(frame.loc[frame.split == s, 'patient_id']) for s in config['splits']}
    for a, b in itertools.combinations(patients, 2):
        assert patients[a].isdisjoint(patients[b]), f'Patient leakage: {a}/{b}'
    # Follow the official example: diagnostic SCP presence, including likelihood 0.
    labels = frame.scp_codes.map(ast.literal_eval).map(
        lambda codes: {mapping[code] for code in codes if code in mapping})
    for cls in classes:
        frame[cls] = labels.map(lambda values: int(cls in values))
    frame['has_diagnostic_label'] = frame[classes].sum(axis=1) > 0
    return frame


def load_signal(root, filename):
    signal, meta = wfdb.rdsamp(str(root / filename), return_res=32)
    assert meta['fs'] == 100, f'Wrong sample rate: {filename}'
    assert signal.shape == (1000, 12), f'Wrong shape: {filename}: {signal.shape}'
    assert [s.upper() for s in meta['sig_name']] == LEADS, f'Wrong leads: {filename}'
    assert all(unit == 'mV' for unit in meta['units']), f'Wrong units: {filename}'
    assert np.isfinite(signal).all(), f'Nonfinite waveform: {filename}'
    return signal.T.copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/pilot.yaml')
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    assert config['sampling_rate'] == 100
    root, out = Path(config['data_dir']), Path(config['results_dir'])
    root.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    base = config['download_base']
    if args.download:
        fetch(root, base, 'SHA256SUMS.txt')
    checksums = {}
    for line in (root / 'SHA256SUMS.txt').read_text().splitlines():
        checksum, name = line.split(maxsplit=1)
        checksums[name.lstrip('*').removeprefix('./')] = checksum
    metadata = ['ptbxl_database.csv', 'scp_statements.csv', 'LICENSE.txt', 'example_physionet.py']
    for name in metadata:
        if args.download:
            fetch(root, base, name, checksums)
        assert digest(root / name) == checksums[name], f'Invalid metadata: {name}'
    frame = build_labels(root, config)
    print(f"PTB-XL {config['dataset_version']}: {len(frame)} metadata records", flush=True)
    print('Patient leakage assertion: PASS (all three split pairs disjoint)', flush=True)

    def validate(filename):
        assert filename.startswith('records100/') and '..' not in filename
        for ext in ('.hea', '.dat'):
            name = filename + ext
            if args.download:
                fetch(root, base, name, checksums)
            else:
                assert digest(root / name) == checksums[name], f'Invalid waveform: {name}'
        load_signal(root, filename)

    with ThreadPoolExecutor(max_workers=config['download_workers']) as executor:
        for n, _ in enumerate(executor.map(validate, frame.filename_lr), 1):
            if n % 500 == 0 or n == len(frame):
                print(f'Loaded and validated 100 Hz records: {n}/{len(frame)}', flush=True)
    rows = []
    for split in config['splits']:
        part = frame.loc[frame.split == split]
        rows.append({'split': split, 'records': len(part), 'patients': part.patient_id.nunique(),
                     **{c: int(part[c].sum()) for c in config['classes']},
                     'unlabeled': int((~part.has_diagnostic_label).sum())})
    counts = pd.DataFrame(rows)
    counts.to_csv(out / 'class_counts.csv', index=False)
    columns = ['patient_id', 'strat_fold', 'split', 'filename_lr', *config['classes'], 'has_diagnostic_label']
    frame[columns].to_csv(out / 'record_labels.csv')
    result = {'config': config, 'source': base, 'metadata_records': len(frame),
              'waveforms_validated': len(frame), 'waveform_shape': [12, 1000],
              'units': 'mV', 'lead_order': LEADS, 'patient_leakage_assertion': 'PASS',
              'counts': rows, 'metadata_sha256': {n: digest(root / n) for n in metadata},
              'source_manifest_sha256': digest(root / 'SHA256SUMS.txt')}
    (out / 'stage1.json').write_text(json.dumps(result, indent=2) + '\n')
    output = (f"PTB-XL {config['dataset_version']}: {len(frame)} records loaded at 100 Hz\n"
              'Waveform validation: PASS (12 leads x 1000 samples, mV, finite, SHA256 verified)\n'
              'Patient leakage assertion: PASS (all three split pairs disjoint)\n'
              + counts.to_string(index=False) + '\n')
    print(output, end='')


if __name__ == '__main__':
    main()
