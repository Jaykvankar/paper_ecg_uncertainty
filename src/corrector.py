"""Trace-only logistic corrector for the raw-millivolt arm.

Features never include the clean waveform. The map is fit on synthetic fold 9
and applied once to fold 10.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .intervals import bootstrap_classwise_delta_ece, classwise_delta
from .metrics import classification_metrics
from .pages import lead_std_mv, page_dir, probability_logit
from .protocol import CORRECTOR_FEATURES, load_protocol
from .train import atomic_json

COEFFICIENTS = Path('results/corrector/raw_mv.json')
LOGIT_COEFFICIENTS = Path('results/corrector/logit_only.json')
FOLD10 = Path('results/corrector/fold10_raw_mv.json')
LOGIT_FEATURES = ('logit',)


def assert_trace_only(features):
    for name in features:
        if 'clean' in str(name).lower():
            raise ValueError('A corrector feature must not use the clean waveform')
    if tuple(features) != CORRECTOR_FEATURES:
        raise ValueError('Corrector features must be the trace-only set')


def load_coefficients(path=COEFFICIENTS):
    payload = json.loads(Path(path).read_text())
    if payload.get('fit_split') != 'validation':
        raise ValueError('The corrector must be fit on validation')
    assert_trace_only(payload['features'])
    return payload


def _fit_logistic(features, labels, l2=1e-2, steps=50):
    design = np.column_stack([np.ones(len(features)), np.asarray(features, dtype=np.float64)])
    outcome = np.asarray(labels, dtype=np.float64)
    weights = np.zeros(design.shape[1])
    penalty = np.ones(design.shape[1])
    penalty[0] = 0.0
    for _ in range(steps):
        eta = np.clip(design @ weights, -20, 20)
        probability = 1 / (1 + np.exp(-eta))
        diagonal = probability * (1 - probability)
        gram = design.T @ (design * diagonal[:, None]) / len(outcome)
        gram.flat[::gram.shape[0] + 1] += l2 * penalty
        gradient = design.T @ (probability - outcome) / len(outcome) + l2 * penalty * weights
        step = np.linalg.solve(gram, gradient)
        weights -= step
    return weights


def _matrices(split, arm='raw_mv'):
    pages = page_dir('creased', split)
    traces = np.load(pages / 'traces.npz', allow_pickle=False)
    probabilities = np.load(pages / 'probabilities.npz', allow_pickle=False)
    paper = probabilities[f'{arm}_paper']
    clean = probabilities[f'{arm}_clean']
    lead_std = lead_std_mv(traces['traced_mv'])
    gap = np.asarray(traces['ink_gap'], dtype=np.float64)
    return traces, clean, paper, lead_std, gap


def fit(protocol=None):
    protocol = protocol or load_protocol()
    block = protocol['corrector']
    assert_trace_only(block['features'])
    if block['fit_split'] != 'validation' or block['arm'] != 'raw_mv':
        raise ValueError('The corrector fit is locked to raw millivolts on validation')
    traces, _, paper, lead_std, gap = _matrices('validation')
    labels = traces['labels']
    classes = [str(name) for name in traces['classes']]
    payload = {
        'fit_split': 'validation',
        'arm': 'raw_mv',
        'features': list(CORRECTOR_FEATURES),
        'target_role': 'secondary',
        'classes': classes,
        'models': _fit_models(paper, labels, classes, CORRECTOR_FEATURES, lead_std, gap),
        'records': int(len(labels)),
    }
    logit_payload = {
        'fit_split': 'validation',
        'arm': 'raw_mv',
        'features': list(LOGIT_FEATURES),
        'classes': classes,
        'models': _fit_models(paper, labels, classes, LOGIT_FEATURES, lead_std, gap),
        'records': int(len(labels)),
    }
    COEFFICIENTS.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(COEFFICIENTS, payload)
    atomic_json(LOGIT_COEFFICIENTS, logit_payload)
    return payload


def _feature_columns(features, paper, lead_std, gap, index):
    for name in features:
        if 'clean' in str(name).lower():
            raise ValueError('A corrector feature must not use the clean waveform')
    logit = probability_logit(paper[:, index])
    pieces = []
    for name in features:
        if name == 'logit':
            pieces.append(logit)
        elif name == 'traced_lead_std_mv':
            pieces.append(lead_std)
        elif name == 'ink_gap_fraction':
            pieces.append(gap)
        else:
            raise ValueError(f'Unknown corrector feature {name}')
    return np.column_stack(pieces)


def _fit_models(paper, labels, classes, features, lead_std, gap):
    models = {}
    for index, name in enumerate(classes):
        column = _feature_columns(features, paper, lead_std, gap, index)
        mean = column.mean(axis=0)
        scale = np.maximum(column.std(axis=0), 1e-6)
        standardized = (column - mean) / scale
        models[name] = {
            'weights': _fit_logistic(standardized, labels[:, index]).tolist(),
            'mean': mean.tolist(),
            'std': scale.tolist(),
        }
    return models


def apply_probabilities(payload, paper, lead_std, gap):
    if payload.get('fit_split') != 'validation':
        raise ValueError('The corrector must be fit on validation')
    features = tuple(payload['features'])
    if features == CORRECTOR_FEATURES:
        assert_trace_only(features)
    elif features != LOGIT_FEATURES:
        raise ValueError('The comparison map must use the frozen logit only')
    corrected = np.empty_like(paper, dtype=np.float64)
    for index, name in enumerate(payload['classes']):
        model = payload['models'][name]
        column = _feature_columns(features, paper, lead_std, gap, index)
        standardized = (column - np.asarray(model['mean'])) / np.asarray(model['std'])
        design = np.column_stack([np.ones(len(column)), standardized])
        eta = np.clip(design @ np.asarray(model['weights']), -20, 20)
        corrected[:, index] = 1 / (1 + np.exp(-eta))
    return corrected


def trace_versus_logit(trace_delta, logit_delta, target, trace_auroc, uncorrected_auroc, limit):
    """Trace features earn their place only by beating the logit-only map."""
    closer = abs(trace_delta[target]) < abs(logit_delta[target])
    auroc_ok = trace_auroc >= uncorrected_auroc - limit
    return 'earned' if closer and auroc_ok else 'miss'


def apply_fold10(protocol=None):
    protocol = protocol or load_protocol()
    payload = load_coefficients()
    traces, clean, paper, lead_std, gap = _matrices('test')
    classes = [str(name) for name in traces['classes']]
    if classes != payload['classes']:
        raise ValueError('Corrector classes do not match the saved fold-10 traces')
    corrected = apply_probabilities(payload, paper, lead_std, gap)
    logit_payload = json.loads(LOGIT_COEFFICIENTS.read_text())
    logit_corrected = apply_probabilities(logit_payload, paper, lead_std, gap)
    labels = traces['labels']
    limit = float(protocol['corrector']['auroc_drop_limit'])
    target = protocol['corrector']['target_class']
    before = classwise_delta(labels, clean, paper, classes)
    after = classwise_delta(labels, clean, corrected, classes)
    logit_after = classwise_delta(labels, clean, logit_corrected, classes)
    uncorrected = classification_metrics(labels, paper, classes)
    corrected_metrics = classification_metrics(labels, corrected, classes)
    closer = abs(after['delta_classwise_ece'][target]) < abs(before['delta_classwise_ece'][target])
    auroc_ok = corrected_metrics['macro_auroc'] >= uncorrected['macro_auroc'] - limit
    versus_logit = trace_versus_logit(
        after['delta_classwise_ece'], logit_after['delta_classwise_ece'], target,
        corrected_metrics['macro_auroc'], uncorrected['macro_auroc'], limit)
    interval = bootstrap_classwise_delta_ece(
        labels, clean, corrected, classes, traces['patient_id'], 15,
        protocol['bootstrap']['resamples'], protocol['bootstrap']['seed'],
        tuple(protocol['bootstrap']['percentiles']))
    result = {
        'status': 'applied' if closer and auroc_ok else 'miss',
        'arm': 'raw_mv',
        'fit_split': payload['fit_split'],
        'apply_split': 'test',
        'features': payload['features'],
        'target_class': target,
        'target_role': 'secondary',
        'cd_closer_to_zero': bool(closer),
        'trace_versus_logit_only': versus_logit,
        'logit_only_delta_ece': logit_after['delta_classwise_ece'],
        'auroc_within_limit': bool(auroc_ok),
        'auroc_drop_limit': limit,
        'uncorrected_delta_ece': before['delta_classwise_ece'],
        'corrected_delta_ece': after['delta_classwise_ece'],
        'delta_classwise_ece': after['delta_classwise_ece'],
        'patient_bootstrap': interval,
        'uncorrected_macro_auroc': uncorrected['macro_auroc'],
        'corrected_macro_auroc': corrected_metrics['macro_auroc'],
        'records': int(len(labels)),
    }
    atomic_json(FOLD10, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['fit', 'apply'])
    parser.add_argument('--protocol', default='configs/paired.yaml')
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    result = fit(protocol) if args.command == 'fit' else apply_fold10(protocol)
    print(json.dumps({key: result[key] for key in result if key != 'models'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
