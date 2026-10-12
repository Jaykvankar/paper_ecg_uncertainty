"""Uncalibrated multilabel AUROC, classwise ECE and Brier score."""
import numpy as np


def binary_auc(y, probability):
    y = np.asarray(y)
    p = np.asarray(probability)
    positive = int(y.sum())
    negative = len(y) - positive
    if not positive or not negative:
        return None
    # Average ranks for tied scores; equivalent to the Mann-Whitney statistic.
    _, inverse, counts = np.unique(p, return_inverse=True, return_counts=True)
    ranks = np.cumsum(counts) - (counts - 1) / 2
    return float((ranks[inverse][y == 1].sum() - positive * (positive + 1) / 2)
                 / (positive * negative))


def classification_metrics(labels, probabilities, classes, bins=15):
    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 2 or len(y) == 0 or y.shape[1] != len(classes):
        raise ValueError('Expected matching nonempty (records, classes) arrays')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid binary labels or probabilities')
    if bins < 1:
        raise ValueError('bins must be positive')
    auc, ece, brier = {}, {}, {}
    for j, name in enumerate(classes):
        auc[name] = binary_auc(y[:, j], p[:, j])
        assignments = np.minimum((p[:, j] * bins).astype(int), bins - 1)
        error = 0.0
        for b in range(bins):
            mask = assignments == b
            if mask.any():
                error += mask.mean() * abs(y[mask, j].mean() - p[mask, j].mean())
        ece[name] = float(error)
        brier[name] = float(np.mean((p[:, j] - y[:, j]) ** 2))
    return {
        'records': len(y),
        'macro_auroc': float(np.mean(list(auc.values()))) if None not in auc.values() else None,
        'per_class_auroc': auc, 'ece': float(np.mean(list(ece.values()))),
        'per_class_ece': ece, 'brier': float(np.mean(list(brier.values()))),
        'per_class_brier': brier, 'ece_bins': bins,
        'ece_definition': 'Mean classwise equal-width positive-probability ECE',
        'brier_definition': 'Mean squared error across all record/class pairs',
    }
