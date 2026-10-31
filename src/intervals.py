"""Patient-cluster intervals for the classwise ECE change."""
import numpy as np

from .metrics import classification_metrics


def classwise_delta(labels, clean_probability, paper_probability, classes, bins=15):
    clean = classification_metrics(labels, clean_probability, classes, bins)
    paper = classification_metrics(labels, paper_probability, classes, bins)
    delta_ece = {name: paper['per_class_ece'][name] - clean['per_class_ece'][name] for name in classes}
    delta_auroc = {}
    for name in classes:
        clean_auc, paper_auc = clean['per_class_auroc'][name], paper['per_class_auroc'][name]
        delta_auroc[name] = None if clean_auc is None or paper_auc is None else paper_auc - clean_auc
    return {'delta_classwise_ece': delta_ece, 'delta_classwise_auroc': delta_auroc,
            'clean': clean, 'paper': paper}


def bootstrap_classwise_delta_ece(labels, clean_probability, paper_probability, classes,
                                  patient_ids, bins=15, resamples=1000, seed=42,
                                  percentiles=(2.5, 97.5)):
    labels = np.asarray(labels)
    clean_probability = np.asarray(clean_probability)
    paper_probability = np.asarray(paper_probability)
    patient_ids = np.asarray(patient_ids)
    if len(labels) != len(patient_ids):
        raise ValueError('Every record needs a patient id')
    point = classwise_delta(labels, clean_probability, paper_probability, classes, bins)
    patients = np.unique(patient_ids)
    groups = {patient: np.flatnonzero(patient_ids == patient) for patient in patients}
    rng = np.random.default_rng(seed)
    draws = {name: np.empty(resamples, dtype=np.float64) for name in classes}
    for draw in range(resamples):
        chosen = rng.choice(patients, size=len(patients), replace=True)
        index = np.concatenate([groups[patient] for patient in chosen])
        delta = classwise_delta(labels[index], clean_probability[index], paper_probability[index],
                                classes, bins)['delta_classwise_ece']
        for name in classes:
            draws[name][draw] = delta[name]
    low_q, high_q = percentiles
    return {
        'point': point['delta_classwise_ece'],
        'low': {name: float(np.percentile(draws[name], low_q)) for name in classes},
        'high': {name: float(np.percentile(draws[name], high_q)) for name in classes},
        'resamples': resamples, 'seed': seed, 'cluster': 'patient',
        'percentiles': [low_q, high_q],
    }


def _arm_difference(labels, clean_raw, paper_raw, clean_standardized, paper_standardized, classes, bins):
    raw = classwise_delta(labels, clean_raw, paper_raw, classes, bins)['delta_classwise_ece']
    standardized = classwise_delta(
        labels, clean_standardized, paper_standardized, classes, bins)['delta_classwise_ece']
    return {name: raw[name] - standardized[name] for name in classes}


def bootstrap_arm_contrast(labels, clean_raw, paper_raw, clean_standardized, paper_standardized,
                           classes, patient_ids, bins=15, resamples=1000, seed=42,
                           percentiles=(2.5, 97.5)):
    """Patient bootstrap of raw-millivolt ΔECE minus standardized ΔECE on the same records."""
    labels = np.asarray(labels)
    clean_raw = np.asarray(clean_raw)
    paper_raw = np.asarray(paper_raw)
    clean_standardized = np.asarray(clean_standardized)
    paper_standardized = np.asarray(paper_standardized)
    patient_ids = np.asarray(patient_ids)
    if not (len(labels) == len(patient_ids) == len(clean_raw) == len(paper_raw)
            == len(clean_standardized) == len(paper_standardized)):
        raise ValueError('Both arms need the same records')
    point = _arm_difference(
        labels, clean_raw, paper_raw, clean_standardized, paper_standardized, classes, bins)
    patients = np.unique(patient_ids)
    groups = {patient: np.flatnonzero(patient_ids == patient) for patient in patients}
    rng = np.random.default_rng(seed)
    draws = {name: np.empty(resamples, dtype=np.float64) for name in classes}
    for draw in range(resamples):
        chosen = rng.choice(patients, size=len(patients), replace=True)
        index = np.concatenate([groups[patient] for patient in chosen])
        delta = _arm_difference(
            labels[index], clean_raw[index], paper_raw[index],
            clean_standardized[index], paper_standardized[index], classes, bins)
        for name in classes:
            draws[name][draw] = delta[name]
    low_q, high_q = percentiles
    return {
        'contrast': 'raw_mv minus standardized',
        'point': point,
        'low': {name: float(np.percentile(draws[name], low_q)) for name in classes},
        'high': {name: float(np.percentile(draws[name], high_q)) for name in classes},
        'resamples': resamples, 'seed': seed, 'cluster': 'patient',
        'percentiles': [low_q, high_q],
    }
