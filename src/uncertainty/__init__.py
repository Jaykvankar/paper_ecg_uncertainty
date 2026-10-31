"""Uncertainty quantification, entropy, and trace-stratified conformal risk control."""
import numpy as np


def mean_binary_entropy(probabilities, epsilon=1e-12):
    p = np.clip(np.asarray(probabilities, dtype=np.float64), epsilon, 1.0 - epsilon)
    if p.ndim != 2 or len(p) == 0:
        raise ValueError('Expected a nonempty (records, classes) probability array')
    entropy = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    return entropy.mean(axis=0)
