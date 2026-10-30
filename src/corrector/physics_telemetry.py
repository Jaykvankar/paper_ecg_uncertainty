"""Reference-free biophysical and physics-closure telemetry extractor phi(X_hat).

Extracts an 8-dimensional instability vector capturing:
1. Einthoven limb closure residual (Kirchhoff frontal loop)
2. Goldberger augmented limb closure residual (Wilson central terminal)
3. Precordial spatial subspace residual (Burger-van Milaan 3D projection)
4. Column-to-column vertical jump / crease spike rate
5. High-amplitude peak clipping / saturation index
6. Out-of-band high-frequency energy ratio (40-50 Hz / 0.5-40 Hz)
7. Reference-free TP-baseline to QRS slew ratio
8. Sokolow-Lyon & Cornell voltage excursion proxy
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from src.baseline_data import baseline_frame
from src.data import LEADS, load_signal
from src.mechanism.qrs_phase_snr import (
    build_phase_masks,
    compute_sokolow_and_cornell,
    detect_r_peaks,
)

PCA_PATH = Path('results/corrector/precordial_pca_p3.npz')
NORMALIZER_PATH = Path('results/corrector/telemetry_normalizer.json')

# Lead index mappings
LEAD_I = LEADS.index('I')
LEAD_II = LEADS.index('II')
LEAD_III = LEADS.index('III')
LEAD_AVR = LEADS.index('AVR')
LEAD_AVL = LEADS.index('AVL')
LEAD_AVF = LEADS.index('AVF')
PRECORDIAL_INDICES = [LEADS.index(f'V{i}') for i in range(1, 7)]


def fit_precordial_pca(
    data_dir: str = 'data/ptb-xl-1.0.3',
    max_train_records: int = 1000,
    save_path: Optional[Path] = PCA_PATH,
) -> np.ndarray:
    """Fit a 3-component PCA (P_3) on clean precordial leads V1-V6 from Folds 1-8."""
    if save_path and save_path.exists():
        saved = np.load(save_path)
        return saved['P_3']

    config = {
        'data_dir': data_dir,
        'results_dir': 'results',
        'classes': ['NORM', 'MI', 'STTC', 'CD', 'HYP'],
        'splits': {'train': list(range(1, 9)), 'validation': [9], 'test': [10]},
    }
    bframe = baseline_frame(config)
    train_files = bframe[bframe.split == 'train']['filename_lr'].tolist()
    root = Path(data_dir)

    selected_files = train_files[:max_train_records]
    samples = []
    for f in selected_files:
        sig = load_signal(root, f)  # (12, 1000)
        samples.append(sig[PRECORDIAL_INDICES])  # (6, 1000)

    # Concatenate across time samples: shape (6, N * 1000)
    X = np.concatenate(samples, axis=1).astype(np.float64)
    cov = np.cov(X)
    vals, vecs = np.linalg.eigh(cov)
    # Sort descending
    order = np.argsort(vals)[::-1]
    P_3 = vecs[:, order[:3]]  # shape (6, 3)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_path, P_3=P_3)

    return P_3


def extract_raw_telemetry(
    waveform_12lead: np.ndarray,
    P_3: np.ndarray,
    sampling_rate: int = 100,
) -> np.ndarray:
    """Extract unstandardized 8-dimensional phi(X_hat) from a 12-lead ECG (12, 1000)."""
    X = np.asarray(waveform_12lead, dtype=np.float64)
    if X.shape[0] != 12 or X.ndim != 2:
        raise ValueError(f"Expected shape (12, N), got {X.shape}")

    n_samples = X.shape[1]

    # phi_1: Einthoven Limb Closure Residual
    # Lead II - (Lead I + Lead III) == 0 in clean physiology
    diff_einthoven = X[LEAD_II] - (X[LEAD_I] + X[LEAD_III])
    norm_ii = float(np.linalg.norm(X[LEAD_II]))
    phi_1 = float(np.linalg.norm(diff_einthoven) / (norm_ii + 1e-6))

    # phi_2: Goldberger Augmented Limb Residual
    # aVR + aVL + aVF == 0 in clean physiology
    sum_goldberger = X[LEAD_AVR] + X[LEAD_AVL] + X[LEAD_AVF]
    norm_augmented_mean = float(
        np.linalg.norm(X[LEAD_AVR]) + np.linalg.norm(X[LEAD_AVL]) + np.linalg.norm(X[LEAD_AVF])
    ) / 3.0
    phi_2 = float(np.linalg.norm(sum_goldberger) / (norm_augmented_mean + 1e-6))

    # phi_3: Precordial Spatial Subspace Residual
    # Projection onto 3-component PCA subspace: P_3 @ P_3.T
    V1_V6 = X[PRECORDIAL_INDICES]  # (6, 1000)
    proj = P_3 @ (P_3.T @ V1_V6)  # (6, 1000)
    residual = V1_V6 - proj
    norm_precordial = float(np.linalg.norm(V1_V6, 'fro'))
    phi_3 = float(np.linalg.norm(residual, 'fro') / (norm_precordial + 1e-6))

    # First difference along time
    diff_t = np.diff(X, axis=-1)  # (12, n_samples - 1)

    # phi_4: Column-to-Column Vertical Jump / Crease Spike Rate
    # |X_l(t) - X_l(t-1)| > 0.35 mV AND (X_l(t) - X_l(t-1)) * (X_l(t+1) - X_l(t)) < 0
    if n_samples > 2:
        d1 = diff_t[:, :-1]
        d2 = diff_t[:, 1:]
        spikes = (np.abs(d1) > 0.35) & (d1 * d2 < 0)
        phi_4 = float(np.sum(spikes) / (12 * (n_samples - 2)))
    else:
        phi_4 = 0.0

    # phi_5: High-Amplitude Peak Clipping / Saturation Index
    # Top 5% absolute amplitude where local first derivative is near-flat (|diff| < 0.005 mV)
    clip_fractions = []
    for l in range(12):
        lead = X[l]
        abs_lead = np.abs(lead)
        t95 = float(np.percentile(abs_lead, 95))
        top_mask = abs_lead[1:] >= t95
        if np.any(top_mask):
            flat_samples = np.abs(diff_t[l, top_mask]) < 0.005
            clip_fractions.append(float(np.mean(flat_samples)))
        else:
            clip_fractions.append(0.0)
    phi_5 = float(np.mean(clip_fractions))

    # phi_6: Out-of-Band High-Frequency Energy Ratio
    # (40 Hz, 50 Hz] power divided by [0.5 Hz, 40 Hz] power
    rfft = np.fft.rfft(X, axis=-1)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sampling_rate)
    band_signal = (freqs >= 0.5) & (freqs <= 40.0)
    band_oob = (freqs > 40.0) & (freqs <= 50.0)
    pow_signal = np.sum(np.abs(rfft[:, band_signal]) ** 2, axis=-1)
    pow_oob = np.sum(np.abs(rfft[:, band_oob]) ** 2, axis=-1)
    phi_6 = float(np.mean(pow_oob / (pow_signal + 1e-6)))

    # phi_7: Reference-Free TP-Baseline to QRS Slew Ratio
    # Detect R-peaks on X_hat itself, partition into QRS and TP baseline
    peaks = detect_r_peaks(X, sampling_rate=sampling_rate)
    qrs_mask, _, tp_mask = build_phase_masks(peaks, n_samples=n_samples, strict_tp=True)
    d_qrs = diff_t[:, qrs_mask[1:]] if np.any(qrs_mask[1:]) else diff_t
    d_tp = diff_t[:, tp_mask[1:]] if np.any(tp_mask[1:]) else diff_t
    rms_qrs = float(np.sqrt(np.mean(d_qrs ** 2)))
    rms_tp = float(np.sqrt(np.mean(d_tp ** 2)))
    phi_7 = float(rms_tp / (rms_qrs + 1e-6))

    # phi_8: Sokolow-Lyon & Cornell Voltage Excursion Proxy
    # [Sokolow_Lyon(X_hat) + Cornell(X_hat)] / 2.0
    volt = compute_sokolow_and_cornell(X)
    phi_8 = float((volt['sokolow_lyon_mv'] + volt['cornell_mv']) / 2.0)

    return np.array([phi_1, phi_2, phi_3, phi_4, phi_5, phi_6, phi_7, phi_8], dtype=np.float64)


def fit_telemetry_normalizer(
    val_traces_path: str = 'results/paired_pages/creased/validation/traces.npz',
    save_path: Optional[Path] = NORMALIZER_PATH,
    P_3: Optional[np.ndarray] = None,
) -> Dict[str, Union[List[float], int]]:
    """Fit feature normalizer on pooled clean and traced Fold-9 records (n = 4,292)."""
    if save_path and save_path.exists():
        return json.loads(save_path.read_text(encoding='utf-8'))

    if P_3 is None:
        P_3 = fit_precordial_pca()

    val_data = np.load(val_traces_path, allow_pickle=False)
    clean_val = val_data['clean_mv']  # (2146, 12, 1000)
    traced_val = val_data['traced_mv']  # (2146, 12, 1000)

    # Pool clean and traced
    pooled = np.concatenate([clean_val, traced_val], axis=0)  # (4292, 12, 1000)
    n_total = len(pooled)

    print(f"Extracting telemetry features on Fold-9 pooled records (N = {n_total})...")
    features = []
    for i in range(n_total):
        phi_raw = extract_raw_telemetry(pooled[i], P_3)
        features.append(phi_raw)

    feat_matrix = np.array(features, dtype=np.float64)
    mean_vec = np.mean(feat_matrix, axis=0)
    std_vec = np.maximum(np.std(feat_matrix, axis=0), 1e-6)

    normalizer = {
        'fit_split': 'validation_fold9_pooled',
        'n_samples': int(n_total),
        'mean': [float(x) for x in mean_vec],
        'std': [float(x) for x in std_vec],
        'feature_names': [
            'phi_1_einthoven_residual',
            'phi_2_goldberger_residual',
            'phi_3_precordial_subspace_residual',
            'phi_4_vertical_spike_rate',
            'phi_5_peak_clipping_index',
            'phi_6_out_of_band_hf_ratio',
            'phi_7_tp_to_qrs_slew_ratio',
            'phi_8_voltage_excursion_proxy',
        ],
    }

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(normalizer, indent=2), encoding='utf-8')

    return normalizer


class PhysicsTelemetryExtractor:
    """Pre-fitted telemetry extractor applying Fold 1-8 PCA and Fold 9 standardization."""

    def __init__(
        self,
        pca_path: Path = PCA_PATH,
        normalizer_path: Path = NORMALIZER_PATH,
    ):
        self.P_3 = fit_precordial_pca(save_path=pca_path)
        self.normalizer = fit_telemetry_normalizer(save_path=normalizer_path, P_3=self.P_3)
        self.mean = np.array(self.normalizer['mean'], dtype=np.float64)
        self.std = np.array(self.normalizer['std'], dtype=np.float64)

    def extract(self, waveform_12lead: np.ndarray, standardize: bool = True) -> np.ndarray:
        raw_phi = extract_raw_telemetry(waveform_12lead, self.P_3)
        if standardize:
            return (raw_phi - self.mean) / self.std
        return raw_phi

    def extract_batch(self, waveforms: np.ndarray, standardize: bool = True) -> np.ndarray:
        raw_list = [extract_raw_telemetry(waveforms[i], self.P_3) for i in range(len(waveforms))]
        raw_matrix = np.array(raw_list, dtype=np.float64)
        if standardize:
            return (raw_matrix - self.mean) / self.std
        return raw_matrix
