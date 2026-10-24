"""PhysioNet Challenge 2024 lead-wise SNR, including its shift caps.

The negative time-shift slice is copied from evaluation-2024 helper_code.py.
A perfect copy has zero noise power and an undefined SNR.
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve


def convert_signal(x, num_quant_levels, min_amp, max_amp, max_t):
    idx = np.isfinite(x)
    t = np.arange(1, np.size(x) + 1)[idx]
    y = np.round((num_quant_levels - 1) * (x[idx] - min_amp) / (max_amp - min_amp) + 1).astype(int)
    y = np.clip(y, 1, num_quant_levels)
    grid = np.zeros((num_quant_levels, max_t))
    grid[y - 1, t - 1] = 1
    return grid


def fft_correlate(reference, estimate):
    flipped = np.flip(np.flip(estimate, axis=0), axis=1)
    return fftconvolve(reference, flipped, mode='full')


def align_signals(reference, estimate, num_quant_levels, smooth=True, sigma=0.5):
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    if np.any(np.isfinite(reference)) and np.any(np.isfinite(estimate)):
        min_amp = min(np.nanmin(reference), np.nanmin(estimate))
        max_amp = max(np.nanmax(reference), np.nanmax(estimate))
        max_t = max(np.size(reference), np.size(estimate))
        reference_grid = convert_signal(reference, num_quant_levels, min_amp, max_amp, max_t)
        estimate_grid = convert_signal(estimate, num_quant_levels, min_amp, max_amp, max_t)
        if smooth:
            reference_grid = gaussian_filter(reference_grid, sigma)
            estimate_grid = gaussian_filter(estimate_grid, sigma)
        cross = fft_correlate(reference_grid, estimate_grid)
        cross_index = np.unravel_index(np.argmax(cross), cross.shape)
        auto = fft_correlate(reference_grid, reference_grid)
        auto_index = np.unravel_index(np.argmax(auto), auto.shape)
        offset_hz = auto_index[1] - cross_index[1]
        offset_vt = (auto_index[0] - cross_index[0]) / (num_quant_levels - 1) * (max_amp - min_amp)
        if offset_hz < 0:
            shifted = np.concatenate((np.nan * np.ones(-offset_hz), estimate[:-offset_hz]))
        else:
            shifted = np.concatenate((estimate[offset_hz:], np.nan * np.ones(offset_hz)))
        shifted = shifted - offset_vt
    else:
        shifted = estimate.copy()
        offset_hz = 0
        offset_vt = 0.0
    return shifted, int(offset_hz), float(offset_vt)


def compute_snr(reference, estimate, keep_nans=True):
    reference = np.asarray(reference, dtype=np.float64).copy()
    estimate = np.asarray(estimate, dtype=np.float64).copy()
    if reference.ndim != 1 or estimate.ndim != 1:
        raise ValueError('SNR expects one lead')
    if estimate.size < reference.size:
        estimate = np.concatenate((estimate, np.nan * np.ones(reference.size - estimate.size)))
    elif estimate.size > reference.size:
        reference = np.concatenate((reference, np.nan * np.ones(estimate.size - reference.size)))
    finite_reference = np.isfinite(reference)
    finite_estimate = np.isfinite(estimate)
    if keep_nans:
        idx = np.logical_and(finite_reference, finite_estimate)
    else:
        estimate[~finite_estimate] = 0
        idx = finite_reference
    noise = reference[idx] - estimate[idx]
    signal_power = np.nanmean(reference[idx] ** 2) if np.any(np.isfinite(reference[idx])) else np.nan
    noise_power = np.nanmean(noise ** 2) if np.any(np.isfinite(noise)) else np.nan
    if signal_power > 0 and noise_power > 0:
        snr = 10 * np.log10(signal_power / noise_power)
    elif np.sum(idx) == 0 and np.sum(finite_reference) > 0:
        snr = 0.0
    else:
        snr = np.nan
    if keep_nans and np.sum(finite_reference):
        snr = snr * (np.sum(idx) / np.sum(finite_reference))
    return float(snr), float(signal_power), float(noise_power)


def leadwise_snr(reference, estimate, sampling_rate=100, num_quant_levels=256):
    """Score each lead in millivolts. Shifts outside ±0.5 s or ±1 mV are rejected."""
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    if reference.shape != estimate.shape or reference.ndim != 2:
        raise ValueError('Expected matching (leads, samples) arrays in mV')
    max_shift = int(np.round(0.5 * sampling_rate))
    rows = []
    for lead_reference, lead_estimate in zip(reference, estimate):
        aligned, shift, vertical = align_signals(lead_reference, lead_estimate, num_quant_levels)
        if abs(shift) <= max_shift and abs(vertical) <= 1.0:
            used, accepted_shift, accepted_vertical = aligned, shift, vertical
        else:
            used, accepted_shift, accepted_vertical = lead_estimate, 0, 0.0
        snr, signal_power, noise_power = compute_snr(lead_reference, used)
        rows.append({
            'snr_db': snr, 'shift_samples': accepted_shift, 'shift_mv': accepted_vertical,
            'p_signal': signal_power, 'p_noise': noise_power,
        })
    finite = [row['snr_db'] for row in rows if np.isfinite(row['snr_db'])]
    return {
        'leads': rows,
        'mean_snr_db': float(np.mean(finite)) if finite else float('nan'),
        'max_shift_samples': max_shift, 'max_shift_mv': 1.0,
    }
