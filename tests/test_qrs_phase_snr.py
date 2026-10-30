"""Unit tests for QRS phase SNR, biophysical metrics, and phase partitioning."""
import unittest
import numpy as np

from src.mechanism.qrs_phase_snr import (
    build_phase_masks,
    compute_masked_snr,
    compute_qrs_slew_and_clipping,
    compute_sokolow_and_cornell,
    detect_r_peaks,
)


class TestQRSPhaseSNR(unittest.TestCase):
    def setUp(self):
        # Deterministic synthetic R-peaks
        self.r_peaks = [100, 250, 420, 590, 750, 910]
        self.n_samples = 1000

    def test_phase_masks_no_overlap(self):
        """Verify that phase masks have zero pairwise overlap."""
        qrs_mask, st_t_mask, tp_mask = build_phase_masks(
            self.r_peaks, n_samples=self.n_samples, strict_tp=True
        )
        self.assertEqual(len(qrs_mask), self.n_samples)
        self.assertEqual(len(st_t_mask), self.n_samples)
        self.assertEqual(len(tp_mask), self.n_samples)

        # Mutually exclusive
        self.assertFalse(
            (qrs_mask & st_t_mask).any(),
            "QRS and ST-T masks must not overlap"
        )
        self.assertFalse(
            (qrs_mask & tp_mask).any(),
            "QRS and TP masks must not overlap"
        )
        self.assertFalse(
            (st_t_mask & tp_mask).any(),
            "ST-T and TP masks must not overlap"
        )

    def test_phase_masks_full_partition_without_overlap(self):
        """Verify that phase masks partition [0, 1000) completely without overlap."""
        qrs_mask, st_t_mask, tp_mask = build_phase_masks(
            self.r_peaks, n_samples=self.n_samples, full_partition=True
        )

        # 1. No overlap
        self.assertFalse((qrs_mask & st_t_mask).any(), "QRS and ST-T overlap")
        self.assertFalse((qrs_mask & tp_mask).any(), "QRS and TP overlap")
        self.assertFalse((st_t_mask & tp_mask).any(), "ST-T and TP overlap")

        # 2. Complete partition (exact sum = 1000, union covers every index)
        union_mask = qrs_mask | st_t_mask | tp_mask
        self.assertTrue(union_mask.all(), "Phase masks must cover all indices [0, 1000)")
        self.assertEqual(
            qrs_mask.sum() + st_t_mask.sum() + tp_mask.sum(),
            self.n_samples,
            "Total mask count must equal n_samples without gap or double-counting"
        )

    def test_identical_signals_yield_infinite_and_capped_snr(self):
        """Identical signals have zero noise power, yielding inf or capped high SNR."""
        t = np.linspace(0, 10, self.n_samples)
        x = np.sin(2 * np.pi * 1.5 * t) + 0.5 * np.cos(2 * np.pi * 3.0 * t)

        # Infinite SNR when uncapped
        snr_inf, p_sig, p_noise = compute_masked_snr(x, x, cap_infinite=False)
        self.assertTrue(np.isinf(snr_inf))
        self.assertGreater(snr_inf, 0.0)
        self.assertEqual(p_noise, 0.0)
        self.assertGreater(p_sig, 0.0)

        # Capped high SNR (100 dB) when cap_infinite=True
        snr_capped, _, _ = compute_masked_snr(x, x, cap_infinite=True, cap_value=100.0)
        self.assertEqual(snr_capped, 100.0)

    def test_identical_signals_yield_unit_slew_rate_correlation(self):
        """Identical signals yield exactly 1.0 slew-rate correlation and 1.0 clipping."""
        rng = np.random.default_rng(42)
        ecg_12lead = rng.normal(0, 1, size=(12, self.n_samples))

        qrs_mask = np.zeros(self.n_samples, dtype=bool)
        qrs_mask[95:108] = True
        qrs_mask[245:258] = True

        metrics = compute_qrs_slew_and_clipping(ecg_12lead, ecg_12lead, qrs_mask)

        self.assertAlmostEqual(metrics['slew_correlation_global'], 1.0, places=5)
        self.assertAlmostEqual(metrics['slew_correlation_mean_leads'], 1.0, places=5)
        self.assertAlmostEqual(metrics['clipping_ratio_global'], 1.0, places=5)
        self.assertAlmostEqual(metrics['clipping_ratio_mean_leads'], 1.0, places=5)

    def test_r_peak_detection_synthetic(self):
        """Test R-peak detection finds impulses in synthetic signal."""
        sig = np.zeros((12, self.n_samples))
        peaks_true = [150, 300, 450, 600, 750, 900]
        for p in peaks_true:
            sig[1, p] = 2.5  # Lead II sharp R-peak
            sig[1, p - 1] = 0.5
            sig[1, p + 1] = 0.5

        detected = detect_r_peaks(sig, sampling_rate=100, lead_idx=1)
        self.assertEqual(len(detected), len(peaks_true))
        for true_p, det_p in zip(peaks_true, detected):
            self.assertLessEqual(abs(true_p - det_p), 2)

    def test_voltage_criteria_sokolow_and_cornell(self):
        """Verify Sokolow-Lyon and Cornell voltage criteria calculation."""
        ecg = np.zeros((12, self.n_samples))
        # Lead order: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
        # AVL = 4, V1 = 6, V3 = 8, V5 = 10, V6 = 11
        ecg[6, 100] = -1.2   # V1 S-wave depth = 1.2 mV
        ecg[10, 100] = 2.0   # V5 R-wave height = 2.0 mV
        ecg[11, 100] = 1.5   # V6 R-wave height = 1.5 mV
        ecg[4, 100] = 1.1    # AVL R-wave height = 1.1 mV
        ecg[8, 100] = -1.4   # V3 S-wave depth = 1.4 mV

        res = compute_sokolow_and_cornell(ecg)
        # Sokolow-Lyon = S_V1 (1.2) + max(R_V5, R_V6) (2.0) = 3.2
        self.assertAlmostEqual(res['sokolow_lyon_mv'], 3.2, places=4)
        # Cornell = R_AVL (1.1) + S_V3 (1.4) = 2.5
        self.assertAlmostEqual(res['cornell_mv'], 2.5, places=4)


if __name__ == '__main__':
    unittest.main()

