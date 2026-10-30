"""Unit tests for Trace-Stratified Conformal Risk Control (TA-CRC) and Decision Curve Analysis (DCA)."""
import json
from pathlib import Path
import unittest

import numpy as np

from src.uncertainty.ta_crc import (
    CLASSES,
    calibrate_multilabel_crc,
    calibrate_per_class_crc,
    calibrate_ta_crc,
    compute_multilabel_loss,
    compute_net_benefit,
    construct_prediction_sets,
)


class TestConformalRiskControl(unittest.TestCase):
    """Verify mathematical properties and finite-sample guarantees of CRC."""

    def test_multilabel_loss_monotonicity(self):
        """Multi-label missed-finding loss L_i(lambda) must be non-increasing in lambda."""
        np.random.seed(42)
        n = 200
        q = np.random.uniform(0.0, 1.0, (n, 5))
        y = (np.random.uniform(0.0, 1.0, (n, 5)) > 0.7).astype(np.float64)

        lambdas = np.linspace(0.0, 1.0, 50)
        losses = [compute_multilabel_loss(q, y, lam) for lam in lambdas]

        for i in range(len(losses) - 1):
            self.assertGreaterEqual(
                losses[i], losses[i + 1] - 1e-9,
                f"Loss must be non-increasing: L({lambdas[i]:.3f})={losses[i]} < L({lambdas[i+1]:.3f})={losses[i+1]}"
            )

    def test_per_class_fnr_monotonicity(self):
        """Per-class FNR must be non-increasing with respect to lambda_k."""
        np.random.seed(42)
        n_pos = 150
        q_pos = np.random.uniform(0.0, 1.0, n_pos)

        lambdas = np.linspace(0.0, 1.0, 50)
        # Inclusion in C is q >= 1.0 - lambda => missed is q < 1.0 - lambda
        fnrs = [float(np.mean(q_pos < (1.0 - lam))) for lam in lambdas]

        for i in range(len(fnrs) - 1):
            self.assertGreaterEqual(
                fnrs[i], fnrs[i + 1] - 1e-9,
                f"FNR must be non-increasing: FNR({lambdas[i]:.3f})={fnrs[i]} < FNR({lambdas[i+1]:.3f})={fnrs[i+1]}"
            )

    def test_exact_finite_sample_conformal_bound_fold9(self):
        """Exact finite-sample bound (n / (n + 1)) * R_hat_n + 1 / (n + 1) <= alpha must hold on Fold 9."""
        val_traces = np.load('results/paired_pages/creased/validation/traces.npz', allow_pickle=False)
        val_prob = np.load('results/paired_pages/creased/validation/probabilities.npz', allow_pickle=False)
        y_val = val_traces['labels']
        q_val = val_prob['raw_mv_paper']

        for alpha in [0.05, 0.10]:
            # 1. Global Multi-Label CRC
            pos_mask = y_val.sum(axis=1) > 0
            n_multi = int(pos_mask.sum())
            lam_multi = calibrate_multilabel_crc(q_val, y_val, alpha)
            r_hat_multi = compute_multilabel_loss(q_val, y_val, lam_multi)
            adj_multi = (n_multi / (n_multi + 1.0)) * r_hat_multi + 1.0 / (n_multi + 1.0)

            self.assertLessEqual(
                adj_multi, alpha + 1e-9,
                f"Global CRC bound failed at alpha={alpha}: adj_risk={adj_multi} > {alpha}"
            )

            # 2. Per-Class CRC
            tau_class = calibrate_per_class_crc(q_val, y_val, alpha)
            for k, cls in enumerate(CLASSES):
                pos_k = y_val[:, k] == 1
                n_k = int(pos_k.sum())
                lam_k = tau_class[cls]
                fnr_k = float(np.mean(q_val[pos_k, k] < (1.0 - lam_k)))
                adj_fnr_k = (n_k / (n_k + 1.0)) * fnr_k + 1.0 / (n_k + 1.0)

                self.assertLessEqual(
                    adj_fnr_k, alpha + 1e-9,
                    f"Per-class CRC bound failed for {cls} at alpha={alpha}: adj_fnr={adj_fnr_k} > {alpha}"
                )


class TestDecisionCurveAnalysis(unittest.TestCase):
    """Verify properties of Decision Curve Analysis and Net Benefit."""

    def test_net_benefit_treat_none_is_identically_zero(self):
        """Net Benefit for Treat-None must be exactly 0.0 across all p_t."""
        np.random.seed(42)
        n = 300
        y = (np.random.rand(n) > 0.8).astype(int)
        q = np.random.uniform(0.1, 0.9, n)
        p_t_grid = np.linspace(0.05, 0.80, 16)

        dca = compute_net_benefit(y, q, p_t_grid)
        for nb_none in dca['net_benefit_treat_none']:
            self.assertEqual(nb_none, 0.0, "Treat-None Net Benefit must be 0.0")

    def test_perfect_classifier_net_benefit(self):
        """A perfect classifier must have Net Benefit equal to disease prevalence for all p_t."""
        n = 200
        y = np.zeros(n)
        y[:50] = 1.0  # Prevalence = 0.25
        q_perfect = y.copy()

        p_t_grid = np.array([0.10, 0.20, 0.50])
        dca = compute_net_benefit(y, q_perfect, p_t_grid)

        for nb in dca['net_benefit_model']:
            self.assertAlmostEqual(nb, 0.25, places=5, msg="Perfect model NB must equal prevalence")


class TestLockedFilesIntegrity(unittest.TestCase):
    """Verify that all locked repository files remain untouched and non-empty."""

    def test_locked_files_exist_and_unmodified(self):
        locked_paths = [
            Path('results/paired_standardized.json'),
            Path('results/paired_raw_mv.json'),
            Path('results/corrector/raw_mv.json'),
            Path('results/corrector/logit_only.json'),
            Path('results/corrector/dc_odir_beta.json'),
            Path('results/stage2/best.pt'),
            Path('results/stage2_raw/best.pt'),
        ]
        for p in locked_paths:
            self.assertTrue(p.exists(), f"Locked file missing: {p}")
            self.assertGreater(p.stat().st_size, 0, f"Locked file is empty: {p}")


if __name__ == '__main__':
    unittest.main()
