"""Unit tests for DC-ODIR-Beta calibration and biophysical telemetry."""
import json
from pathlib import Path
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from src.corrector.dc_odir_beta import (
    CLASSES,
    DCODIRBetaCalibrator,
    DCODIRBetaModel,
    prepare_log_prob_features,
    train_single_model,
)
from src.corrector.physics_telemetry import extract_raw_telemetry, fit_precordial_pca


class TestPhysicsTelemetry(unittest.TestCase):
    """Verify physical properties of reference-free biophysical telemetry."""

    def test_einthoven_and_goldberger_near_zero_on_clean_synthetic(self):
        """Clean synthetic ECG with strict Kirchhoff & Wilson closure must have phi_1, phi_2 ~ 0."""
        # 12 leads: [I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6]
        P_3 = np.zeros((6, 3))
        P_3[0, 0] = 1.0
        P_3[1, 1] = 1.0
        P_3[2, 2] = 1.0

        t = np.linspace(0, 10, 1000)
        lead_I = 0.5 * np.sin(2 * np.pi * 1.0 * t)
        lead_III = 0.7 * np.sin(2 * np.pi * 1.0 * t)
        lead_II = lead_I + lead_III  # Exact Einthoven closure: II - (I + III) == 0

        # Goldberger relations: aVR = -(I + II)/2, aVL = (I - III)/2, aVF = (II + III)/2
        lead_aVR = -(lead_I + lead_II) / 2.0
        lead_aVL = (lead_I - lead_III) / 2.0
        lead_aVF = (lead_II + lead_III) / 2.0
        # Check sum: aVR + aVL + aVF = -(I + II)/2 + (I - III)/2 + (II + III)/2 = 0

        precordials = np.zeros((6, 1000))
        waveform = np.vstack([
            lead_I, lead_II, lead_III,
            lead_aVR, lead_aVL, lead_aVF,
            precordials,
        ])  # (12, 1000)

        phi = extract_raw_telemetry(waveform, P_3)
        phi_1_einthoven = phi[0]
        phi_2_goldberger = phi[1]

        self.assertLess(phi_1_einthoven, 1e-4, f"Einthoven residual should be ~0, got {phi_1_einthoven}")
        self.assertLess(phi_2_goldberger, 1e-4, f"Goldberger residual should be ~0, got {phi_2_goldberger}")

    def test_einthoven_violation_detects_corrupted_leads(self):
        """Corrupting Lead II must significantly increase phi_1."""
        P_3 = np.zeros((6, 3))
        t = np.linspace(0, 10, 1000)
        lead_I = 0.5 * np.sin(2 * np.pi * 1.0 * t)
        lead_III = 0.7 * np.sin(2 * np.pi * 1.0 * t)
        lead_II = lead_I + lead_III + 0.8 * np.random.randn(1000)  # corrupt lead II

        lead_aVR = -(lead_I + lead_II) / 2.0
        lead_aVL = (lead_I - lead_III) / 2.0
        lead_aVF = (lead_II + lead_III) / 2.0
        waveform = np.vstack([
            lead_I, lead_II, lead_III,
            lead_aVR, lead_aVL, lead_aVF,
            np.zeros((6, 1000)),
        ])

        phi = extract_raw_telemetry(waveform, P_3)
        self.assertGreater(phi[0], 0.2, f"Corrupted Einthoven residual must be elevated, got {phi[0]}")


class TestDCODIRBetaModel(unittest.TestCase):
    """Verify architectural properties and limiting behavior of DC-ODIR-Beta."""

    def test_reduces_to_logit_linear_under_high_regularization(self):
        """Under very high lambda penalties, W->0, r->0, v->0, a->1, reducing to logit-linear."""
        torch.manual_seed(42)
        np.random.seed(42)

        N = 500
        K = 5
        D = 8
        P = np.random.uniform(0.05, 0.95, size=(N, K))
        Phi = np.random.randn(N, D)
        Y = (P > 0.5).astype(np.float64)

        u_plus, u_minus = prepare_log_prob_features(P)

        # Train with strong regularization
        model = train_single_model(
            u_plus, u_minus, Phi, Y,
            lambda_odir=100.0,
            lambda_phi=100.0,
            lambda_beta=100.0,
            max_iter=50,
        )

        params = model.get_parameter_dict()

        # Check that W, r, v are close to zero
        norm_W = np.linalg.norm(params['W'])
        norm_r = np.linalg.norm(params['r'])
        norm_v = np.linalg.norm(params['v'])
        mean_a_plus = np.mean(params['a_plus'])
        mean_a_minus = np.mean(params['a_minus'])

        self.assertLess(norm_W, 0.05, f"W should be ~0 under high regularization, got norm {norm_W}")
        self.assertLess(norm_r, 0.05, f"r should be ~0 under high regularization, got norm {norm_r}")
        self.assertLess(norm_v, 0.05, f"v should be ~0 under high regularization, got norm {norm_v}")
        self.assertAlmostEqual(mean_a_plus, 1.0, places=1, msg=f"a+ should be ~1.0, got {mean_a_plus}")
        self.assertAlmostEqual(mean_a_minus, 1.0, places=1, msg=f"a- should be ~1.0, got {mean_a_minus}")

    def test_fold10_guardrails_no_leakage(self):
        """Calibrator inference must accept only (p, phi) and never require clean signal or y."""
        model = DCODIRBetaModel(num_classes=5, phi_dim=8)
        calibrator = DCODIRBetaCalibrator(model=model)

        test_p = np.random.uniform(0.1, 0.9, size=(10, 5))
        test_phi = np.random.randn(10, 8)

        # Inference requires only (p, phi)
        calibrated_p = calibrator.predict_proba(test_p, test_phi)
        self.assertEqual(calibrated_p.shape, (10, 5))
        self.assertTrue(np.all((calibrated_p >= 0.0) & (calibrated_p <= 1.0)))

    def test_strictly_improves_or_maintains_validation_loss(self):
        """DC-ODIR-Beta calibration must achieve lower or equal cross-entropy than uncalibrated inputs."""
        torch.manual_seed(42)
        np.random.seed(42)

        N = 400
        K = 5
        D = 8
        P = np.random.uniform(0.1, 0.9, size=(N, K))
        Phi = np.random.randn(N, D)
        # Shift targets away from P to simulate miscalibration
        Y = (P ** 1.8 > 0.4).astype(np.float64)

        u_plus, u_minus = prepare_log_prob_features(P)
        uncal_bce = F.binary_cross_entropy(torch.tensor(P, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)).item()

        model = train_single_model(
            u_plus, u_minus, Phi, Y,
            lambda_odir=0.1,
            lambda_phi=0.05,
            lambda_beta=0.1,
            max_iter=50,
        )

        with torch.no_grad():
            eta = model(
                torch.tensor(u_plus, dtype=torch.float32),
                torch.tensor(u_minus, dtype=torch.float32),
                torch.tensor(Phi, dtype=torch.float32),
            )
            cal_bce = F.binary_cross_entropy_with_logits(eta, torch.tensor(Y, dtype=torch.float32)).item()

        self.assertLessEqual(cal_bce, uncal_bce, f"Calibrated BCE ({cal_bce}) must be <= uncalibrated BCE ({uncal_bce})")


if __name__ == '__main__':
    unittest.main()

