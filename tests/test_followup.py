"""Guards for the trace-only corrector, the oracle scale check, and the scan gate."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from src.corrector import load_coefficients, trace_versus_logit
from src.sensitivity import agrees_with_locked
from src.intervals import bootstrap_arm_contrast
from src.mechanism import reliability_bins, snr_by_superclass
from src.pages import affine_match, assert_matches_locked
from src.protocol import load_protocol
from src.real_scan import score


class ProtocolRejection(unittest.TestCase):
    def _copy(self, directory):
        protocol = yaml.safe_load(Path('configs/paired.yaml').read_text())
        path = Path(directory) / 'paired.yaml'
        path.write_text(yaml.safe_dump(protocol), encoding='utf-8')
        return protocol, path

    def test_changed_primary_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            protocol, path = self._copy(directory)
            protocol['primary_outcome'] = 'accuracy'
            path.write_text(yaml.safe_dump(protocol), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Primary outcome'):
                load_protocol(path)

    def test_fit_on_the_test_split_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            protocol, path = self._copy(directory)
            protocol['corrector']['fit_split'] = 'test'
            path.write_text(yaml.safe_dump(protocol), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'validation'):
                load_protocol(path)

    def test_clean_waveform_feature_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            protocol, path = self._copy(directory)
            protocol['corrector']['features'] = ['logit', 'clean_scale_ratio', 'ink_gap_fraction']
            path.write_text(yaml.safe_dump(protocol), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'clean waveform'):
                load_protocol(path)


class LockedResult(unittest.TestCase):
    def test_mismatch_does_not_change_the_locked_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'paired_raw_mv.json'
            payload = {'delta_classwise_ece': {'CD': 0.142}}
            path.write_text(json.dumps(payload), encoding='utf-8')
            before = path.read_bytes()
            with self.assertRaisesRegex(RuntimeError, 'locked JSON was not modified'):
                assert_matches_locked(
                    {'arm': 'raw_mv', 'delta_classwise_ece': {'CD': 9.0}},
                    json.loads(path.read_text()))
            self.assertEqual(path.read_bytes(), before)


class OracleScale(unittest.TestCase):
    def test_affine_match_restores_a_scaled_lead(self):
        clean = np.zeros((1, 2, 8), dtype=np.float32)
        clean[0, 0] = np.linspace(-1, 1, 8)
        clean[0, 1] = np.linspace(0.2, 0.8, 8)
        traced = clean * 2 + 3
        matched = affine_match(traced, clean)
        self.assertTrue(np.allclose(matched, clean, atol=1e-5))

    def test_reliability_bin_counts_cover_the_records(self):
        labels = np.array([[1], [0], [1], [0]], dtype=np.float64)
        probability = np.array([[0.1], [0.2], [0.9], [0.8]])
        diagram = reliability_bins(labels, probability, ['CD'], bins=2)
        self.assertEqual(sum(row['count'] for row in diagram['CD']), 4)


class CorrectorGuard(unittest.TestCase):
    def test_test_split_coefficients_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'raw_mv.json'
            path.write_text(json.dumps({
                'fit_split': 'test',
                'features': ['logit', 'traced_lead_std_mv', 'ink_gap_fraction'],
            }), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'validation'):
                load_coefficients(path)

    def test_clean_feature_in_a_coefficient_file_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'raw_mv.json'
            path.write_text(json.dumps({
                'fit_split': 'validation',
                'features': ['logit', 'clean_scale_ratio', 'ink_gap_fraction'],
            }), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'clean waveform'):
                load_coefficients(path)


class ValidityChecks(unittest.TestCase):
    def test_identical_arms_have_a_zero_contrast(self):
        labels = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        clean = np.array([[0.9, 0.2], [0.2, 0.8], [0.8, 0.1], [0.1, 0.7]])
        paper = np.array([[0.6, 0.4], [0.4, 0.6], [0.55, 0.3], [0.3, 0.5]])
        patients = np.array([1, 1, 2, 2])
        contrast = bootstrap_arm_contrast(
            labels, clean, paper, clean, paper, ['CD', 'MI'], patients, resamples=20, seed=1)
        self.assertEqual(contrast['point']['CD'], 0.0)
        self.assertEqual(contrast['low']['CD'], 0.0)
        self.assertEqual(contrast['high']['CD'], 0.0)

    def test_snr_is_split_by_superclass(self):
        labels = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        snr = np.array([10.0, 20.0, 30.0])
        report = snr_by_superclass(labels, snr, ['CD', 'MI'])
        self.assertEqual(report['CD']['with_label_mean_snr_db'], 20.0)
        self.assertEqual(report['CD']['without_label_records'], 1)
        self.assertEqual(report['MI']['with_label_records'], 2)

    def test_trace_features_miss_when_logit_only_is_closer(self):
        status = trace_versus_logit(
            {'CD': 0.10}, {'CD': 0.02}, 'CD', trace_auroc=0.90, uncorrected_auroc=0.90, limit=0.01)
        self.assertEqual(status, 'miss')

    def test_width64_agreement_requires_the_locked_sign_and_an_interval_off_zero(self):
        positive = {'point': {'CD': 0.10}, 'low': {'CD': 0.02}, 'high': {'CD': 0.20}}
        covering = {'point': {'CD': 0.10}, 'low': {'CD': -0.01}, 'high': {'CD': 0.20}}
        opposite = {'point': {'CD': -0.10}, 'low': {'CD': -0.20}, 'high': {'CD': -0.02}}
        self.assertTrue(agrees_with_locked(positive, 0.16, 'CD'))
        self.assertFalse(agrees_with_locked(covering, 0.16, 'CD'))
        self.assertFalse(agrees_with_locked(opposite, 0.16, 'CD'))

    def test_trace_features_earn_their_place_against_logit_only(self):
        status = trace_versus_logit(
            {'CD': 0.01}, {'CD': 0.08}, 'CD', trace_auroc=0.90, uncorrected_auroc=0.90, limit=0.01)
        self.assertEqual(status, 'earned')


class ScanGate(unittest.TestCase):
    def test_missing_gate_writes_no_fold10_file(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            result = score(
                load_protocol(),
                gate_path=destination / 'missing-gate.json',
                output_dir=destination)
            self.assertEqual(result['status'], 'blocked')
            self.assertEqual(result['reason'], 'gate file missing')
            self.assertFalse((destination / 'fold10_raw_mv.json').exists())


if __name__ == '__main__':
    unittest.main()
