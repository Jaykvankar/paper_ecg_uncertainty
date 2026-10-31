import json
from pathlib import Path
import unittest

import numpy as np

from src.baseline_data import prepare_signal
from src.digitize import image_to_sequence, lead_correlation, recover_leads
from src.intervals import bootstrap_classwise_delta_ece
from src.protocol import load_protocol
from src.snr import leadwise_snr
from src import train
from src.uncertainty import mean_binary_entropy


class ProtocolChecks(unittest.TestCase):
    def test_entropy_of_one_half_is_ln2(self):
        entropy = mean_binary_entropy(np.full((4, 2), 0.5))
        self.assertTrue(np.allclose(entropy, np.log(2)))

    def test_protocol_lock(self):
        protocol = load_protocol()
        self.assertEqual(protocol['primary_outcome'], 'delta_classwise_ece')
        self.assertEqual(protocol['primary_level'], 'creased')
        self.assertFalse(protocol['diagnosis_as_input'])
        self.assertTrue(protocol['resolved']['standardized']['standardize'])
        self.assertFalse(protocol['resolved']['raw_mv']['standardize'])
        self.assertEqual(protocol['resolved']['standardized']['seed'], 42)

    def test_raw_millivolts_keep_voltage(self):
        signal = np.zeros((12, 1000), dtype=np.float32)
        signal[:, 0] = 2
        raw = prepare_signal(signal, standardize=False)
        self.assertEqual(float(raw[0, 0]), 2.0)
        scaled = prepare_signal(signal, standardize=True)
        self.assertLess(abs(float(scaled[0].mean())), 1e-5)

    def test_cuda_config_stops_without_a_gpu(self):
        from unittest.mock import patch
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'NVIDIA'):
                train.resolve_device('cuda', smoke=False)
        self.assertEqual(train.resolve_device('cuda', smoke=True).type, 'cpu')
        self.assertEqual(train.resolve_device('cpu', smoke=False).type, 'cpu')

    def test_image_to_sequence_matches_matlab_ends(self):
        mask = np.zeros((6, 2), dtype=np.uint8)
        mask[5, 0] = 255
        mask[0, 1] = 255
        values = image_to_sequence(mask)
        self.assertEqual(values[0], 0)
        self.assertEqual(values[1], 5)

    def test_drawn_trace_correlates(self):
        image = np.full((90, 220, 3), 255, dtype=np.uint8)
        xs = np.arange(30, 190)
        center = (50 + 12 * np.sin(xs / 7.0)).astype(int)
        for x, y in zip(xs, center):
            image[y - 1:y + 2, x] = 0
        box = {'0': [int(center.min()) - 2, 28], '1': [int(center.min()) - 2, 191],
               '2': [int(center.max()) + 2, 191], '3': [int(center.max()) + 2, 28]}
        leads = []
        for name in ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']:
            leads.append({'lead_name': name, 'start_sample': 0, 'end_sample': 1000,
                          'lead_bounding_box': box, 'plotted_pixels': [[0, 0]]})
        recovered = recover_leads(image, {'y_grid': 25, 'leads': leads})
        expected_rows = 50 + 12 * np.sin(np.linspace(30, 189, 1000) / 7.0)
        expected = expected_rows.max() - expected_rows
        score = lead_correlation(np.tile(expected, (12, 1)), recovered)['I']
        self.assertGreater(score, 0.9)

    def test_delayed_copy_keeps_a_high_snr(self):
        time = np.linspace(0, 1, 800, endpoint=False)
        clean = np.sin(2 * np.pi * 3 * time) + 0.25 * np.sin(2 * np.pi * 13 * time)
        noise = np.random.default_rng(4).normal(0, 1e-3, clean.shape)
        delayed = np.concatenate([noise[:10], clean[:-10]]) + noise
        reference = np.tile(clean, (2, 1))
        estimate = np.tile(delayed, (2, 1))
        scored = leadwise_snr(reference, estimate, sampling_rate=100)
        self.assertEqual(scored['leads'][0]['shift_samples'], 10)
        self.assertGreater(scored['mean_snr_db'], 40)
        perfect = leadwise_snr(reference, reference, sampling_rate=100)
        self.assertTrue(np.isnan(perfect['leads'][0]['snr_db']))

    def test_large_voltage_shift_is_not_applied(self):
        time = np.linspace(0, 1, 400, endpoint=False)
        clean = np.sin(2 * np.pi * 2 * time)
        shifted = np.tile(clean + 1.5, (1, 1))
        scored = leadwise_snr(np.tile(clean, (1, 1)), shifted, sampling_rate=100)
        self.assertEqual(scored['leads'][0]['shift_samples'], 0)
        self.assertEqual(scored['leads'][0]['shift_mv'], 0.0)

    def test_bootstrap_interval_contains_the_point(self):
        labels = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float64)
        probability = np.array([[0.9, 0.2], [0.2, 0.8], [0.85, 0.1], [0.15, 0.7]])
        patients = np.array([10, 10, 11, 11])
        interval = bootstrap_classwise_delta_ece(
            labels, probability, probability, ['NORM', 'MI'], patients, resamples=25, seed=42)
        self.assertEqual(interval['point']['NORM'], 0.0)
        self.assertEqual(interval['low']['NORM'], 0.0)
        self.assertEqual(interval['high']['NORM'], 0.0)

    def test_paired_score_requires_the_trace_check(self):
        from src.paired import require_digitizer_check
        with self.assertRaisesRegex(RuntimeError, 'fold-10'):
            require_digitizer_check({
                'digitizer_check': 'results/does_not_exist_digitizer_check.json',
                'correlation_threshold': 0.8,
            })


class PrintedRecordCheck(unittest.TestCase):
    def test_training_record_printed_page(self):
        kit = Path('.tools/ecg-image-kit/codes/ecg-image-generator/extract_leads.py')
        record = Path('data/ptb-xl-1.0.3/records100/00000/00001_lr.hea')
        if not kit.exists() or not record.exists():
            self.skipTest('ECG-Image-Kit or PTB-XL is not available')
        from src.digitize import check_training_record
        result = check_training_record()
        self.assertEqual(result['split'], 'train')
        self.assertEqual(result['status'], 'passed')
        self.assertGreaterEqual(result['minimum_correlation'], 0.8)
        saved = json.loads(Path('results/digitizer_check.json').read_text())
        self.assertEqual(saved['status'], 'passed')


if __name__ == '__main__':
    unittest.main()
