"""Regression checks for diagnostic aggregation and patient separation."""
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.data import build_labels


class DataChecks(unittest.TestCase):
    def prepare(self, root, leak=False):
        pd.DataFrame([
            {'ecg_id': 1, 'patient_id': 10, 'strat_fold': 1, 'scp_codes': "{'A': 0, 'B': 100, 'R': 100}"},
            {'ecg_id': 2, 'patient_id': 10 if leak else 20, 'strat_fold': 9, 'scp_codes': "{'A': 100}"},
            {'ecg_id': 3, 'patient_id': 30, 'strat_fold': 10, 'scp_codes': "{'R': 100}"},
        ]).to_csv(root / 'ptbxl_database.csv', index=False)
        pd.DataFrame([
            {'code': 'A', 'diagnostic': 1, 'diagnostic_class': 'MI'},
            {'code': 'B', 'diagnostic': 1, 'diagnostic_class': 'CD'},
            {'code': 'R', 'diagnostic': 0, 'diagnostic_class': 'NORM'},
        ]).set_index('code').to_csv(root / 'scp_statements.csv')
        return {'classes': ['NORM', 'MI', 'STTC', 'CD', 'HYP'],
                'splits': {'train': list(range(1, 9)), 'validation': [9], 'test': [10]}}

    def test_multilabel_zero_likelihood_and_nondiagnostic_codes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = build_labels(root, self.prepare(root))
            self.assertEqual(frame.loc[1, ['MI', 'CD', 'NORM']].tolist(), [1, 1, 0])
            self.assertFalse(frame.loc[3, 'has_diagnostic_label'])
            self.assertEqual(frame.split.tolist(), ['train', 'validation', 'test'])

    def test_patient_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.prepare(root, leak=True)
            with self.assertRaisesRegex(AssertionError, 'Patient leakage'):
                build_labels(root, config)


if __name__ == '__main__':
    unittest.main()
