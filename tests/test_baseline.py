import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import TensorDataset
import yaml

from src.baseline_data import normalize
from src.metrics import classification_metrics
from src.models import ECGResNet
from src import train
from src import eval as baseline_eval


class BaselineChecks(unittest.TestCase):
    def test_evaluator_rejects_smoke_and_unreviewed_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for status in ['smoke_only', 'needs_review']:
                (out / 'training_summary.json').write_text(json.dumps({'status': status}))
                with patch('sys.argv', ['eval', '--run-dir', str(out)]):
                    with self.assertRaisesRegex(ValueError, 'do not evaluate test yet'):
                        baseline_eval.main()

    def test_auc_ties_and_missing_class(self):
        labels = np.array([[0], [0], [1], [1]])
        perfect = classification_metrics(labels, labels, ['MI'])
        self.assertEqual(perfect['macro_auroc'], 1)
        self.assertEqual(perfect['ece'], 0)
        self.assertEqual(perfect['brier'], 0)
        tied = classification_metrics(labels, np.full((4, 1), 0.5), ['MI'])
        self.assertEqual(tied['macro_auroc'], 0.5)
        self.assertEqual(tied['brier'], 0.25)
        self.assertEqual(tied['ece'], 0)
        reversed_scores = classification_metrics(labels, 1-labels, ['MI'])
        self.assertEqual(reversed_scores['macro_auroc'], 0)
        absent = classification_metrics(np.zeros((4, 1)), labels, ['MI'])
        self.assertIsNone(absent['macro_auroc'])

    def test_normalization_constant_and_lead_independence(self):
        x = np.tile(np.arange(1000, dtype=np.float32), (12, 1))
        x[0] = 4
        result = normalize(x)
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_array_equal(result[0], 0)
        np.testing.assert_allclose(result[1:].mean(axis=1), 0, atol=1e-6)
        np.testing.assert_allclose(result[1:].std(axis=1), 1, atol=1e-6)
        np.testing.assert_allclose(normalize(x[1:] * 2 + 20), result[1:], atol=1e-6)

    def test_model_gradients(self):
        torch.set_num_threads(2)
        model = ECGResNet(width=4, blocks_per_stage=1)
        logits = model(torch.randn(3, 12, 1000))
        self.assertEqual(tuple(logits.shape), (3, 5))
        torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits)).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_resume_matches_uninterrupted_cpu_training(self):
        config = yaml.safe_load(Path('configs/stage2.yaml').read_text())
        config.update(epochs=2, width=4, blocks_per_stage=1, batch_size=4, cpu_threads=2)

        def fake_split(frame, split, data, config, limit):
            rng = np.random.default_rng(12 if split == 'train' else 13)
            x = torch.from_numpy(rng.normal(size=(8, 12, 1000)).astype(np.float32))
            y = torch.from_numpy(np.tile([[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]], (4, 1)).astype(np.float32))
            return TensorDataset(x, y), np.arange(8)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'config.yaml'
            path.write_text(yaml.safe_dump(config))

            def run(out, resume=False):
                arguments = ['train', '--config', str(path), '--output-dir', str(out)]
                if resume:
                    arguments += ['--resume']
                with patch('sys.argv', arguments), patch.object(train, 'baseline_frame', return_value=None), \
                     patch.object(train, 'load_split', side_effect=fake_split), \
                     patch.object(train, 'provenance', return_value={'test': 'fixture'}), \
                     patch('torch.cuda.is_available', return_value=False), contextlib.redirect_stdout(io.StringIO()):
                    train.main()

            full, resumed = root / 'full', root / 'resumed'
            run(full)
            original = train.atomic_checkpoint

            def interrupted(path, state):
                original(path, state)
                if path.name == 'last.pt' and state['epoch'] == 0:
                    raise RuntimeError('Simulated disconnect after epoch checkpoint')

            with patch.object(train, 'atomic_checkpoint', side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, 'Simulated disconnect'):
                    run(resumed)
            run(resumed, resume=True)
            a = torch.load(full / 'last.pt', weights_only=False)
            b = torch.load(resumed / 'last.pt', weights_only=False)
            self.assertEqual(a['history'], b['history'])
            for key in a['model']:
                torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
            self.assertFalse(json.loads((resumed / 'training_summary.json').read_text())['test_evaluated'])


if __name__ == '__main__':
    unittest.main()
