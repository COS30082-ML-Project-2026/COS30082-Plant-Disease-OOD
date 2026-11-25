"""Regression checks for Norman's 323-development / 333-final protocol."""
import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch
from common import config, dataset, evaluate


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        dataset._pair_table.cache_clear()

    def tearDown(self):
        dataset._pair_table.cache_clear()

    def test_development_never_reads_official_metadata(self):
        original = pd.read_csv
        reads = []
        def guarded(path, *args, **kwargs):
            self.assertEqual(Path(path).resolve(), config.TRAIN_CSV.resolve())
            reads.append(path)
            return original(path, *args, **kwargs)
        with patch('pandas.read_csv', side_effect=guarded), patch.object(config, 'NUM_WORKERS', 0):
            self.assertEqual(dataset.get_pairs().shape, (323, 2))
            self.assertEqual(len(dataset.get_pair_names()), 323)
            crops, diseases = dataset.get_crop_and_disease_names()
            self.assertEqual((len(crops), len(diseases)), (55, 175))
            train, val, seen, unseen = dataset.get_loaders(2)
            self.assertEqual((len(train.dataset), len(val.dataset)), (7223, 777))
            self.assertIsNone(seen); self.assertIsNone(unseen)
            self.assertFalse(set(train.dataset.frame.image) & set(val.dataset.frame.image))
            images, crop, disease, labels = next(iter(val))
            self.assertEqual(images.shape, (2, 3, 224, 224))
            self.assertTrue(torch.equal(dataset.get_pairs()[labels], torch.stack([crop, disease], 1)))
        self.assertTrue(reads)

    def test_unguarded_test_access_fails_before_read(self):
        with patch('pandas.read_csv', side_effect=AssertionError('Must reject before reading')):
            for path in (config.TEST_SEEN_CSV, config.TEST_UNSEEN_CSV):
                with self.assertRaises(ValueError): dataset.read_csv(path)
                with self.assertRaises(ValueError): dataset.load_split(path)
            with self.assertRaises(ValueError): dataset.get_final_test_loaders(2)

    def test_original_split_membership_unchanged(self):
        frame = dataset.load_split(config.TRAIN_CSV)
        train, val = dataset._split_train_val(frame)
        # Original Norman split: same pandas operation, seed and ordered source.
        expected = frame.groupby('pair_row').sample(frac=.1, random_state=42)
        self.assertEqual(val.image.tolist(), expected.image.tolist())
        self.assertEqual(train.image.tolist(), frame.drop(expected.index).image.tolist())

    def test_final_order_mapping_and_cache_isolation(self):
        development = dataset.get_pairs()
        final = dataset.get_pairs(final_evaluation=True)
        self.assertEqual(final.shape, (333, 2))
        self.assertTrue(torch.equal(final[:323], development))
        self.assertEqual(len(set(map(tuple, final.tolist()))), 333)
        self.assertTrue(torch.equal(dataset.get_pairs(), development))
        with patch.object(config, 'NUM_WORKERS', 0):
            seen, unseen = dataset.get_final_test_loaders(2, final_evaluation=True)
        self.assertEqual((len(seen.dataset), len(unseen.dataset)), (2000, 1448))
        self.assertEqual(seen.dataset.role, 'official_seen')
        self.assertEqual(set(unseen.dataset.frame.pair_row), set(range(323, 333)))
        # Exactly the legacy final order, without using test identities for development.
        raw_seen = dataset.read_csv(config.TEST_SEEN_CSV, final_evaluation=True)
        raw_train = dataset.read_csv(config.TRAIN_CSV)
        raw_unseen = dataset.read_csv(config.TEST_UNSEEN_CSV, final_evaluation=True)
        legacy = pd.concat([
            pd.concat([raw_train, raw_seen]).drop_duplicates(['crop', 'disease']).sort_values('pair_label'),
            raw_unseen.drop_duplicates(['crop', 'disease']).sort_values('pair_label')])
        self.assertEqual(final.tolist(), legacy[['crop', 'disease']].values.tolist())

    def test_validation_rejects_333_and_preserves_topk(self):
        loader = SimpleNamespace(dataset=SimpleNamespace(role='seen_dev'))
        scores = torch.zeros(2, 323)
        labels = scores.topk(1, dim=1).indices[:, 0]
        with patch.object(evaluate, 'collect_scores', return_value=(scores, labels)):
            self.assertEqual(evaluate.val_accuracy(None, loader), 100.)
        with patch.object(evaluate, 'collect_scores', return_value=(torch.zeros(2, 333), labels)):
            with self.assertRaises(ValueError): evaluate.val_accuracy(None, loader)

    def test_final_metric_formulas_preserved_and_roles_checked(self):
        seen_loader = SimpleNamespace(dataset=SimpleNamespace(role='official_seen'))
        unseen_loader = SimpleNamespace(dataset=SimpleNamespace(role='official_unseen'))
        a, b = torch.zeros(2, 333), torch.zeros(2, 333)
        a[0, 0] = 1; a[1, 1] = 1; b[0, 323] = 1; b[1, 324] = 1
        ay, by = torch.tensor([0, 1]), torch.tensor([323, 325])
        with tempfile.TemporaryDirectory() as folder, patch.object(config, 'RESULTS_DIR', Path(folder)), \
                patch.object(evaluate, 'collect_scores', side_effect=[(a, ay), (b, by)]), \
                patch.object(evaluate, 'save_confusion'):
            report = evaluate.evaluate(None, seen_loader, unseen_loader, 'synthetic_only', {})
        self.assertEqual(report['seen_top1'], 100.)
        self.assertEqual(report['unseen_top1'], 50.)
        self.assertEqual(report['harmonic_mean'], 66.67)
        for prefix, scores, labels in [('seen', a, ay), ('unseen', b, by)]:
            for k in (1, 5):
                expected = round(100 * (scores.topk(k, dim=1).indices == labels[:, None]).any(1).float().mean().item(), 2)
                self.assertEqual(report[f'{prefix}_top{k}'], expected)
        with self.assertRaises(ValueError): evaluate.evaluate(None, None, None, 'bad', {})

    def test_notebooks_access_tests_only_in_final_cell(self):
        paths = list((config.ROOT_DIR / 'baselines').glob('*.ipynb'))
        paths += list((config.ROOT_DIR / 'approaches').glob('approach*.ipynb'))
        self.assertEqual(len(paths), 5)
        for path in paths:
            notebook = json.loads(path.read_text(encoding='utf8'))
            final_cells = []
            for cell in notebook['cells']:
                if cell['cell_type'] != 'code': continue
                text = ''.join(cell['source'])
                tree = ast.parse(text)
                is_final = any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                               and node.func.id == 'evaluate' for node in ast.walk(tree))
                if is_final:
                    final_cells.append(text)
                    self.assertIn('get_pairs(final_evaluation=True)', text)
                    self.assertIn('get_final_test_loaders(', text)
                else:
                    self.assertNotIn('final_evaluation=True', text)
                    self.assertNotIn('cfg.TEST_UNSEEN_CSV', text)
                    self.assertNotIn('cfg.TEST_SEEN_CSV', text)
            self.assertEqual(len(final_cells), 1, str(path))


if __name__ == '__main__':
    unittest.main()
