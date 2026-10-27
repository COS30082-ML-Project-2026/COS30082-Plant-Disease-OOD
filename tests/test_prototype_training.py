"""Training split isolation and an actual small optimization run on synthetic features."""
from types import SimpleNamespace
import unittest
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from approaches.prototype_model import CompositionalPrototypeRetrieval
from approaches.prototype_training import _fit, support_loader
from test_prototype_model import IdentityBackbone


class TrainingTests(unittest.TestCase):
    def test_support_uses_training_rows_and_validation_transform_only(self):
        train = SimpleNamespace(dataset=SimpleNamespace(frame=pd.DataFrame({'image': ['train.jpg']})),
                                batch_size=8)
        validation = SimpleNamespace(dataset=SimpleNamespace(
            frame=pd.DataFrame({'image': ['validation.jpg']}), transform=object()))
        support = support_loader(train, validation)
        self.assertEqual(support.dataset.frame['image'].tolist(), ['train.jpg'])
        self.assertIs(support.dataset.transform, validation.dataset.transform)

    def test_optimization_prototypes_and_memory_complete(self):
        torch.manual_seed(42)
        hp = {'embedding_dim': 4, 'temperature': .1, 'crop_weight': .5,
              'num_seen_pairs': 2, 'seen_bias': 0., 'prototype_loss_weight': 1., 'epochs': 3}
        model = CompositionalPrototypeRetrieval(hp, torch.tensor([[0, 0], [1, 1], [0, 1]]),
                                                2, 2, backbone=IdentityBackbone())
        features = torch.eye(4).repeat(3, 1)
        crops = torch.tensor([0, 0, 1, 1]).repeat(3)
        diseases = torch.tensor([0, 1, 0, 1]).repeat(3)
        pairs = torch.tensor([0, 2, 1, 1]).repeat(3)
        reference = (features, crops, diseases, pairs)
        loader = DataLoader(TensorDataset(*reference), batch_size=4)
        support = SimpleNamespace(dataset=SimpleNamespace(
            frame=pd.DataFrame({'image': [f'train-{i}.jpg' for i in range(12)]})))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
        history, memory = _fit(model, loader, optimizer, reference, reference, hp,
                               False, torch.device('cpu'), support)
        self.assertEqual(len(history), 3)
        self.assertTrue(all(torch.isfinite(torch.tensor(row['loss'])) for row in history))
        self.assertLess(history[-1]['loss'], history[0]['loss'])
        self.assertEqual(memory['source'], 'training_subset_only')
        self.assertEqual(len(memory['paths']), 12)
        self.assertEqual(model(features).shape, (12, 3))


if __name__ == '__main__':
    unittest.main()
