"""Algorithmic checks without downloading weights or using benchmark images."""
import unittest
import torch
from torch import nn
from approaches.prototype_model import CompositionalPrototypeRetrieval, balanced_prototypes, retrieve


class IdentityBackbone(nn.Module):
    num_features = 4

    def forward(self, images):
        return images


class PrototypeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.hp = {'embedding_dim': 4, 'temperature': .1, 'crop_weight': .5,
                   'num_seen_pairs': 2, 'seen_bias': 0., 'prototype_loss_weight': 1.}
        self.pairs = torch.tensor([[0, 0], [1, 1], [0, 1]])
        self.model = CompositionalPrototypeRetrieval(self.hp, self.pairs, 2, 2,
                                                    backbone=IdentityBackbone())
        self.features = torch.eye(4)
        self.crops = torch.tensor([0, 0, 1, 1])
        self.diseases = torch.tensor([0, 1, 0, 1])
        self.model.fit_prototypes(self.features, self.crops, self.diseases)

    def test_balancing_removes_partner_frequency(self):
        embeddings = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
        prototype, present = balanced_prototypes(embeddings, torch.zeros(3, dtype=torch.long),
                                                 torch.tensor([0, 0, 1]), 2)
        torch.testing.assert_close(prototype[0], torch.tensor([2**-.5, 2**-.5]))
        self.assertEqual(present.tolist(), [True, False])

    def test_unseen_pair_is_composed_in_declared_order(self):
        crop, disease = self.model.concept_scores(self.model.project(self.features))
        scores = self.model.score_features(self.features)
        expected = .5 * crop[:, self.pairs[:, 0]] + .5 * disease[:, self.pairs[:, 1]]
        torch.testing.assert_close(scores, expected)
        self.assertTrue(torch.isfinite(scores).all())
        self.assertTrue((scores[:, 2] > -1000).all())

    def test_gradients_and_backbone_frozen(self):
        self.model.train()
        self.assertFalse(self.model.backbone.training)
        loss = self.model.training_loss(self.features, self.crops, self.diseases)
        loss.backward()
        self.assertGreater(float(self.model.crop_head.weight.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for p in self.model.backbone.parameters()))

    def test_bias_only_changes_seen_columns_and_state_roundtrip(self):
        before = self.model(self.features).detach()
        self.model.hp['seen_bias'] = .7
        after = self.model(self.features).detach()
        torch.testing.assert_close(after[:, :2], before[:, :2] - .7)
        torch.testing.assert_close(after[:, 2], before[:, 2])
        restored = CompositionalPrototypeRetrieval(self.hp, self.pairs, 2, 2,
                                                  backbone=IdentityBackbone())
        restored.load_state_dict(self.model.state_dict())
        restored.hp['seen_bias'] = .7
        torch.testing.assert_close(restored(self.features), after)

    def test_retrieval_concept_exemplars_match_prediction(self):
        crop_emb, disease_emb = self.model.project(self.features)
        memory = {'crop_embeddings': crop_emb, 'disease_embeddings': disease_emb,
                  'crops': self.crops, 'diseases': self.diseases,
                  'paths': ['a.jpg', 'b.jpg', 'c.jpg', 'd.jpg']}
        outputs = retrieve(self.model, self.features[:1], memory, k=10)
        for item in outputs[0]['crop']:
            self.assertEqual(item['crop'], outputs[0]['pair'][0])
        for item in outputs[0]['disease']:
            self.assertEqual(item['disease'], outputs[0]['pair'][1])
        self.assertEqual(len(outputs[0]['joint']), 4)


if __name__ == '__main__':
    unittest.main()
