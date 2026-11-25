"""Shared-vs-extended metric regression, especially intentionally tied scores."""
import unittest
import torch

from common import evaluate as shared
from common.dataset import get_pairs
from approach3_ood_backbones.ood_evaluation import metrics,top1_predictions
from approach3_ood_backbones.experiments import score_report
from approach3_ood_backbones.data import shared_frames


class EvaluationParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.pairs=get_pairs();cls.frames=shared_frames()

    def assert_same(self,scores,labels):
        original=scores.clone()
        expected=scores.topk(1,dim=1).indices[:,0]
        self.assertTrue(torch.equal(top1_predictions(scores),expected))
        result=metrics(scores,labels,self.pairs)
        self.assertEqual(result['top1'],shared.top_k_accuracy(scores,labels,1))
        self.assertEqual(result['top5'],shared.top_k_accuracy(scores,labels,5))
        crop,disease=shared.concept_accuracy(scores,labels)
        self.assertEqual((result['crop_acc'],result['disease_acc']),(crop,disease))
        self.assertTrue(torch.equal(scores,original))

    def test_all_333_scores_tied(self):
        scores=torch.zeros(333,333)
        # Every possible target, so a wrong tied prediction cannot be hidden by
        # an aggregate coincidence. Check the predicted indices as well as counts.
        self.assert_same(scores,torch.arange(333))
        labels=scores.topk(1,dim=1).indices[:,0]
        result=metrics(scores,labels,self.pairs)
        self.assertEqual(result['top1'],100.)
        self.assertTrue(all(row['top1']==100. for row in result['per_class']))

    def test_two_equal_maxima(self):
        scores=torch.full((32,333),-5.)
        for i in range(32):scores[i,i]=scores[i,332-i]=7.
        self.assert_same(scores,scores.topk(1,dim=1).indices[:,0])

    def test_top5_boundary_ties(self):
        scores=torch.zeros(40,333)
        scores[:,:9]=1.;scores[:,4]=2.
        self.assert_same(scores,torch.arange(40)%10)

    def test_quantized_logits_ties(self):
        generator=torch.Generator().manual_seed(42)
        scores=torch.randn(101,333,generator=generator).mul(2).round()
        labels=torch.randint(333,(101,),generator=generator)
        self.assert_same(scores,labels)

    def test_concepts_keep_shared_argmax_rule(self):
        scores=torch.zeros(2,333);labels=torch.tensor([0,1])
        crop,disease=shared.concept_accuracy(scores,labels)
        result=metrics(scores,labels,self.pairs)
        self.assertEqual(result['crop_acc'],crop)
        self.assertEqual(result['disease_acc'],disease)

    def test_full_split_metrics_and_harmonic_mean(self):
        generator=torch.Generator().manual_seed(101)
        seen=torch.randn(2000,333,generator=generator).round()
        unseen=torch.randn(1448,333,generator=generator).round()
        # Include nontrivial correct predictions so H is exercised, not just zero.
        for scores,frame in [(seen,self.frames['seen']),(unseen,self.frames['unseen'])]:
            labels=torch.tensor(frame.pair_row.to_numpy())
            scores[torch.arange(0,len(scores),3),labels[::3]]=10.
        report=score_report(seen,unseen,self.frames,self.pairs)
        for name,scores in [('seen',seen),('unseen',unseen)]:
            labels=torch.tensor(self.frames[name].pair_row.to_numpy())
            for k in (1,5):self.assertEqual(report[f'{name}_top{k}'],shared.top_k_accuracy(scores,labels,k))
        self.assertEqual(report['harmonic_mean'],shared.harmonic_mean(report['seen_top1'],report['unseen_top1']))


if __name__=='__main__':unittest.main()
