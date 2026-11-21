"""Offline tests: synthetic scores are mechanics checks, never accuracy evidence."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
import torch.nn.functional as F

from approach3_ood_backbones.data import shared_frames, compositional_split, signature
from approach3_ood_backbones.factorized import combine, select_weights, standardize
from approach3_ood_backbones.ood_evaluation import metrics, breakdown, counts
from approach3_ood_backbones.adapters import load_adapter, OpenClipAdapter, Siglip2Adapter, snapshot
from approach3_ood_backbones.experiments import run, text_bank, score_report
from common.dataset import get_pairs


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = shared_frames()
        cls.pairs = get_pairs()
        cls.config = json.loads((Path(__file__).parents[1] / 'configs/default.json').read_text())

    def test_original_membership(self):
        self.assertEqual([len(f) for f in self.frames.values()], [7223, 777, 2000, 1448])
        rows = counts(self.frames['unseen'], self.frames['train'])
        self.assertEqual(sum(r['samples'] for r in rows if r['crop_seen'] and r['disease_seen'] and not r['pair_seen']), 1448)
        self.assertEqual(sum(r['samples'] for r in rows if not r['crop_seen'] or not r['disease_seen']), 0)

    def test_holdout_reproducible_and_disjoint(self):
        a, b, record = compositional_split(self.frames['train'])
        a2, b2, record2 = compositional_split(self.frames['train'])
        self.assertEqual(record, record2)
        self.assertEqual(signature(a), signature(a2))
        self.assertEqual(signature(b), signature(b2))
        self.assertFalse(set(a.pair_row) & set(b.pair_row))
        self.assertEqual(set(a.crop), set(self.frames['train'].crop))
        self.assertEqual(set(a.disease), set(self.frames['train'].disease))
        self.assertEqual(set(a.image) | set(b.image), set(self.frames['train'].image))
        for other in ['validation', 'seen', 'unseen']:
            self.assertFalse(set(b.image) & set(self.frames[other].image))

    def test_infeasible_holdout(self):
        frame = pd.DataFrame({'pair_row':[0,1], 'crop':[0,1], 'disease':[0,1], 'image':['a','b']})
        with self.assertRaisesRegex(ValueError, 'No pair'):
            compositional_split(frame)

    def test_empty_and_exact_metrics(self):
        pairs = torch.tensor([[0,0], [1,0], [0,1], [1,1], [2,2], [3,3]])
        scores = torch.eye(6) * 10
        record = metrics(scores, torch.arange(6), pairs)
        self.assertEqual(record['top1'], 100)
        self.assertEqual(record['crop_acc'], 100)
        empty = metrics(scores[:0], torch.empty(0, dtype=torch.long), pairs)
        self.assertIsNone(empty['top1'])
        self.assertEqual(empty['samples'], 0)
        with self.assertRaises(ValueError):
            metrics(torch.full((1,6), float('nan')), [0], pairs)

    def test_breakdown_partitions_and_na(self):
        frame = self.frames['unseen'].iloc[:3]
        scores = torch.zeros(3,333)
        rows = breakdown(scores, frame, self.frames['train'], self.pairs)
        self.assertEqual(sum(r['samples'] for r in rows), 3)
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(r['top1'] is None for r in rows if r['samples'] == 0))

    def test_four_component_categories(self):
        train = pd.DataFrame({'crop':[0], 'disease':[0]})
        frame = pd.DataFrame({'crop':[0,0,1,1], 'disease':[0,1,0,1], 'pair_row':[0,1,2,3]})
        rows = breakdown(torch.eye(4), frame, train, torch.tensor([[0,0],[0,1],[1,0],[1,1]]))
        self.assertEqual(sum(r['samples'] for r in rows),4)
        self.assertEqual(sum(r['samples'] > 0 for r in rows),4)

    def test_factorized_alignment_and_pair_ablation(self):
        components = (torch.tensor([[0.,2.,1.]]), torch.tensor([[10.,20.]]), torch.tensor([[30.,10.]]))
        columns = torch.tensor([[1,0], [0,1], [1,1]])
        pair_only = combine(components, columns, [1,0,0])
        self.assertEqual(pair_only.argmax().item(),1)
        expected = standardize(components[1])[:,columns[:,0]]
        torch.testing.assert_close(combine(components,columns,[0,1,0]), expected)
        self.assertTrue(torch.isfinite(standardize(torch.ones(2,3))).all())
        with self.assertRaises(ValueError):
            combine(components,columns,[1,1,0])

    def test_selection_uses_only_supplied_validation(self):
        components=(torch.tensor([[2.,1.]]),torch.tensor([[0.,1.]]),torch.tensor([[0.,1.]]))
        result=select_weights(components,[[0,0],[1,1]],[0],[[1,0,0],[0,.5,.5]])
        self.assertEqual(result['weights'],[1,0,0])
        with self.assertRaises(ValueError):
            select_weights(components,[[0,0],[1,1]],[],[[1,0,0]])

    def test_logit_scale_and_bias(self):
        class Model:
            logit_scale=torch.tensor(2.).log()
            logit_bias=torch.tensor(-1.)
        for adapter_type in (OpenClipAdapter,Siglip2Adapter):
            adapter=adapter_type.__new__(adapter_type); adapter.model=Model()
            torch.testing.assert_close(adapter.similarity(torch.eye(2),torch.eye(2)),2*torch.eye(2)-1)

    def test_unknown_model_rejected(self):
        with self.assertRaises(ValueError):
            load_adapter({'backend':'invented'}, 'cpu')

    def test_full_benchmark_requires_cuda(self):
        with tempfile.TemporaryDirectory() as temporary, patch('torch.cuda.is_available',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'require CUDA'):
                run(self.config,'F1',['clip_b16'],smoke=False,output=temporary)

    def test_adapter_dispatch_and_download_opt_in(self):
        with patch('approach3_ood_backbones.adapters.Siglip2Adapter') as siglip:
            spec={'backend':'transformers'}
            self.assertIs(load_adapter(spec,'cpu'),siglip.return_value)
            siglip.assert_called_once_with(spec,'cpu',False)
        with patch('approach3_ood_backbones.adapters.OpenClipAdapter') as clip:
            spec={'backend':'open_clip_hub'}
            load_adapter(spec,'cuda',True)
            clip.assert_called_once_with(spec,'cuda',True)

    def test_snapshot_is_offline_by_default(self):
        with patch('huggingface_hub.snapshot_download', return_value='/cache/snapshots/abc123') as download:
            path,revision=snapshot('example/model','main',False,['*.json'])
            self.assertEqual(revision,'abc123')
            self.assertTrue(download.call_args.kwargs['local_files_only'])

    def test_nonperfect_topk_and_component_accuracy(self):
        pairs=torch.tensor([[0,0],[0,1],[1,0],[1,1],[2,0],[2,1]])
        scores=torch.tensor([[6.,5.,4.,3.,2.,1.],[6.,5.,4.,3.,2.,1.]])
        report=metrics(scores,[1,5],pairs)
        self.assertEqual(report['top1'],0)
        self.assertEqual(report['top5'],50)
        self.assertEqual(report['crop_acc'],50)
        self.assertEqual(report['disease_acc'],0)

    def test_true_unseen_components_are_na_not_unseen_split_accuracy(self):
        frames={k:(v.iloc[:2] if k!='train' else v) for k,v in self.frames.items()}
        report=score_report(torch.zeros(2,333),torch.zeros(2,333),frames,self.pairs)
        self.assertIsNone(report['unseen_crop_top1'])
        self.assertIsNone(report['unseen_disease_top1'])
        self.assertEqual(report['unseen_combination_samples'],2)
        self.assertIsNotNone(report['unseen_split_crop_acc'])

    def test_siglip_preprocessing_contract(self):
        class Batch(dict):
            def to(self,device): return self
        class Processor:
            calls=[]
            def __call__(self,**kwargs):
                self.calls.append(kwargs)
                return Batch(input_ids=torch.zeros(len(kwargs['text']),64,dtype=torch.long))
        class Model:
            def get_text_features(self,**kwargs): return torch.ones(len(kwargs['input_ids']),4)
        adapter=Siglip2Adapter.__new__(Siglip2Adapter)
        adapter.processor=Processor(); adapter.model=Model(); adapter.device='cpu'
        self.assertEqual(tuple(adapter.encode_text(['HELLO']).shape),(1,4))
        call=adapter.processor.calls[-1]
        self.assertEqual(call['text'],['hello'])
        self.assertEqual(call['max_length'],64)
        self.assertEqual(call['padding'],'max_length')

    def test_cpu_pipeline_smoke(self):
        class FakeAdapter:
            metadata={'scope':'synthetic mechanics only, no pretrained inference'}
            def encode_text(self,texts): return F.normalize(torch.arange(len(texts)*8).reshape(len(texts),8).float()+1,dim=-1)
            def encode_images(self,images): return F.normalize(torch.ones(len(images),8),dim=-1)
            def similarity(self,images,text): return images @ text.T
        with tempfile.TemporaryDirectory() as temporary:
            for stage,names in [('F1',['clip_b16','clip_l14','siglip2','bioclip']),('F2',['siglip2'])]:
                from approach3_ood_backbones.experiments import collect as real_collect
                test_images=set(self.frames['seen'].image) | set(self.frames['unseen'].image)
                def guarded_collect(adapter,banks,frame,batch_size,perturbation=None):
                    if stage=='F2' and test_images & set(frame.image):
                        frozen=list(Path(temporary).rglob('siglip2_selection_frozen.json'))
                        self.assertTrue(frozen,'Official test inference occurred before selection freeze.')
                    return real_collect(adapter,banks,frame,batch_size,perturbation)
                with patch('approach3_ood_backbones.experiments.collect',side_effect=guarded_collect):
                    path,reports=run(self.config,stage,names,smoke=True,output=temporary,
                                     adapter_factory=lambda *args:FakeAdapter())
                self.assertTrue((path/'comparison.csv').is_file())
                self.assertTrue((path/'ood_breakdown.csv').is_file())
                self.assertTrue(all(r['smoke'] and r['neural_training_images']==0 for r in reports))
                self.assertTrue(all(r['unseen_crop_samples']==0 and r['unseen_crop_top1'] is None for r in reports))
                self.assertTrue(all(r['unseen_disease_samples']==0 and r['unseen_disease_top1'] is None for r in reports))
                self.assertTrue(all(r['unseen_combination_samples']==2 for r in reports))
                if stage=='F2': self.assertTrue((path/'siglip2_selection_frozen.json').is_file())


if __name__ == '__main__':
    unittest.main()
