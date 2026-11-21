import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from approach3_dino_fusion.core import (
    prototypes,pair_crop_columns,fuse,zscore,crop_accuracy,reproduction,
    canonical_hash,validate_e1,Dino,E1,
)
from approach3_dino_fusion.run import check_config,require_reproduction,execute
from approach3_ood_backbones.data import shared_frames
from common.dataset import get_pairs


class FusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(__file__).parents[1]
        cls.config=json.loads((cls.root/'config.json').read_text())
        cls.reference=json.loads((cls.root/'reference_results/E1_wise.json').read_text())
        cls.frames=shared_frames()

    def test_original_linux_split_hashes(self):
        for name in ('train','validation'):
            self.assertEqual(canonical_hash(self.frames[name]),self.reference['hp']['split'][name])

    def test_prototype_normalization_and_counts(self):
        features=torch.tensor([[1.,0.],[1.,1.],[0.,1.]])
        centers,counts=prototypes(features,[8,8,3],[3,8])
        torch.testing.assert_close(centers.norm(dim=1),torch.ones(2))
        self.assertEqual(counts.tolist(),[1,2])
        self.assertGreater(centers[1,0],centers[1,1])

    def test_missing_crop_rejected(self):
        with self.assertRaisesRegex(ValueError,'No training examples'):
            prototypes(torch.eye(2),[0,0],[0,1])
        with self.assertRaises(ValueError):
            pair_crop_columns([[8,0]],[0])

    def test_pair_mapping_noncontiguous_ids(self):
        self.assertEqual(pair_crop_columns([[7,9],[2,3],[7,1]],[2,7]).tolist(),[1,0,1])
        columns=pair_crop_columns(get_pairs(),sorted(set(self.frames['train'].crop)))
        self.assertEqual(tuple(columns.shape),(333,))

    def test_fixed_fusion_and_crop_ties(self):
        clip=torch.tensor([[0.,1.,2.,3.]])
        crop=torch.tensor([[4.,1.]])
        columns=torch.tensor([0,0,1,1])
        actual=fuse(clip,crop,columns)
        torch.testing.assert_close(actual,.8*zscore(clip)+.2*zscore(crop)[:,columns])
        # DINO gives the SAME contribution to every disease of a given crop.
        delta=actual-.8*zscore(clip)
        torch.testing.assert_close(delta[0,0],delta[0,1])
        with self.assertRaises(ValueError):fuse(clip,crop,columns,.5)

    def test_degenerate_calibration_finite(self):
        self.assertTrue(torch.isfinite(zscore(torch.ones(2,55))).all())
        with self.assertRaises(ValueError):zscore(torch.full((1,2),float('nan')))

    def test_crop_only_is_not_pair_accuracy(self):
        report=crop_accuracy(torch.eye(2),[7,2],[2,7])
        self.assertEqual(report['crop_top1'],0)
        self.assertEqual(report['crop_top5'],100)
        self.assertIsNone(report['pair_top1'])
        self.assertIsNone(report['disease_acc'])

    def test_reproduction_gate_checks_robustness(self):
        actual=copy.deepcopy(self.reference)
        self.assertTrue(reproduction(actual,self.reference)['passed'])
        actual['robustness']['noise']['top1']+=.05
        record=reproduction(actual,self.reference)
        self.assertFalse(record['passed'])
        with self.assertRaisesRegex(RuntimeError,'Fusion aborted'):require_reproduction(record)

    def test_no_retuning_or_loose_tolerance(self):
        for key,value in [('clip_fraction',.5),('reproduction_tolerance_percentage_points',1.)]:
            config=copy.deepcopy(self.config);config[key]=value
            with self.assertRaises(ValueError):check_config(config)

    def test_reproduction_requires_cuda(self):
        with patch('torch.cuda.is_available',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'school CUDA'):
                execute(self.config,Path('unused.pt'),'verify')

    def test_e1_candidate_order_rejected(self):
        checkpoint=dict(model_name='E1_wise',debug=False,
            hp=dict(backbone='ViT-B-16',pretrained='openai',norm='clip',seed=42,batch_size=8),
            selection=dict(prompt_style='single',alpha=.75),pairs=get_pairs().flip(0))
        with self.assertRaisesRegex(ValueError,'candidate order'):
            validate_e1(checkpoint,self.frames)

    def test_e1_reuses_existing_loader_and_freezes_model(self):
        from approach3_clip_improvements.approaches.clip_improvement_core import prompt_groups
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__();self.visual=torch.nn.Linear(2,2);self.logit_scale=torch.nn.Parameter(torch.tensor(0.))
            def encode_text(self,tokens):return tokens.float()
            def encode_image(self,images):return torch.ones(len(images),2,device=images.device)
        model=TinyModel()
        hp=dict(backbone='ViT-B-16',pretrained='openai',norm='clip',seed=42,text_chunk_size=16,batch_size=8,
                prompt='a photo of a {crop} leaf with {disease}')
        payload=dict(model_name='E1_wise',debug=False,hp=hp,selection=self.reference['hp'],
                     pairs=get_pairs(),prompts=prompt_groups(get_pairs(),hp,'single'),state_dict=model.state_dict())
        with patch('approach3_dino_fusion.core.file_hash',return_value='expected'), \
             patch('approach3_dino_fusion.core.read_checkpoint',return_value=payload) as loader, \
             patch('open_clip.create_model',return_value=model) as create, \
             patch('open_clip.get_tokenizer',return_value=lambda texts:torch.ones(len(texts),2)):
            adapter=E1(Path('E1_wise.pt'),self.frames,'cpu','expected')
            loader.assert_called_once_with(Path('E1_wise.pt'))
            create.assert_called_once_with('ViT-B-16',force_quick_gelu=True)
            self.assertFalse(adapter.model.training)
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in adapter.model.parameters()))
            self.assertEqual(adapter.prompts,payload['prompts'])
            self.assertEqual(tuple(adapter.scores(self.frames['seen'].iloc[:2]).shape),(2,333))
            changed=copy.deepcopy(payload);changed['prompts'][0]=['wrong prompt']
            with self.assertRaisesRegex(ValueError,'Stored E1 prompts'):
                validate_e1(changed,self.frames)

    def test_failed_full_reproduction_never_loads_dino(self):
        class FakeE1:
            metadata={}; source_hash='unused'
            def __init__(self,*args):self.pairs=get_pairs()
            def scores(self,frame):return torch.zeros(len(frame),333)
            def loader(self,frame):return None
            def predict(self,images):return None
        actual=copy.deepcopy(self.reference)
        wrong=copy.deepcopy(self.reference['robustness']);wrong['noise']['top1']+=1
        with tempfile.TemporaryDirectory() as temporary, \
             patch('torch.cuda.is_available',return_value=True), \
             patch('torch.cuda.get_device_name',return_value='Mock RTX 5090'), \
             patch('torch.cuda.reset_peak_memory_stats'), \
             patch('approach3_dino_fusion.run.E1',FakeE1), \
             patch('approach3_dino_fusion.run.Dino') as dino, \
             patch('approach3_dino_fusion.run.score_report',return_value=actual), \
             patch('approach3_dino_fusion.run.robustness',return_value=wrong):
            with self.assertRaisesRegex(RuntimeError,'Fusion aborted'):
                execute(self.config,Path('unused.pt'),'full',output=temporary)
            dino.assert_not_called()
            records=list(Path(temporary).rglob('reproduction.json'))
            self.assertEqual(len(records),1)
            self.assertFalse(json.loads(records[0].read_text())['passed'])

    def test_cpu_pipeline_smoke_no_gradients(self):
        class FakeE1:
            def __init__(self,*args):
                self.model=torch.nn.Linear(2,2).eval().requires_grad_(False)
                self.metadata={'scope':'synthetic mechanics only'}
                self.pairs=get_pairs()
            def scores(self,frame):return torch.zeros(len(frame),333)
        class FakeDino:
            def __init__(self,*args):
                self.model=torch.nn.Linear(2,2).eval().requires_grad_(False)
                self.metadata={'scope':'synthetic mechanics only'}
            def embeddings(self,frame,batch_size):
                return torch.nn.functional.normalize(torch.arange(len(frame)*8).reshape(len(frame),8).float()+1,dim=1)
        with tempfile.TemporaryDirectory() as temporary, \
             patch('torch.cuda.is_available',return_value=False), \
             patch('approach3_dino_fusion.run.E1',FakeE1), \
             patch('approach3_dino_fusion.run.Dino',FakeDino), \
             patch('approach3_dino_fusion.run.file_hash',return_value=self.config['e1_sha256']):
            path=execute(self.config,Path('unused.pt'),'smoke',output=temporary)
            report=json.loads((path/'comparison.json').read_text())
            self.assertTrue(report['smoke'])
            self.assertEqual(report['prototype_images'],55)
            self.assertIsNone(report['fusion']['unseen_crop_top1'])
            self.assertIsNone(report['DINO_crop']['seen']['pair_top1'])
            self.assertIsNone(json.loads((path/'reproduction.json').read_text())['passed'])
            self.assertTrue((path/'predictions.csv').is_file())

    def test_dino_cls_gpu_interface_cpu_mock(self):
        class Batch(dict):
            def to(self,device):return self
        class Processor:
            def __call__(self,**kwargs):return Batch(pixel_values=torch.ones(len(kwargs['images']),3,2,2))
        class Output:pass
        class Model:
            def __call__(self,**kwargs):
                result=Output();result.last_hidden_state=torch.ones(len(kwargs['pixel_values']),3,4)
                result.last_hidden_state[:,0,0]=2
                return result
        dino=Dino.__new__(Dino);dino.processor=Processor();dino.model=Model();dino.device='cpu'
        features=dino.embeddings(self.frames['train'].iloc[:2],1)
        self.assertEqual(tuple(features.shape),(2,4))
        self.assertGreater(features[0,0],features[0,1])
        self.assertFalse(features.requires_grad)
        torch.testing.assert_close(features.norm(dim=1),torch.ones(2))


if __name__=='__main__':unittest.main()
