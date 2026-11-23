import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torchvision import transforms

from common.dataset import get_pairs
from approach3_ood_backbones.data import shared_frames,compositional_split
from approach3_experiment_suite.protocol import (
    METHODS,class_space,training_targets,bias_scores,select_bias,accumulation_divisor,validate_config,
)
from approach3_experiment_suite.models import PromptModel,SiglipModel
from approach3_experiment_suite.engine import loss_for,train,symptom_banks
from approach3_experiment_suite.suite import Suite


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.config=json.loads((Path(__file__).parents[1]/'config.json').read_text())
        cls.frames=shared_frames();cls.pairs=get_pairs()

    def tiny_adapter(self):
        from open_clip.model import CLIP
        import open_clip
        base=CLIP(embed_dim=64,vision_cfg=dict(image_size=28,layers=1,width=64,patch_size=14,head_width=64),
                  text_cfg=dict(context_length=77,vocab_size=49408,width=64,heads=1,layers=1)).eval().requires_grad_(False)
        preprocess=transforms.Compose([transforms.Resize((28,28)),transforms.ToTensor()])
        tokenizer=open_clip.get_tokenizer('ViT-B-16')
        return SimpleNamespace(model=base,device=torch.device('cpu'),preprocess=preprocess,
            encode_text=lambda texts:nn.functional.normalize(base.encode_text(tokenizer(texts)).float(),dim=1),
            encode_images=lambda images:nn.functional.normalize(base.encode_image(torch.stack([preprocess(image) for image in images])).float(),dim=1),
            similarity=lambda image,text:base.logit_scale.exp()*image @ text.T,
            metadata={'scope':'random tiny CLIP; mechanics only'})

    def test_catalog_matches_recommended_only(self):
        self.assertEqual(set(METHODS),{'D3','R1','F2','R5','R4','R3','F3','F4'})
        config=copy.deepcopy(self.config);config['selected']=['R6']
        with self.assertRaises(ValueError):validate_config(config)

    def test_training_rows_not_contiguous_and_heldout_rejected(self):
        self.assertEqual(training_targets(torch.tensor([4,1]),[1,4,8]).tolist(),[1,0])
        with self.assertRaises(ValueError):training_targets(torch.tensor([2]),[1,4,8])

    def test_component_alignment(self):
        crops,diseases,columns=class_space([[8,7],[2,3],[8,3]])
        self.assertEqual(crops,[2,8]);self.assertEqual(diseases,[3,7])
        self.assertEqual(columns.tolist(),[[1,1],[0,0],[1,0]])

    def test_bias_validation_only_selection(self):
        seen=torch.tensor([[4.,1.,0.]])
        novel=torch.tensor([[4.,3.,0.]])
        record=select_bias(seen,torch.tensor([0]),novel,torch.tensor([1]),[0],[0.,1.,2.])
        self.assertGreater(record['bias'],0)
        self.assertEqual(len(record['history']),3)
        self.assertTrue(torch.isfinite(bias_scores(torch.ones(2,3),[0],1)).all())

    def test_accumulation_last_window(self):
        self.assertEqual([accumulation_divisor(i,6,4) for i in range(6)],[4,4,4,4,2,2])

    def test_loss_masks_heldout_negative_and_primitive_labels(self):
        logits=[torch.randn(2,3,requires_grad=True),torch.randn(2,2,requires_grad=True),torch.randn(2,2,requires_grad=True)]
        columns=torch.tensor([[0,0],[0,1],[1,1]])
        loss_for('F4',logits,torch.tensor([0,2]),[0,2],columns,[1,.5,.5]).backward()
        self.assertTrue(torch.equal(logits[0].grad[:,1],torch.zeros(2)))
        self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() for v in logits))

    def test_csp_and_troika_real_tiny_transformer_gradients(self):
        for troika in (False,True):
            adapter=self.tiny_adapter()
            model=PromptModel(adapter,self.pairs,self.config,troika)
            model.train();out=model(torch.randn(2,3,28,28))
            self.assertEqual(tuple(out[0].shape),(2,333))
            if troika:
                self.assertEqual(tuple(out[1].shape),(2,55));self.assertEqual(tuple(out[2].shape),(2,175))
                self.assertTrue(hasattr(model,'cross') and hasattr(model,'adapters'))
            nn.functional.cross_entropy(out[0],torch.tensor([0,1])).backward()
            self.assertIsNotNone(model.soft.grad)
            self.assertTrue(torch.isfinite(model.soft.grad).all())
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.base.parameters()))
            if troika:self.assertTrue(any(p.grad is not None for p in model.cross.parameters()))

    def test_prompt_training_checkpoint_roundtrip(self):
        adapter=self.tiny_adapter();config=copy.deepcopy(self.config)
        config.update(batch_size=2,accumulation_steps=1,precision='float32')
        reduced,holdout,_=compositional_split(self.frames['train'])
        with tempfile.TemporaryDirectory() as temporary:
            model,selection=train('R4',adapter,[],self.pairs,config,reduced.iloc[:2],holdout.iloc[:2],Path(temporary),True)
            self.assertTrue((Path(temporary)/'best.pt').is_file())
            self.assertEqual(selection['train_images'],2)
            self.assertEqual(len(selection['history']),1)
            self.assertTrue(all(not p.requires_grad for p in model.base.parameters()))

    def test_native_training_smoke_reporting_and_selection_gate(self):
        from approach3_experiment_suite.engine import execute_native
        config=copy.deepcopy(self.config);config.update(batch_size=2,accumulation_steps=1,precision='float32')
        adapter=self.tiny_adapter()
        with tempfile.TemporaryDirectory() as temporary,patch('approach3_experiment_suite.engine.load_adapter',return_value=adapter):
            path=Path(temporary)
            report=execute_native('R4',config,path,True,False)
            self.assertTrue((path/'selection_frozen.json').is_file())
            self.assertTrue((path/'comparison.csv').is_file())
            self.assertEqual(report['neural_training_images'],2)
            self.assertIsNone(report['unseen_crop_top1'])
            saved=json.loads((path/'selection_frozen.json').read_text())
            self.assertEqual(saved['membership_reference'],'full intended reduced-training partition')
            self.assertEqual(sum(r['samples'] for r in report['ood_breakdown']),4)

    def test_siglip_final_blocks_only_and_frozen_teacher(self):
        class TinySiglip(nn.Module):
            def __init__(self):
                super().__init__();self.input=nn.Linear(3,8)
                self.vision_model=nn.Module();self.vision_model.encoder=nn.Module()
                self.vision_model.encoder.layers=nn.ModuleList([nn.Linear(8,8) for _ in range(3)])
                self.text_model=nn.Linear(8,8);self.logit_scale=nn.Parameter(torch.tensor(0.));self.logit_bias=nn.Parameter(torch.tensor(0.))
            def get_image_features(self,pixel_values):
                x=self.input(pixel_values.mean((2,3)))
                for block in self.vision_model.encoder.layers:x=torch.tanh(block(x))
                return x
            def gradient_checkpointing_enable(self,**kwargs):pass
            def gradient_checkpointing_disable(self):pass
        adapter=SimpleNamespace(model=TinySiglip())
        config=copy.deepcopy(self.config);config['partial_blocks']=1
        banks=[nn.functional.normalize(torch.randn(n,8),dim=1) for n in (333,55,175)]
        model=SiglipModel(adapter,banks,config)
        pixels=torch.randn(2,3,4,4)
        logits=model(pixels);value=logits[0].sum()+model.preservation(pixels)
        value.backward()
        names=[name for name,p in model.named_parameters() if p.requires_grad]
        self.assertTrue(all('encoder.layers.2.' in name for name in names))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.teacher.parameters()))

    def test_symptom_labels_and_name_only_fallback(self):
        class Adapter:
            def encode_text(self,texts):return torch.ones(len(texts),4)
        bank,coverage=symptom_banks(Adapter(),self.config,self.pairs)
        self.assertEqual(tuple(bank.shape),(333,4))
        self.assertEqual(coverage['described_diseases'],6)
        self.assertGreater(coverage['covered_pairs'],0)
        self.assertLess(coverage['covered_pairs'],333)

    def test_notebook_session_does_not_run_models_or_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            suite=Suite(self.config,'smoke',False,temporary)
            self.assertEqual(len(suite.catalog()),8)
            self.assertFalse(suite.completed)
            suite.save()
            with self.assertRaises(RuntimeError):suite.save()

    def test_full_suite_requires_cuda(self):
        with patch('torch.cuda.is_available',return_value=False):
            with self.assertRaises(RuntimeError):Suite(self.config,'full')


if __name__=='__main__':unittest.main()
