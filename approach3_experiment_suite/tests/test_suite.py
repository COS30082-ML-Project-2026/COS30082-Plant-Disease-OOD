"""Fresh-protocol tests. Tiny/random backbones test mechanics, NOT accuracy."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pandas as pd
import torch
from torch import nn
from torchvision import transforms
from common import config as shared
from approach3_experiment_suite.suite import Suite, final_evaluate
from approach3_experiment_suite.fresh import banks_for, expand_model, choose_weights
from approach3_experiment_suite.fresh_data import development, final_data
from approach3_experiment_suite.protocol import training_targets
from approach3_experiment_suite.engine import loss_for


class TinySiglip(nn.Module):
    def __init__(self):
        super().__init__(); self.input=nn.Linear(3,8)
        self.vision_model=nn.Module(); self.vision_model.encoder=nn.Module()
        self.vision_model.encoder.layers=nn.ModuleList([nn.Linear(8,8) for _ in range(2)])
        self.text_model=nn.Linear(8,8)
        self.logit_scale=nn.Parameter(torch.tensor(0.)); self.logit_bias=nn.Parameter(torch.tensor(0.))
    def get_image_features(self,pixel_values):
        x=self.input(pixel_values.mean((2,3)))
        for block in self.vision_model.encoder.layers:x=torch.tanh(block(x))
        return x
    def gradient_checkpointing_enable(self,**kwargs):pass
    def gradient_checkpointing_disable(self):pass


class TinyDino:
    def __init__(self,config,device,allow_download=False):
        self.metadata=dict(repository=config['dino_repository'],resolved_revision='tiny-dino-test-only',preprocessing='mock',pooling=config['dino_pooling'])
    def embeddings(self,frame,batch_size):
        values=torch.tensor([[sum(name.encode())%(i+7)+1 for i in range(8)] for name in frame.image]).float()
        return nn.functional.normalize(values,dim=1)


def tiny_adapter(spec,device,allow_download):
    # Factory resets under suite.seed(), so final reload recreates the same base.
    siglip=spec['backend']=='transformers'
    if siglip:base=TinySiglip(); dim=8
    else:
        from open_clip.model import CLIP
        base=CLIP(embed_dim=64,vision_cfg=dict(image_size=28,layers=1,width=64,patch_size=14,head_width=64),
                  text_cfg=dict(context_length=77,vocab_size=49408,width=64,heads=1,layers=1));dim=64
    base.eval().requires_grad_(False)
    preprocess=transforms.Compose([transforms.Resize((28,28)),transforms.ToTensor()])
    def text(texts):
        # Deterministic mock semantic embeddings; never loads model weights.
        return nn.functional.normalize(torch.tensor([[sum(s.encode())%(i+11)+1 for i in range(dim)] for s in texts]).float(),dim=1)
    def images(images):
        pixels=torch.stack([preprocess(im) for im in images])
        with torch.no_grad():features=base.get_image_features(pixels) if siglip else base.encode_image(pixels)
        return nn.functional.normalize(features.float(),dim=1)
    return SimpleNamespace(model=base,device=torch.device('cpu'),preprocess=preprocess,
        processor=lambda images,return_tensors:dict(pixel_values=torch.stack([preprocess(im) for im in images])),
        encode_text=text,encode_images=images,similarity=lambda image,text:base.logit_scale.exp()*image@text.T,
        metadata=dict(spec=spec,resolved_revision='tiny-test-only',preprocessing='tiny28',weight_sha256='mock-only'))


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.config=json.loads((Path(__file__).parents[1]/'config.json').read_text())
        cls.config.update(partial_blocks=1,gradient_checkpointing=False,preservation_weight=.1)

    def test_development_never_opens_official_csv_or_shared_pair_getter(self):
        original=pd.read_csv
        def guard(path,*a,**kw):
            self.assertEqual(Path(path).resolve(),shared.TRAIN_CSV.resolve())
            return original(path,*a,**kw)
        with tempfile.TemporaryDirectory() as folder,patch('pandas.read_csv',side_effect=guard),              patch('common.dataset.get_pairs',side_effect=AssertionError('No shared all-pair getter')),              patch('common.dataset.get_crop_and_disease_names',side_effect=AssertionError('Use train names')):
            suite=Suite(self.config,'smoke',False,folder)
            self.assertEqual(suite.dev['pairs'].shape,(323,2))
            self.assertEqual((len(suite.dev['train']),len(suite.dev['validation'])),(7223,777))
            self.assertFalse(set(suite.dev['reduced'].pair_row)&set(suite.dev['holdout'].pair_row))
            self.assertFalse((suite.path/'development_sealed.json').exists())
            with self.assertRaises(RuntimeError):suite.save()

    def test_old_checkpoint_inputs_rejected(self):
        for key in ['e1_path','checkpoint_dir','score_caches','resume']:
            config=copy.deepcopy(self.config);config[key]='old'
            with self.assertRaises(ValueError):Suite(config)
        config=copy.deepcopy(self.config);config['backbones']['clip_b16']['pretrained']='old.pt'
        with self.assertRaises(ValueError):Suite(config)

    def test_final_metadata_requires_explicit_flag_and_preserves_order(self):
        dev=development();pairs,frames=final_data(dev)
        self.assertEqual(pairs.shape,(333,2));self.assertTrue(torch.equal(pairs[:323],dev['pairs']))
        self.assertEqual((len(frames['seen']),len(frames['unseen'])),(2000,1448))

    def test_heldout_training_targets_and_loss_are_masked(self):
        with self.assertRaises(ValueError):training_targets(torch.tensor([1]),[0,2])
        logits=[torch.randn(2,3,requires_grad=True),torch.randn(2,2,requires_grad=True),torch.randn(2,2,requires_grad=True)]
        loss_for('F4',logits,torch.tensor([0,2]),[0,2],torch.tensor([[0,0],[0,1],[1,1]]),[1,.5,.5]).backward()
        self.assertTrue(torch.equal(logits[0].grad[:,1],torch.zeros(2)))

    def test_weight_selection_uses_shared_topk_ties(self):
        pair=torch.zeros(2,323);crop=torch.zeros(2,2);disease=torch.zeros(2,2)
        columns=torch.zeros(323,2,dtype=torch.long)
        labels=pair.topk(1,dim=1).indices[:,0]
        result=choose_weights((pair,crop,disease),columns,labels,[[1,0,0]])
        self.assertEqual(result['history'][0]['correct'],2)

    def test_all_eight_methods_train_only_smoke_then_final_proxy(self):
        original=pd.read_csv
        def guard(path,*a,**kw):
            self.assertEqual(Path(path).resolve(),shared.TRAIN_CSV.resolve())
            return original(path,*a,**kw)
        config=copy.deepcopy(self.config);config['selected']=['R1','F2','R5','D3','R4','R3','F3','F4']
        with tempfile.TemporaryDirectory() as folder,patch('pandas.read_csv',side_effect=guard),              patch('approach3_experiment_suite.fresh.load_adapter',side_effect=tiny_adapter),              patch('approach3_dino_fusion.core.Dino',TinyDino),              patch('common.dataset.get_pairs',side_effect=AssertionError('No implicit candidates')),              patch('common.dataset.get_crop_and_disease_names',side_effect=AssertionError('No shared names')):
            suite=Suite(config,'smoke',False,folder);root=suite.run_selected()
            self.assertTrue((root/'development_sealed.json').is_file())
            for method in config['selected']:
                prepared=json.loads((root/method/'prepared.json').read_text())
                self.assertEqual(prepared['candidates'],323)
                self.assertFalse(prepared['official_test_csvs_read'])
            # Smoke final uses ONLY proxy images from train.csv, never official tests.
            final=final_evaluate(root)
            candidates=json.loads((final/'candidate_space.json').read_text())
            self.assertFalse(candidates['official_test_read'])
            self.assertEqual(len(original(final/'all_comparisons.csv')),16)
            with self.assertRaises(RuntimeError):suite.run('R1')
            # Any change to a frozen input must abort before test data access.
            (root/'effective_config.json').write_text('{}')
            with self.assertRaises(ValueError):final_evaluate(root)

    def test_prompt_and_siglip_expansion_preserves_learned_parameters(self):
        from approach3_experiment_suite.models import build_model
        dev=development();pairs,_=final_data(dev)
        for method in ['R4','R3','F3','F4']:
            spec=self.config['backbones']['siglip2' if method in ['F3','F4'] else 'clip_b16']
            adapter=tiny_adapter(spec,'cpu',False)
            old,_,_=banks_for(adapter,dev['pairs'],dev['names'],self.config)
            model=build_model(method,adapter,old,dev['pairs'],self.config,name_maps=dev['names'])
            before={k:v.detach().clone() for k,v in model.named_parameters()}
            banks,_,_=banks_for(adapter,pairs,dev['names'],self.config)
            expand_model(model,method,pairs,banks)
            self.assertTrue(all(torch.equal(v,dict(model.named_parameters())[k]) for k,v in before.items()))
            model.eval()
            with torch.no_grad():self.assertEqual(model(torch.randn(2,3,28,28))[0].shape,(2,333))


if __name__=='__main__':unittest.main()
