import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from common.dataset import get_pairs
from common.evaluate import top_k_accuracy
from approach3_ood_backbones.data import shared_frames
from approach3_training_free.methods import validate,correct_bias,ensemble,map_components,fuse_component,diagnostics,component_accuracy
from approach3_training_free.cache import manifest,validate_cache,save_cache
from approach3_training_free.run import Sequence
from approach3_training_free.models import FrozenCLIP
from approach3_training_free.import_e1_cache import convert
from approach3_dino_fusion.core import canonical_hash
from common.dataset import get_transforms


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.config=json.loads((Path(__file__).parents[1]/'config.json').read_text())
        cls.frames=shared_frames();cls.pairs=get_pairs()

    def test_fixed_recipes_reject_tuning(self):
        for key,value in [('bias',1.),('clip_fraction',.5),('ensemble_weights',[.5,.25,.25]),('tta_views',['identity'])]:
            changed=copy.deepcopy(self.config);changed[key]=value
            with self.assertRaises(ValueError):validate(changed)

    def payload(self,frames=None):
        frames=frames or self.frames
        return dict(**manifest(frames,self.pairs,False),kind='clip_scores',checkpoint_sha256='expected',
                    scores={s:torch.zeros(len(frames[s]),333) for s in ('seen','unseen')})

    def test_cache_candidate_and_image_order(self):
        payload=self.payload();validate_cache(payload,self.frames,self.pairs,False,'clip_scores','expected')
        wrong=copy.deepcopy(payload);wrong['pairs']=wrong['pairs'].flip(0)
        with self.assertRaises(ValueError):validate_cache(wrong,self.frames,self.pairs,False,'clip_scores','expected')
        wrong=copy.deepcopy(payload);wrong['frames']['unseen']['images'].reverse()
        with self.assertRaises(ValueError):validate_cache(wrong,self.frames,self.pairs,False,'clip_scores','expected')

    def test_cache_smoke_or_10_candidates_rejected(self):
        payload=self.payload();payload['scores']['unseen']=torch.zeros(1448,10)
        with self.assertRaises(ValueError):validate_cache(payload,self.frames,self.pairs,False,'clip_scores','expected')
        payload=self.payload();payload['smoke']=True
        with self.assertRaises(ValueError):validate_cache(payload,self.frames,self.pairs,False,'clip_scores','expected')

    def test_component_mapping_uses_verified_ids(self):
        pairs=torch.tensor([[9,7],[3,2],[9,2]])
        self.assertEqual(map_components(pairs,[3,9],0).tolist(),[1,0,1])
        self.assertEqual(map_components(pairs,[2,7],1).tolist(),[1,0,0])
        with self.assertRaises(ValueError):map_components(pairs,[2],1)

    def test_recipe_math_and_no_gradient_updates(self):
        values=[torch.randn(3,333) for _ in range(3)]
        originals=[v.clone() for v in values]
        mixed=ensemble(values);self.assertEqual(tuple(mixed.shape),(3,333))
        corrected=correct_bias(values[0],range(323))
        fused=fuse_component(values[0],torch.randn(3,55),self.pairs[:,0])
        self.assertFalse(any(v.requires_grad for v in [mixed,corrected,fused]))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(values,originals)))

    def test_diagnostics_never_needs_test_labels(self):
        result=diagnostics(torch.ones(4,333),range(323))
        self.assertEqual(result['samples'],4)
        self.assertIn('label-free',result['scope'])

    def test_direct_component_ties_use_topk1(self):
        scores=torch.zeros(3,55);ids=torch.arange(55)*2
        labels=ids[scores.topk(1,dim=1).indices[:,0]]
        result=component_accuracy(scores,labels,ids,'crop')
        self.assertEqual(result['crop_only_top1'],100.)
        self.assertIsNone(result['pair_top1'])

    def test_wrong_checkpoint_cache_rejected(self):
        with self.assertRaises(ValueError):validate_cache(self.payload(),self.frames,self.pairs,False,'clip_scores','wrong')

    def test_formal_reproduction_failure_stops_before_correction(self):
        with tempfile.TemporaryDirectory() as temporary:
            run=Sequence(self.config,'full',False,temporary)
            wrong={s:torch.zeros(len(self.frames[s]),333) for s in ('seen','unseen')}
            with self.assertRaisesRegex(RuntimeError,'reproduction failed'):run._check_reproduction('E1',wrong)
            self.assertFalse(run.reports)
            self.assertFalse(json.loads((run.path/'E1_clean_reproduction.json').read_text())['passed'])

    def test_tta_is_exact_two_view_raw_logit_mean(self):
        model=FrozenCLIP.__new__(FrozenCLIP);images=torch.arange(24).reshape(2,3,2,2).float()
        calls=[]
        def predict(value):
            calls.append(value.clone())
            return value[:,:,:,:1].sum((1,2,3))[:,None]*torch.arange(333)[None]
        model.predict=predict;model.loader=lambda frame:[(images,None,None,None)]
        result=model.tta_scores(None)
        expected=((images[:,:,:,:1].sum((1,2,3))+images.flip(-1)[:,:,:,:1].sum((1,2,3)))/2)[:,None]*torch.arange(333)[None]
        torch.testing.assert_close(result,expected)
        self.assertEqual(len(calls),2)
        self.assertTrue(torch.equal(calls[0],images))
        self.assertTrue(torch.equal(calls[1],images.flip(-1)))

    def test_prior_d3_import_provenance_and_image_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            source=Path(temporary)/'old_run';source.mkdir()
            metadata=dict(checkpoint_sha256=self.config['models']['E1']['sha256'],candidate_pairs=self.pairs.tolist(),
                          preprocessing=repr(get_transforms('clip')[1]))
            protocol=dict(stage='full',splits={s:dict(sha256=canonical_hash(f)) for s,f in self.frames.items()})
            for name,value in [('protocol_frozen.json',protocol),('e1_metadata.json',metadata),('reproduction.json',{'passed':True})]:
                (source/name).write_text(json.dumps(value))
            for split in ('seen','unseen'):
                payload=dict(image_names=self.frames[split].image.tolist(),pair_labels=torch.tensor(self.frames[split].pair_row.to_numpy()),
                             pairs=self.pairs,clip_pair_scores=torch.zeros(len(self.frames[split]),333))
                save_cache(source/(split+'_scores.pt'),payload)
            with patch('approach3_training_free.import_e1_cache.reproduction',return_value={'passed':True}):
                target=convert(source,Path(temporary)/'imported.pt')
                self.assertTrue(target.is_file())
                cached=torch.load(source/'unseen_scores.pt',weights_only=True)
                cached['image_names'].reverse()
                torch.save(cached,source/'unseen_scores.pt')
                with self.assertRaisesRegex(ValueError,'image order'):
                    convert(source,Path(temporary)/'must_not_exist.pt')

    def test_cache_only_full_T1_no_models_and_shared_metrics(self):
        config=copy.deepcopy(self.config)
        with tempfile.TemporaryDirectory() as temporary:
            payload=dict(**manifest(self.frames,self.pairs,False),kind='clip_scores',checkpoint_sha256=config['models']['E1']['sha256'],
                scores={s:torch.randn(len(self.frames[s]),333) for s in ('seen','unseen')})
            source=Path(temporary)/'input.pt';save_cache(source,payload);before=source.read_bytes()
            config['score_caches']['E1']=str(source)
            with patch('approach3_training_free.run.Sequence._check_reproduction',return_value={'passed':True}), \
                 patch('approach3_training_free.run.Sequence._model',side_effect=AssertionError('Must not load weights for cached T1')):
                run=Sequence(config,'full',False,temporary);output=run.run()
                report=json.loads((output/'T1_bias/metrics.json').read_text())
                cached=torch.load(output/'T1_bias/scores.pt',weights_only=True)
                for split in ('seen','unseen'):
                    self.assertEqual(report[split+'_top1'],top_k_accuracy(cached['scores'][split],cached['labels'][split],1))
                self.assertEqual(report['unseen']['samples'],1448)
                self.assertIsNone(report['unseen_crop_top1'])
            self.assertEqual(source.read_bytes(),before)

    def test_smoke_all_steps_with_mock_backbones(self):
        config=copy.deepcopy(self.config);config['steps']=['T1','T2','T3','T4','T5']
        class FakeCLIP:
            def __init__(self,*args):
                self.model=torch.nn.Linear(2,2).eval().requires_grad_(False)
                self.metadata={'scope':'mock mechanics only'};self.prompts=[['original prompt'] for _ in range(333)]
            def scores(self,frame):return torch.arange(333).float().repeat(len(frame),1)
            def tta_scores(self,frame):return self.scores(frame)
            def prompted_scores(self,frame,prompts):return self.scores(frame)
        class FakeDino:
            def __init__(self,*args):self.metadata={'repository':'facebook/dinov2-base','resolved_revision':'mock'}
            def embeddings(self,frame,batch_size):return torch.nn.functional.normalize(torch.arange(len(frame)*8).reshape(len(frame),8).float()+1,dim=1)
        with tempfile.TemporaryDirectory() as temporary,patch('approach3_training_free.run.FrozenCLIP',FakeCLIP),patch('approach3_training_free.run.Dino',FakeDino):
            run=Sequence(config,'smoke',False,temporary);path=run.run()
            self.assertEqual(len(run.reports),10)
            self.assertTrue((path/'comparison.csv').is_file())
            self.assertTrue((path/'disease_prototypes.pt').is_file())
            self.assertEqual(run.reports['E1']['unseen']['samples'],2)


if __name__=='__main__':unittest.main()
