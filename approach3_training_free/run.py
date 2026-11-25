"""Five predeclared training-free experiments. No parameter search on tests."""
import argparse
import copy
import gc
import json
import time
from pathlib import Path

import pandas as pd
import torch

from common import config as shared
from common.dataset import get_pairs,get_crop_and_disease_names
from common.evaluate import robustness
from approach3_clip_improvements.approaches.clip_improvement_core import file_hash,seed_all
from approach3_ood_backbones.data import shared_frames
from approach3_ood_backbones.experiments import new_run,write_json,score_report,save_confusion
from approach3_ood_backbones.ood_evaluation import top1_predictions
from approach3_dino_fusion.core import Dino,prototypes,reproduction
from approach3_ood_backbones.adapters import versions
from approach3_training_free.models import FrozenCLIP
from approach3_training_free.cache import manifest,validate_cache,save_cache
from approach3_training_free.methods import validate,correct_bias,ensemble,map_components,fuse_component,diagnostics,component_accuracy


class Sequence:
    def __init__(self,config,mode='smoke',allow_download=False,output=None):
        validate(config)
        if mode not in ('smoke','full'):raise ValueError('Use smoke or full.')
        self.config=copy.deepcopy(config);self.smoke=mode=='smoke';self.allow_download=allow_download
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type=='cpu':torch.set_num_threads(4)
        seed_all(42)
        self.original=shared_frames();self.pairs=get_pairs()
        self.frames={k:(v.iloc[:2].copy() if self.smoke and k in ('seen','unseen') else v.copy()) for k,v in self.original.items()}
        self.path=new_run('sequence_'+mode,output or shared.RESULTS_DIR/'training_free')
        self.models={};self.model_metadata={};self.caches={};self.features=None;self.reports={};self.robust_verified=False
        # Lock EVERY variant before reading test predictions. No grid is searched.
        write_json(self.path/'protocol_frozen.json',dict(config=config,mode=mode,
            candidates=self.pairs.tolist(),versions=versions(),device=str(self.device),
            neural_training_images=0,prototype_source_images=7223,
            rules='No tuning on test labels; fixed prior coefficients/views; no E1 holdout tuning; immutable input caches.',
            variants=['E1','T1_bias','T2_crop','E3','E4','T3_ensemble','T4_tta','T5_symptoms','T5_disease','T5_symptoms_disease']))
        # Snapshot symptom content before any test-based diagnostics, not a later
        # lookup that could accidentally change after viewing results.
        self.symptoms=json.loads((shared.ROOT_DIR/'approach3_experiment_suite/symptoms.json').read_text())
        write_json(self.path/'symptoms_frozen.json',self.symptoms)
        self.seen_rows=sorted(set(self.original['train'].pair_row))

    def _path(self,value):
        value=Path(value)
        return value if value.is_absolute() else shared.ROOT_DIR/value

    def _model(self,key):
        if key not in self.models:
            if not self.smoke and self.device.type!='cuda':raise RuntimeError('Full neural inference requires CUDA; cached score-only evaluation can use CPU.')
            spec=self.config['models'][key]
            path=self._path(self.config['checkpoint_dir'])/(spec['name']+'.pt')
            self.models[key]=FrozenCLIP(path,self.original,self.device,spec)
            metadata=self.models[key].metadata
            if key not in self.model_metadata:
                write_json(self.path/(key+'_model.json'),metadata)
                self.model_metadata[key]=copy.deepcopy(metadata)
            elif metadata!=self.model_metadata[key]:
                raise RuntimeError('Reloaded model provenance changed within the frozen sequence.')
        return self.models[key]

    def _release_models(self):
        self.models.clear();gc.collect()
        if self.device.type=='cuda':torch.cuda.empty_cache()

    def _reference(self,key):
        return json.loads((Path(__file__).parent/'reference_results'/(key+'.json')).read_text())

    def _check_reproduction(self,key,scores):
        if self.smoke:return dict(passed=None,status='NOT_VERIFIED: two-image smoke is not full reproduction')
        actual=score_report(scores['seen'],scores['unseen'],self.frames,self.pairs)
        reference={k:v for k,v in self._reference(key).items() if k!='robustness'}
        check=reproduction(actual,reference,self.config['reproduction_tolerance'])
        write_json(self.path/(key+'_clean_reproduction.json'),check)
        if not check['passed']:raise RuntimeError(key+' clean reproduction failed; no fusion/ensemble proceeds.')
        return check

    def scores(self,key):
        if key in self.caches:return self.caches[key]['scores']
        requested=self.config['score_caches'][key];spec=self.config['models'][key]
        if requested:
            source=self._path(requested);before=file_hash(source)
            payload=torch.load(source,map_location='cpu',weights_only=True)
            validate_cache(payload,self.frames,self.pairs,self.smoke,'clip_scores',spec['sha256'])
            if file_hash(source)!=before:raise RuntimeError('Input cache changed.')
            write_json(self.path/(key+'_cache_source.json'),dict(path=str(source),sha256=before))
        else:
            model=self._model(key)
            payload=dict(**manifest(self.frames,self.pairs,self.smoke),kind='clip_scores',checkpoint_sha256=spec['sha256'],
                model_metadata=model.metadata,scores={s:model.scores(self.frames[s]) for s in ('seen','unseen')})
            validate_cache(payload,self.frames,self.pairs,self.smoke,'clip_scores',spec['sha256'])
        self._check_reproduction(key,payload['scores'])
        save_cache(self.path/(key+'_scores.pt'),payload)
        self.caches[key]=payload
        return payload['scores']

    def _verify_e1_robustness(self):
        if self.smoke or self.robust_verified:return
        # Preserve the existing formal D3 reproduction gate in addition to clean
        # metric checks. Cached T1/T3 do not require this extra image inference.
        model=self._model('E1');reference=self._reference('E1')
        actual=score_report(self.scores('E1')['seen'],self.scores('E1')['unseen'],self.frames,self.pairs)
        actual['robustness']=robustness(model.predict,model.loader(self.frames['seen']),actual['seen_top1'])
        check=reproduction(actual,reference,self.config['reproduction_tolerance'])
        write_json(self.path/'E1_robust_reproduction.json',check)
        if not check['passed']:raise RuntimeError('Original E1 robustness reproduction failed; DINO fusion stopped.')
        self.robust_verified=True

    def dino_features(self):
        if self.features is not None:return self.features
        frames={k:self.frames[k] for k in ('train','seen','unseen')}
        if self.smoke:
            # One image per TRAIN PAIR covers every crop and disease component;
            # crop-only sampling would omit some disease classes in T5 smoke.
            frames['train']=self.original['train'].groupby('pair_row',sort=True).head(1)
        requested=self.config['dino_cache']
        if requested:
            source=self._path(requested);before=file_hash(source)
            payload=torch.load(source,map_location='cpu',weights_only=True)
            validate_cache(payload,frames,self.pairs,self.smoke,'dino_features')
            if payload['metadata']['repository']!=self.config['dino_repository']:raise ValueError('Wrong DINO feature source.')
            revision=self.config['dino_revision']
            if revision!='main' and payload['metadata']['resolved_revision']!=revision:raise ValueError('DINO revision mismatch.')
            if file_hash(source)!=before:raise RuntimeError('Input feature cache changed.')
            write_json(self.path/'dino_cache_source.json',dict(path=str(source),sha256=before))
        else:
            if not self.smoke and self.device.type!='cuda':raise RuntimeError('Full DINO extraction requires CUDA.')
            model=Dino(self.config,self.device,self.allow_download)
            payload=dict(**manifest(frames,self.pairs,self.smoke),kind='dino_features',metadata=model.metadata,
                features={s:model.embeddings(frame,self.config['dino_batch_size']) for s,frame in frames.items()})
            validate_cache(payload,frames,self.pairs,self.smoke,'dino_features')
            del model
        save_cache(self.path/'dino_features.pt',payload)
        self.features=(payload,frames)
        return self.features

    def component_scores(self,column):
        payload,frames=self.dino_features();name='crop' if column==0 else 'disease'
        ids=torch.tensor(sorted(set(self.original['train'][name])))
        centers,counts=prototypes(payload['features']['train'],frames['train'][name].to_numpy(),ids)
        columns=map_components(self.pairs,ids,column)
        values={s:(payload['features'][s].to(self.device) @ centers.to(self.device).T).cpu() for s in ('seen','unseen')}
        save_cache(self.path/(name+'_prototypes.pt'),dict(prototypes=centers,ids=ids,counts=counts,pair_columns=columns,
            prototype_images=frames['train'].image.tolist(),training_images=len(frames['train']),smoke=self.smoke))
        direct={s:component_accuracy(values[s],self.frames[s][name].to_numpy(),ids,name) for s in values}
        write_json(self.path/(name+'_only_diagnostics.json'),direct)
        return values,columns

    def report(self,name,scores,recipe):
        if name in self.reports:return
        output=self.path/name;output.mkdir(exist_ok=False)
        result=score_report(scores['seen'],scores['unseen'],self.frames,self.pairs)
        result.update(variant=name,smoke=self.smoke,recipe=recipe,neural_training_images=0,
                      robustness=dict(status='not_run_for_variant; original E1 robustness gate separately recorded for T2'))
        write_json(output/'metrics.json',result)
        save_cache(output/'scores.pt',dict(pairs=self.pairs,scores=scores,
            labels={s:torch.tensor(self.frames[s].pair_row.to_numpy()) for s in ('seen','unseen')},
            images={s:self.frames[s].image.tolist() for s in ('seen','unseen')},smoke=self.smoke))
        save_confusion(output,name,scores['seen'],self.frames['seen'],self.pairs)
        pd.DataFrame([{k:v for k,v in row.items() if k!='per_class'} for row in result['ood_breakdown']]).to_csv(output/'ood_breakdown.csv',index=False,na_rep='N/A')
        self.reports[name]=result

    @torch.inference_mode()
    def run(self):
        start=time.perf_counter()
        e1=self.scores('E1');self.report('E1',e1,'Unmodified original E1 single-view logits')
        for step in self.config['steps']:
            if step=='T1':
                write_json(self.path/'score_diagnostics.json',{s:diagnostics(e1[s],self.seen_rows) for s in e1})
                self.report('T1_bias',{s:correct_bias(e1[s],self.seen_rows) for s in e1},'Fixed0.5 subtracted from seen-pair per-image zscores; no fitted bias')
            elif step=='T2':
                self._verify_e1_robustness();self._release_models()
                values,columns=self.component_scores(0)
                self.report('T2_crop',{s:fuse_component(e1[s],values[s],columns) for s in e1},'Fixed0.8 E1 +0.2 DINO crop zscores')
            elif step=='T3':
                for key in ('E3','E4'):
                    other=self.scores(key);self.report(key,other,'Original '+key+' checkpoint, original prompts/preprocessing')
                    self._release_models()
                self.report('T3_ensemble',{s:ensemble([self.scores(k)[s] for k in ('E1','E3','E4')]) for s in e1},'Equal mean of per-image zscores; no ensemble search')
            elif step=='T4':
                model=self._model('E1')
                self.report('T4_tta',{s:model.tta_scores(self.frames[s]) for s in e1},'Raw-logit mean: original view + horizontal flip; two forwards per image')
                del model
                self._release_models()
            elif step=='T5':
                self._verify_e1_robustness()
                model=self._model('E1');crops,diseases=get_crop_and_disease_names();prompts=[];covered=[]
                for index,(_,disease) in enumerate(self.pairs.tolist()):
                    text=model.prompts[index][0];entry=self.symptoms['descriptions'].get(str(disease))
                    if entry:
                        if entry['label']!=diseases[disease]:raise ValueError('Symptom label mismatch.')
                        text+='; '+entry['text'];covered.append(index)
                    prompts.append(text)
                write_json(self.path/'symptom_coverage.json',dict(covered_pairs=covered,described_diseases=len(self.symptoms['descriptions']),
                    fallback='original E1 prompt unchanged for other labels',prompts=prompts))
                symptom={s:model.prompted_scores(self.frames[s],prompts) for s in e1}
                self.report('T5_symptoms',symptom,'Frozen E1; six sourced disease descriptions; name-only fallback')
                del model
                self._release_models();values,columns=self.component_scores(1)
                self.report('T5_disease',{s:fuse_component(e1[s],values[s],columns) for s in e1},'Fixed0.8 E1 +0.2 DINO disease zscores')
                self.report('T5_symptoms_disease',{s:fuse_component(symptom[s],values[s],columns) for s in e1},'Predeclared combined symptom + disease-prototype ablation, fixed0.8/0.2')
        keys=['seen_top1','seen_top5','unseen_top1','unseen_top5','harmonic_mean','crop_acc','disease_acc','unseen_crop_top1','unseen_disease_top1','unseen_combination_top1']
        pd.DataFrame([dict(variant=name,**{k:r[k] for k in keys}) for name,r in self.reports.items()]).to_csv(self.path/'comparison.csv',index=False,na_rep='N/A')
        write_json(self.path/'comparison.json',dict(reports=self.reports,smoke=self.smoke,elapsed_seconds=time.perf_counter()-start,
            note='Fixed recipes, exploratory official-test comparisons, not newly blind confirmatory results. Different inference budgets disclosed.'))
        # Descriptive AFTER frozen predictions; never feeds coefficients/prompts.
        errors={}
        for split in ('seen','unseen'):
            truth=torch.tensor(self.frames[split].pair_row.to_numpy())
            base=top1_predictions(e1[split])==truth
            for name in self.reports:
                if name=='E1':continue
                cache=torch.load(self.path/name/'scores.pt',map_location='cpu',weights_only=True)
                correct=top1_predictions(cache['scores'][split])==truth
                errors[name+'_'+split]=dict(both_correct=int((base&correct).sum()),rescued=int((~base&correct).sum()),
                    harmed=int((base&~correct).sum()),both_wrong=int((~base&~correct).sum()),use='post-hoc description only; not tuning')
        write_json(self.path/'error_overlap.json',errors)
        self._release_models()
        return self.path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).parent/'config.json')
    parser.add_argument('--steps',nargs='+',choices=['T1','T2','T3','T4','T5'])
    parser.add_argument('--mode',choices=['smoke','full'],default='smoke')
    parser.add_argument('--allow-download',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();config=json.loads(args.config.read_text())
    if args.steps:config['steps']=args.steps
    sequence=Sequence(config,args.mode,args.allow_download,args.output)
    try:print('Saved:',sequence.run())
    except Exception as error:
        write_json(sequence.path/'error.json',dict(error=type(error).__name__,message=str(error),no_automatic_setting_changes=True))
        raise


if __name__=='__main__':main()
