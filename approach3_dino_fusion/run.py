"""Six-step frozen inference experiment: smoke, verify E1, then full fusion."""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch

from common import config as cfg
from common.evaluate import robustness
from approach3_clip_improvements.approaches.clip_improvement_core import file_hash,seed_all
from approach3_ood_backbones.data import shared_frames
from approach3_ood_backbones.experiments import new_run,write_json,score_report,save_confusion
from approach3_dino_fusion.core import (
    E1,Dino,canonical_hash,prototypes,pair_crop_columns,fuse,crop_accuracy,reproduction,
)


def check_config(config):
    if config['seed']!=42 or config['clip_fraction']!=0.8 or config['dino_fraction']!=0.2:
        raise ValueError('Fixed seed42 and fusion0.8/0.2 contract changed; do not tune this run.')
    if config['prototype_source']!='original 7223-image optimization-training partition only':
        raise ValueError('Only the original optimization-training images may form prototypes.')
    if config['dino_repository']!='facebook/dinov2-base' or config['dino_pooling']!='last_hidden_state[:, 0, :] (CLS)':
        raise ValueError('Unexpected DINO architecture or pooling.')
    if config['reproduction_tolerance_percentage_points']!=0.005:
        raise ValueError('Do not loosen E1 reproduction tolerance.')


def require_reproduction(record):
    if not record['passed']:
        raise RuntimeError('E1 reproduction failed. Fusion aborted; inspect reproduction.json. No weights or settings were adjusted.')


def execute(config,e1_path,stage='smoke',allow_download=False,output=None):
    check_config(config)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if stage not in ('smoke','verify','full'):
        raise ValueError('Stage must be smoke, verify or full.')
    if stage!='smoke' and device.type!='cuda':
        raise RuntimeError('Verification/full extraction require the school CUDA GPU. Use smoke locally.')
    seed_all(42)
    if device.type=='cpu': torch.set_num_threads(4)
    frames=shared_frames()
    path=new_run('dino_e1_'+stage,output or cfg.RESULTS_DIR/'dino_e1_fusion')
    # Freeze all settings before any official test prediction or reproduction.
    write_json(path/'protocol_frozen.json',dict(config=config,stage=stage,
        device=str(device),gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else None,
        prototype_images=7223,neural_training_images=0,
        splits={k:dict(samples=len(v),sha256=canonical_hash(v)) for k,v in frames.items()},
        note='No coefficient search, no optimization; E1 reproduction is a pass/fail gate.'))
    started=time.perf_counter()
    try:
        e1=E1(Path(e1_path),frames,device,config['e1_sha256'])
        reference_path=Path(__file__).parent/'reference_results/E1_wise.json'
        reference=json.loads(reference_path.read_text(encoding='utf-8'))
        write_json(path/'e1_metadata.json',e1.metadata)
        smoke=stage=='smoke'
        evaluation_frames={k:(v.iloc[:2] if smoke and k!='train' else v) for k,v in frames.items()}
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats()
        seen_scores=e1.scores(evaluation_frames['seen'])
        unseen_scores=e1.scores(evaluation_frames['unseen'])
        baseline=score_report(seen_scores,unseen_scores,evaluation_frames,e1.pairs)
        if not smoke:
            baseline['robustness']=robustness(e1.predict,e1.loader(frames['seen']),baseline['seen_top1'])
            check=reproduction(baseline,reference,config['reproduction_tolerance_percentage_points'])
            check.update(reference_sha256=file_hash(reference_path),checkpoint_sha256=e1.source_hash)
            write_json(path/'reproduction.json',check)
            write_json(path/'E1_reproduced.json',baseline)
            require_reproduction(check)
        else:
            write_json(path/'reproduction.json',dict(passed=None,status='NOT_VERIFIED',
                reason='Tiny smoke set cannot reproduce the full recorded E1 metrics. Full fusion remains gated.'))
        if stage=='verify':
            print('E1 reproduction passed. Saved:',path)
            return path
        # DINO is loaded only AFTER E1 reproduction passes, except explicitly
        # non-formal smoke. Neither backbone receives gradients or optimizer steps.
        dino=Dino(config,device,allow_download)
        write_json(path/'dino_metadata.json',dino.metadata)
        train=frames['train']
        if smoke:
            train=train.groupby('crop',sort=True).head(1)
        train_features=dino.embeddings(train,config['dino_batch_size'])
        crop_ids=torch.tensor(sorted(set(frames['train'].crop)))
        centers,counts=prototypes(train_features,train.crop.to_numpy(),crop_ids)
        columns=pair_crop_columns(e1.pairs,crop_ids)
        # No official/validation images appear in this data-dependent model.
        with (path/'crop_prototypes.pt').open('xb') as stream:
            torch.save(dict(prototypes=centers,crop_ids=crop_ids,counts=counts,
                            pair_crop_columns=columns,pairs=e1.pairs,
                            train_image_names=train.image.tolist(),train_sha256=canonical_hash(train),
                            smoke=smoke,method='mean of unit CLS embeddings, then normalize'),stream)
        # Persist a reproducible feature cache only in this NEW run directory.
        with (path/'train_dino_embeddings.pt').open('xb') as stream:
            torch.save(dict(embeddings=train_features,crop_ids=torch.tensor(train.crop.to_numpy()),
                            image_names=train.image.tolist(),smoke=smoke),stream)
        scores={};dino_report={};predictions=[]
        for split,clip_scores in [('seen',seen_scores),('unseen',unseen_scores)]:
            frame=evaluation_frames[split]
            features=dino.embeddings(frame,config['dino_batch_size'])
            # Cosine inference stays on the selected GPU; CPU copies are for reports.
            cosine=(features.to(device) @ centers.to(device).T).cpu()
            scores[split]=fuse(clip_scores,cosine,columns,config['clip_fraction'])
            dino_report[split]=crop_accuracy(cosine,frame.crop.to_numpy(),crop_ids)
            with (path/(split+'_scores.pt')).open('xb') as stream:
                torch.save(dict(image_names=frame.image.tolist(),pair_labels=torch.tensor(frame.pair_row.to_numpy()),
                                clip_pair_scores=clip_scores,dino_crop_scores=cosine,
                                fused_pair_scores=scores[split],pairs=e1.pairs,crop_ids=crop_ids),stream)
            for i,row in enumerate(frame.itertuples()):
                predictions.append(dict(split=split,image=row.image,true_pair=row.pair_row,true_crop=row.crop,
                    e1_pair=int(clip_scores[i].argmax()),dino_crop=int(crop_ids[cosine[i].argmax()]),
                    fusion_pair=int(scores[split][i].argmax())))
        fusion=score_report(scores['seen'],scores['unseen'],evaluation_frames,e1.pairs)
        baseline['smoke']=smoke;fusion['smoke']=smoke
        write_json(path/'E1.json',baseline);write_json(path/'DINO_crop.json',dino_report)
        write_json(path/'E1_DINO_fusion.json',fusion)
        save_confusion(path,'E1',seen_scores,evaluation_frames['seen'],e1.pairs)
        save_confusion(path,'E1_DINO_fusion',scores['seen'],evaluation_frames['seen'],e1.pairs)
        pd.DataFrame(predictions).to_csv(path/'predictions.csv',index=False)
        rows=[]
        keys=['seen_top1','seen_top5','unseen_top1','unseen_top5','harmonic_mean','crop_acc','disease_acc',
              'unseen_crop_samples','unseen_crop_top1','unseen_disease_samples','unseen_disease_top1',
              'unseen_combination_samples','unseen_combination_top1','unseen_combination_top5']
        for name,report in [('E1',baseline),('E1_DINO_fusion',fusion)]:
            rows.append(dict(model=name,**{k:report[k] for k in keys},
                             seen_crop_only_top1=None,unseen_split_crop_only_top1=None))
        rows.append(dict(model='DINO_crop_prototypes',**{k:None for k in keys},
                         seen_crop_only_top1=dino_report['seen']['crop_top1'],
                         unseen_split_crop_only_top1=dino_report['unseen']['crop_top1']))
        pd.DataFrame(rows).to_csv(path/'comparison.csv',index=False,na_rep='N/A')
        write_json(path/'comparison.json',dict(E1=baseline,DINO_crop=dino_report,fusion=fusion,
            smoke=smoke,prototype_images=len(train),neural_training_images=0,
            seconds=time.perf_counter()-started,
            peak_gpu_memory_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else None,
                limitation='DINO crop scores cannot resolve diseases within the same crop. Weight fixed at 0.2.'))
        breakdown_rows=[dict(model=name,**{k:v for k,v in row.items() if k!='per_class'})
                        for name,report in [('E1',baseline),('E1_DINO_fusion',fusion)] for row in report['ood_breakdown']]
        pd.DataFrame(breakdown_rows).to_csv(path/'ood_breakdown.csv',index=False,na_rep='N/A')
        if file_hash(Path(e1_path))!=config['e1_sha256']:
            raise RuntimeError('Source E1 checkpoint changed during this run.')
        if any(p.requires_grad or p.grad is not None for model in (e1.model,dino.model) for p in model.parameters()):
            raise RuntimeError('Frozen inference contract violated.')
        print('Saved:',path)
        return path
    except Exception as error:
        write_json(path/'error.json',dict(type=type(error).__name__,message=str(error),settings_reduced=False))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--e1',type=Path,required=True)
    parser.add_argument('--stage',choices=['smoke','verify','full'],default='smoke')
    parser.add_argument('--allow-download',action='store_true',help='Explicitly allow DINO pretrained weight download.')
    parser.add_argument('--config',type=Path,default=Path(__file__).parent/'config.json')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    execute(json.loads(args.config.read_text(encoding='utf-8')),args.e1,args.stage,args.allow_download,args.output)


if __name__=='__main__':
    main()
