"""Train/select on development splits, freeze selection, then evaluate once."""
import gc
import json
import time
from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.nn import functional as F

from common import config as shared
from common.dataset import get_pairs,get_crop_and_disease_names
from approach3_ood_backbones.adapters import load_adapter
from approach3_ood_backbones.data import shared_frames,compositional_split
from approach3_ood_backbones.experiments import text_bank,collect,score_report,save_confusion,write_json
from approach3_ood_backbones.factorized import combine,select_weights
from approach3_experiment_suite.protocol import (
    class_space,training_targets,select_bias,bias_scores,accumulation_divisor,
)
from approach3_experiment_suite.models import build_model


def cpu_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def image_batches(frame,adapter,size,siglip=False,shuffle=False,seed=42):
    order=torch.randperm(len(frame),generator=torch.Generator().manual_seed(seed)).tolist() if shuffle else list(range(len(frame)))
    for start in range(0,len(order),size):
        batch=frame.iloc[order[start:start+size]];images=[]
        for name in batch.image:
            with Image.open(shared.IMAGE_DIR/name) as image:images.append(image.convert('RGB'))
        pixels=adapter.processor(images=images,return_tensors='pt')['pixel_values'] if siglip else torch.stack([adapter.preprocess(image) for image in images])
        yield pixels.to(adapter.device),torch.tensor(batch.pair_row.to_numpy(),device=adapter.device)


@torch.no_grad()
def model_scores(model,adapter,frame,config,siglip):
    model.eval();result=[[],[],[]]
    for pixels,_ in image_batches(frame,adapter,config['batch_size'],siglip):
        for store,scores in zip(result,model(pixels)):
            if scores is not None:store.append(scores.float().cpu())
    return tuple(torch.cat(s) if s else None for s in result)


def inference(method,components,columns,weights=None):
    pair,crop,disease=components
    if method=='F4':return combine(components,columns,weights or [1,0,0])
    if method=='R3':
        # Troika-style inference combines composition logits with the product
        # of primitive probabilities; no independent closed-set disease head.
        return pair+crop.softmax(1)[:,columns[:,0]]*disease.softmax(1)[:,columns[:,1]]
    return pair


def loss_for(method,logits,labels,train_rows,columns,weights):
    target=training_targets(labels,train_rows)
    pair_loss=F.cross_entropy(logits[0][:,train_rows],target)
    if method in ('F4','R3'):
        concepts=columns.to(labels.device)[labels]
        return weights[0]*pair_loss+weights[1]*F.cross_entropy(logits[1],concepts[:,0])+weights[2]*F.cross_entropy(logits[2],concepts[:,1])
    return pair_loss


def train(method,adapter,banks,pairs,config,train_frame,val_frame,path,smoke):
    siglip=method in ('F3','F4')
    model=build_model(method,adapter,banks,pairs,config).to(adapter.device)
    _,_,columns=class_space(pairs)
    train_rows=sorted(set(train_frame.pair_row))
    names=[name for name,p in model.named_parameters() if p.requires_grad]
    if not names:raise ValueError('No trainable parameters.')
    if any('text_model.' in name for name in names):raise ValueError('SigLIP2 text tower must remain frozen.')
    rate=config['prompt_lr'] if method in ('R3','R4') else config['lr']
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=rate,weight_decay=config['weight_decay'])
    initial=cpu_state(model)
    def epoch(data,index,mechanics=False):
        model.train();model.base.eval()
        if siglip:
            # HF gradient checkpointing is gated on vision-model training mode.
            # Keep frozen text in eval, but activate the visual checkpoint path.
            model.base.vision_model.train()
        if getattr(model,'teacher',None) is not None:model.teacher.eval()
        optimizer.zero_grad(set_to_none=True);total=0.;count=0
        batches=(len(data)+config['batch_size']-1)//config['batch_size']
        for step,(pixels,labels) in enumerate(image_batches(data,adapter,config['batch_size'],siglip,True,42+index)):
            if len(labels)==1:
                for module in model.modules():
                    if isinstance(module,torch.nn.BatchNorm1d):module.eval()
            amp=(torch.autocast('cuda',dtype=torch.bfloat16) if adapter.device.type=='cuda' and config['precision']=='bf16' else nullcontext())
            with amp:
                logits=model(pixels)
                value=loss_for(method,logits,labels,train_rows,columns,config['loss_weights'])
                if siglip:value=value+config['preservation_weight']*model.preservation(pixels)
            if not torch.isfinite(value):raise RuntimeError('Nonfinite training loss.')
            (value/accumulation_divisor(step,batches,config['accumulation_steps'])).backward()
            if (step+1)%config['accumulation_steps']==0 or step+1==batches:
                grads=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
                if not grads or not all(torch.isfinite(g).all() for g in grads):raise RuntimeError('Missing/nonfinite gradients.')
                optimizer.step();optimizer.zero_grad(set_to_none=True)
            total+=float(value.detach())*len(labels);count+=len(labels)
        return total/count
    # Every model gets a real update preflight before expensive full training.
    # Restore initial weights and optimizer so the smoke update never contaminates
    # the formal initial state. No epoch/BF16/batch reduction occurs silently.
    probe=train_frame.iloc[:max(2,min(config['batch_size'],len(train_frame)))].copy()
    probe_loss=epoch(probe,0,True)
    model.load_state_dict(initial,strict=True)
    optimizer.state.clear();optimizer.zero_grad(set_to_none=True)
    torch.manual_seed(42)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(42)
    write_json(path/'training_preflight.json',dict(loss=probe_loss,restored_initial_state=True,trainable_names=names,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
    del initial
    history=[];best=-1.;best_state=None;best_path=None
    epochs=1 if smoke else config['epochs']
    for index in range(epochs):
        start=time.perf_counter();value=epoch(train_frame,index)
        components=model_scores(model,adapter,val_frame,config,siglip)
        scores=inference(method,components,columns)
        labels=torch.tensor(val_frame.pair_row.to_numpy())
        accuracy=100*(scores.argmax(1)==labels).float().mean().item()
        history.append(dict(epoch=index+1,loss=value,compositional_validation_top1=accuracy,seconds=time.perf_counter()-start))
        print(method,history[-1],flush=True)
        if accuracy>best:
            best=accuracy;best_state=cpu_state(model)
            best_path=path/f'checkpoint_epoch_{index+1:03d}.pt'
            with best_path.open('xb') as stream:
                torch.save(dict(state_dict=best_state,method=method,config=config,pairs=pairs,
                    train_pair_rows=train_rows,history=list(history),smoke=smoke,train_images=len(train_frame),
                    selection='compositional validation only',initialization='pristine pretrained backbone, never E1'),stream)
    model.load_state_dict(best_state,strict=True)
    # Immutable recovery checkpoints are saved immediately on improvement.
    # Create a best alias without overwriting any checkpoint or duplicating
    # several GB when the filesystem supports hard links.
    import os,shutil
    try:os.link(best_path,path/'best.pt')
    except OSError:
        with best_path.open('rb') as source,(path/'best.pt').open('xb') as target:shutil.copyfileobj(source,target)
    write_json(path/'best_checkpoint.json',dict(checkpoint=best_path.name,best_validation_top1=best,history=history))
    # Round-trip verifies that the saved selected checkpoint really loads.
    saved=torch.load(path/'best.pt',map_location='cpu',weights_only=True)
    if not torch.equal(saved['pairs'],pairs):raise ValueError('Saved pair ordering changed.')
    model.load_state_dict(saved['state_dict'],strict=True)
    selection=dict(best_validation_top1=best,history=history,train_images=len(train_frame),
                   trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    if method=='F4':
        components=model_scores(model,adapter,val_frame,config,True)
        selection['factorized']=select_weights(components,columns,val_frame.pair_row.to_numpy(),config['score_grid'])
    return model,selection


def symptom_banks(adapter,config,pairs):
    source=json.loads((Path(__file__).parent/'symptoms.json').read_text(encoding='utf-8'))
    crops,diseases=get_crop_and_disease_names();descriptions=source['descriptions']
    prompts=[];covered=0
    for crop,disease in pairs.tolist():
        text=f'a photo of a {crops[crop].replace("_"," ")} leaf with {diseases[disease].replace("_"," ")}'
        row=descriptions.get(str(disease))
        if row:
            if row['label']!=diseases[disease] or not row['source'].startswith('https://'):raise ValueError('Symptom mapping/source mismatch.')
            text+='; '+row['text'];covered+=1
        prompts.append(text)
    chunk=config['text_batch_size']
    bank=torch.cat([adapter.encode_text(prompts[i:i+chunk]) for i in range(0,len(prompts),chunk)])
    return bank,dict(described_diseases=len(descriptions),total_diseases=len(diseases),covered_pairs=covered,
                     fallback='unlisted disease labels retain name-only prompts',prompts=prompts,sources=source)


def execute_native(method,config,path,smoke,allow_download):
    frames=shared_frames();pairs=get_pairs()
    reduced,holdout,manifest=compositional_split(frames['train'],config['pair_holdout_fraction'],42)
    write_json(path/'split_manifest.json',manifest)
    reduced.to_csv(path/'reduced_train.csv',index=False);holdout.to_csv(path/'compositional_validation.csv',index=False)
    _,_,columns=class_space(pairs)
    model=None;adapter=None
    try:
        specs=json.loads((shared.ROOT_DIR/'approach3_ood_backbones/configs/default.json').read_text())
        spec=specs['models']['siglip2' if method in ('F3','F4') else 'clip_b16']
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not smoke and device.type!='cuda':raise RuntimeError('Full runs require CUDA.')
        if device.type=='cpu':torch.set_num_threads(4)
        if device.type=='cuda' and config['precision']=='bf16' and not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 unsupported; choose float32 explicitly.')
        torch.manual_seed(42)
        adapter=load_adapter(spec,device,allow_download)
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
        banks,_,_=text_bank(adapter,specs)
        write_json(path/'model_metadata.json',adapter.metadata)
        check=collect(adapter,banks,frames['validation'].iloc[:2],2)
        if check[0].shape!=(2,333) or not torch.isfinite(check[0]).all():raise RuntimeError('Model inference preflight failed.')
        selection={};coverage=None
        if method in ('F3','F4','R3','R4'):
            training_frame=reduced.iloc[:2] if smoke else reduced
            validation_frame=holdout.iloc[:2] if smoke else holdout
            model,selection=train(method,adapter,banks,pairs,config,training_frame,validation_frame,path,smoke)
        elif method=='R1':
            seenval=frames['validation'][frames['validation'].pair_row.isin(reduced.pair_row)]
            novelval=holdout
            if smoke:seenval=seenval.iloc[:2];novelval=novelval.iloc[:2]
            a=collect(adapter,banks,seenval,config['batch_size'])[0]
            b=collect(adapter,banks,novelval,config['batch_size'])[0]
            selection=select_bias(a,torch.tensor(seenval.pair_row.to_numpy()),b,torch.tensor(novelval.pair_row.to_numpy()),sorted(set(reduced.pair_row)),config['bias_grid'])
        elif method=='R5':
            symptom,coverage=symptom_banks(adapter,config,pairs)
            write_json(path/'symptom_coverage.json',coverage)
        write_json(path/'selection_frozen.json',dict(method=method,selection=selection,config=config,
            optimization_partition_images=len(reduced) if model is not None else 0,
            membership_reference='full intended reduced-training partition' if model is not None else 'original task training partition',
            no_test_selection=True))
        evaluation_frames={k:(v.iloc[:2] if smoke and k!='train' else v) for k,v in frames.items()}
        if model is not None:
            # Keep official Seen/Unseen split metrics, but derive OOD categories
            # from this model's actual intended training membership. Some of the
            # official Seen pairs are now held out, so they must count as novel.
            evaluation_frames['train']=reduced
        predictions={};raw={}
        for split in ('seen','unseen'):
            frame=evaluation_frames[split]
            if model is not None:
                components=model_scores(model,adapter,frame,config,method in ('F3','F4'))
                weights=selection.get('factorized',{}).get('weights')
                predictions[split]=inference(method,components,columns,weights)
                raw[split]=components[0]
            elif method=='R1':
                raw[split]=collect(adapter,banks,frame,config['batch_size'])[0]
                predictions[split]=bias_scores(raw[split],sorted(set(frames['train'].pair_row)),selection['bias'])
            else:
                raw[split]=collect(adapter,[banks[0]],frame,config['batch_size'])[0]
                predictions[split]=collect(adapter,[symptom],frame,config['batch_size'])[0]
        report=score_report(predictions['seen'],predictions['unseen'],evaluation_frames,pairs)
        report.update(method=method,smoke=smoke,selection=selection,
                      neural_training_images=len(reduced) if model is not None and not smoke else (2 if model is not None else 0),
                      peak_gpu_memory_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else None,
                      robustness={'status':'not_run; clean comparison only'})
        write_json(path/'report.json',report)
        baseline=score_report(raw['seen'],raw['unseen'],evaluation_frames,pairs)
        write_json(path/'pair_only_or_name_only.json',baseline)
        save_confusion(path,method,predictions['seen'],evaluation_frames['seen'],pairs)
        keys=['seen_top1','seen_top5','unseen_top1','unseen_top5','harmonic_mean','crop_acc','disease_acc','unseen_crop_top1','unseen_disease_top1','unseen_combination_top1']
        pd.DataFrame([dict(variant='pair/name-only',**{k:baseline[k] for k in keys}),dict(variant=method,**{k:report[k] for k in keys})]).to_csv(path/'comparison.csv',index=False,na_rep='N/A')
        pd.DataFrame([{k:v for k,v in row.items() if k!='per_class'} for row in report['ood_breakdown']]).to_csv(path/'ood_breakdown.csv',index=False,na_rep='N/A')
        with (path/'scores.pt').open('xb') as stream:torch.save(dict(scores=predictions,pairs=pairs,smoke=smoke),stream)
        return report
    finally:
        del model,adapter
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
