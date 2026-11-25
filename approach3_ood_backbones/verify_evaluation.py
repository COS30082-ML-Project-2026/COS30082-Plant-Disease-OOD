"""Read-only cached-score parity audit. No images/models/training are executed."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch

from common import config, evaluate as original
from common.dataset import get_pairs
from approach3_ood_backbones.data import shared_frames
from approach3_ood_backbones.experiments import new_run,write_json,score_report
from approach3_ood_backbones.ood_evaluation import top1_predictions

KEYS=['seen_top1','seen_top5','unseen_top1','unseen_top5','harmonic_mean','crop_acc','disease_acc']


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


class CachedLoader:
    """Return row IDs, not images; predict reads the exact same cached tensors."""
    def __init__(self,labels,offset=0):
        self.labels=labels;self.offset=offset
        self.dataset=SimpleNamespace(norm_mean=(0.,0.,0.),norm_std=(1.,1.,1.))
    def __iter__(self):
        for start in range(0,len(self.labels),8):
            labels=self.labels[start:start+8]
            rows=torch.arange(start+self.offset,start+self.offset+len(labels))
            yield rows,torch.zeros_like(labels),torch.zeros_like(labels),labels


def compare_cached(payload,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    pairs=get_pairs()
    if not torch.equal(payload['pairs'].cpu(),pairs):raise ValueError('Candidate order mismatch; refusing remap.')
    frames=shared_frames();scores={};labels={}
    for split in ('seen','unseen'):
        scores[split]=payload[split+'_scores'].float().cpu()
        labels[split]=payload[split+'_labels'].long().cpu()
        if scores[split].shape!=(len(labels[split]),333):raise ValueError('Expected aligned N x333 cached scores.')
        indices=frames[split].pair_row.to_numpy()
        # Current audit payloads are full splits or recorded first-N smoke rows.
        if not torch.equal(labels[split],torch.tensor(indices[:len(labels[split])])):
            raise ValueError('Labels differ from the shared split ordering.')
        frames[split]=frames[split].iloc[:len(labels[split])]
    combined=torch.cat([scores['seen'],scores['unseen']])
    predict=lambda row_ids:combined[row_ids.long()]
    reference=output/'shared_reference';reference.mkdir()
    with patch.object(config,'RESULTS_DIR',reference),contextlib.redirect_stdout(io.StringIO()):
        shared_result=original.evaluate(predict,CachedLoader(labels['seen']),
            CachedLoader(labels['unseen'],len(labels['seen'])),'cached_reference',
            dict(scope=payload['scope'],no_model_inference=True),robust=False)
    extended=score_report(scores['seen'],scores['unseen'],frames,pairs)
    comparisons={key:dict(shared=shared_result[key],extended=extended[key],
                          identical=shared_result[key]==extended[key]) for key in KEYS}
    prediction_checks={}
    old_top1={}
    for split in ('seen','unseen'):
        expected=scores[split].topk(1,dim=1).indices[:,0]
        actual=top1_predictions(scores[split])
        old=scores[split].topk(5,dim=1).indices[:,0]
        old_top1[split]=round(100*(old==labels[split]).float().mean().item(),2)
        prediction_checks[split]=dict(rows=len(expected),identical=bool(torch.equal(actual,expected)),
            old_top5_first_prediction_disagreements=int((old!=expected).sum()))
    report=dict(scope=payload['scope'],label_provenance=payload['label_provenance'],
                candidate_count=333,seen_images=len(labels['seen']),unseen_images=len(labels['unseen']),
                torch_version=torch.__version__,comparison_device='CPU float32',
                metrics=comparisons,predictions=prediction_checks,
                old_extended_top1=old_top1,
                passed=all(v['identical'] for v in comparisons.values()) and all(v['identical'] for v in prediction_checks.values()),
                remaining_notes=[
                    'Shared pair Top1 uses topk(1); shared concept/confusion predictions use argmax. This intentional legacy distinction is preserved.',
                    'Topk tied indices are not guaranteed stable across PyTorch versions/devices. Parity here uses identical float32 CPU tensors in the same environment.',
                    'Empty extra OOD groups remain N/A; per-class/macro metrics are additions without a shared counterpart.',
                    'Existing argmax-based validation selection routines are not changed by this evaluation-only fix. Historical checkpoints/results remain untouched.'
                ])
    write_json(output/'parity.json',report)
    write_json(output/'extended_metrics.json',extended)
    pd.DataFrame([dict(metric=k,**v) for k,v in comparisons.items()]).to_csv(output/'metrics.csv',index=False)
    if not report['passed']:raise AssertionError('Shared/extended metric parity failed.')
    return report


def audit(cache_paths,output=None,reconstruct_smoke_labels=False,include_fixture=False):
    root=new_run('parity',output or config.RESULTS_DIR/'evaluation_consistency')
    before=digest(Path(original.__file__))
    frames=shared_frames();pairs=get_pairs();records=[]
    for index,path in enumerate(cache_paths):
        path=Path(path);source_hash=digest(path)
        cached=torch.load(path,map_location='cpu',weights_only=True)
        if 'seen_scores' in cached:
            payload=cached
        else:
            if not reconstruct_smoke_labels or cached.get('smoke') is not True:
                raise ValueError('Legacy cache has no labels; explicit smoke-label reconstruction is required.')
            if any(len(cached['scores'][s])!=2 for s in ('seen','unseen')):
                raise ValueError('Only the known two-image smoke cache format is supported for reconstruction.')
            payload=dict(pairs=cached['pairs'],scope='real-model cached smoke, two images per split',
                label_provenance='Reconstructed from unchanged shared split head(2), matching recorded smoke runner; legacy cache did not embed labels.')
            for split in ('seen','unseen'):
                payload[split+'_scores']=cached['scores'][split]
                payload[split+'_labels']=torch.tensor(frames[split].pair_row.iloc[:2].to_numpy())
        payload.setdefault('scope','provided cached predictions')
        payload.setdefault('label_provenance','embedded in provided cache')
        canonical=root/f'cache_{index}_labeled.pt'
        with canonical.open('xb') as stream:torch.save(payload,stream)
        # Both evaluators consume this exact saved-and-reloaded labelled cache.
        report=compare_cached(torch.load(canonical,map_location='cpu',weights_only=True),root/f'cache_{index}')
        if digest(path)!=source_hash:raise AssertionError('Source cache changed.')
        records.append(dict(source=str(path),source_sha256=source_hash,canonical_cache=str(canonical),**report))
    if include_fixture:
        generator=torch.Generator().manual_seed(42)
        payload=dict(pairs=pairs,scope='SYNTHETIC SCORES: full-size metric regression, NOT model accuracy',
                     label_provenance='Actual shared split labels; synthetic random/quantized scores with intentional ties.')
        for split in ('seen','unseen'):
            labels=torch.tensor(frames[split].pair_row.to_numpy())
            scores=torch.randn(len(labels),333,generator=generator).mul(2).round()
            scores[::13]=0.
            scores[torch.arange(1,len(labels),3),labels[1::3]]=10.
            payload[split+'_scores']=scores;payload[split+'_labels']=labels
        canonical=root/'synthetic_full_split_cache.pt'
        with canonical.open('xb') as stream:torch.save(payload,stream)
        records.append(compare_cached(torch.load(canonical,map_location='cpu',weights_only=True),root/'synthetic_full_split'))
    if digest(Path(original.__file__))!=before:raise AssertionError('Original common.evaluate changed.')
    write_json(root/'summary.json',dict(records=records,all_passed=all(r['passed'] for r in records),
        original_common_evaluate_sha256=before,source_files_unchanged=True,no_model_run=True))
    return root,records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,nargs='*',default=[])
    parser.add_argument('--reconstruct-smoke-labels',action='store_true')
    parser.add_argument('--include-synthetic-full-split',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if not args.cache and not args.include_synthetic_full_split:parser.error('Provide a cache or the labelled synthetic regression fixture.')
    root,records=audit(args.cache,args.output,args.reconstruct_smoke_labels,args.include_synthetic_full_split)
    for row in records:print(row['scope'],row['passed'],row['seen_images'],row['unseen_images'])
    print('Saved:',root)


if __name__=='__main__':main()
