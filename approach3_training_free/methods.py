"""Label-free prediction recipes. Test labels are never arguments here."""
import torch
from approach3_dino_fusion.core import zscore
from approach3_ood_backbones.ood_evaluation import top1_predictions

STEPS={'T1':'E1 diagnostics + fixed bias','T2':'E1 + DINO crop prototypes',
       'T3':'E1/E3/E4 equal ensemble','T4':'E1 two-view TTA',
       'T5':'E1 symptom prompts + DINO disease prototypes'}


def validate(config):
    if config['seed']!=42 or not config['steps'] or set(config['steps'])-STEPS.keys():raise ValueError('Select T1–T5 with seed42.')
    if len(config['steps'])!=len(set(config['steps'])):raise ValueError('Each step may run only once per immutable sequence.')
    if config['bias']!=.5 or config['clip_fraction']!=.8:raise ValueError('This predeclared recipe fixes bias0.5 and CLIP fraction0.8.')
    if config['tta_views']!=['identity','horizontal_flip']:raise ValueError('TTA views were predeclared; no test-based view search.')
    if config['ensemble_weights']!=[1/3]*3:raise ValueError('Use predeclared equal ensemble weights.')
    if config['reproduction_tolerance']!=.005:raise ValueError('Do not loosen reference reproduction tolerance.')
    if config['dino_repository']!='facebook/dinov2-base':raise ValueError('Unexpected DINO backbone.')
    if config['dino_pooling']!='last_hidden_state[:, 0, :] (CLS)':raise ValueError('Keep the predeclared CLS feature definition.')
    if not isinstance(config['dino_batch_size'],int) or config['dino_batch_size']<1:raise ValueError('DINO batch size must be a positive integer.')


def correct_bias(scores,seen_pair_rows,bias=.5):
    result=zscore(scores).clone()
    result[:,list(seen_pair_rows)]-=bias
    return result


def ensemble(score_list):
    if len(score_list)!=3 or len({tuple(s.shape) for s in score_list})!=1:raise ValueError('Require three aligned score matrices.')
    return torch.stack([zscore(s) for s in score_list]).mean(0)


def map_components(pairs,ids,column):
    lookup={int(value):i for i,value in enumerate(ids)}
    if len(lookup)!=len(ids):raise ValueError('Duplicate component IDs.')
    try:return torch.tensor([lookup[int(v)] for v in pairs[:,column]])
    except KeyError as error:raise ValueError('Candidate component missing from training prototypes.') from error


def fuse_component(pair_scores,component_scores,columns):
    if len(pair_scores)!=len(component_scores) or pair_scores.shape[1]!=len(columns):raise ValueError('Unaligned fusion scores.')
    return .8*zscore(pair_scores)+.2*zscore(component_scores)[:,columns]


def diagnostics(scores,seen_pair_rows):
    scores=scores.float().cpu();mask=torch.zeros(scores.shape[1],dtype=torch.bool);mask[list(seen_pair_rows)]=True
    predicted=top1_predictions(scores)
    margin=scores[:,mask].max(1).values-scores[:,~mask].max(1).values
    return dict(samples=len(scores),predicted_seen_fraction=float(mask[predicted].float().mean()),
        mean_seen_minus_novel_margin=float(margin.mean()),median_seen_minus_novel_margin=float(margin.median()),
        zscore_top1_changes=int((top1_predictions(zscore(scores))!=predicted).sum()),
        scope='label-free score diagnostics; no correction parameter is fitted')


def component_accuracy(scores,labels,ids,name):
    scores=scores.float().cpu();labels=torch.as_tensor(labels).long().cpu();ids=torch.as_tensor(ids).long().cpu()
    if scores.shape!=(len(labels),len(ids)) or not len(labels):raise ValueError('Invalid component evaluation.')
    predicted=ids[top1_predictions(scores)]
    top5=ids[scores.topk(min(5,len(ids)),dim=1).indices]
    return dict(samples=len(labels),pair_top1=None,
        **{name+'_only_top1':round(100*(predicted==labels).float().mean().item(),2),
           name+'_only_top5':round(100*(top5==labels[:,None]).any(1).float().mean().item(),2)})
