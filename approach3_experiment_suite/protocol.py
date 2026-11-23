"""Pure selection/mapping functions, independent of pretrained weights."""
import math
import torch
from approach3_ood_backbones.factorized import standardize

METHODS = {
    'D3': ('Frozen E1 + DINO crop fusion', 'inference'),
    'R1': ('Seen/novel combination bias calibration (pristine CLIP)', 'inference'),
    'F2': ('SigLIP2 factorized zero-shot scoring', 'inference'),
    'R5': ('Source-backed symptom prompts (partial label coverage)', 'inference'),
    'R4': ('CSP: learned crop/disease vocabulary, DPD adaptation', 'training'),
    'R3': ('Troika: three paths + adapters + cross-modal traction, DPD adaptation', 'training'),
    'F3': ('SigLIP2 partial visual fine-tuning', 'training'),
    'F4': ('SigLIP2 factorized supervision and scoring', 'training'),
}


def validate_config(config):
    if config['seed'] != 42:
        raise ValueError('Keep shared seed 42.')
    if not config['selected'] or set(config['selected']) - METHODS.keys():
        raise ValueError('Select only recommended IDs: '+', '.join(METHODS))
    for name in ('epochs','batch_size','accumulation_steps','partial_blocks','text_batch_size'):
        if not isinstance(config[name],int) or config[name]<1:
            raise ValueError(name+' must be a positive integer.')
    if config['precision'] not in ('float32','bf16'):
        raise ValueError('Use float32 or bf16; no silent precision fallback.')
    for value in [config['lr'],config['prompt_lr'],config['preservation_weight'],*config['loss_weights'],*config['bias_grid']]:
        if not math.isfinite(value): raise ValueError('Nonfinite hyperparameter.')
    if config['lr']<=0 or config['prompt_lr']<=0 or config['preservation_weight']<0:
        raise ValueError('Learning rates must be positive and preservation nonnegative.')
    if len(config['loss_weights'])!=3 or min(config['loss_weights'])<0 or not config['bias_grid']:
        raise ValueError('Supply three nonnegative loss weights and a nonempty bias grid.')


def class_space(pairs):
    pairs=torch.as_tensor(pairs).long()
    crops=sorted(set(pairs[:,0].tolist())); diseases=sorted(set(pairs[:,1].tolist()))
    crop_col={c:i for i,c in enumerate(crops)}; disease_col={d:i for i,d in enumerate(diseases)}
    return crops,diseases,torch.tensor([[crop_col[c],disease_col[d]] for c,d in pairs.tolist()])


def training_targets(labels, train_pair_rows):
    lookup={int(row):i for i,row in enumerate(train_pair_rows)}
    try:return torch.tensor([lookup[int(label)] for label in labels],device=labels.device)
    except KeyError as error:raise ValueError('Held-out pair supplied as a training target.') from error


def bias_scores(scores, seen_rows, bias):
    result=standardize(scores).clone()
    result[:,list(seen_rows)]-=bias
    return result


def select_bias(seen_scores,seen_labels,novel_scores,novel_labels,seen_rows,grid):
    if not len(seen_labels) or not len(novel_labels):raise ValueError('Both proxy categories are required.')
    history=[]
    for bias in grid:
        a=100*(bias_scores(seen_scores,seen_rows,bias).argmax(1)==seen_labels).float().mean().item()
        b=100*(bias_scores(novel_scores,seen_rows,bias).argmax(1)==novel_labels).float().mean().item()
        history.append(dict(bias=bias,seen_top1=a,novel_top1=b,harmonic_mean=2*a*b/(a+b) if a+b else 0.))
    winner=max(history,key=lambda r:r['harmonic_mean'])
    return dict(bias=winner['bias'],history=history,selection='proxy harmonic mean; config-order tie break',
                source='Original validation images of reduced-training pairs + training-only held-out pairs. No E1 checkpoint used.')


def accumulation_divisor(batch_index,batches,accumulation):
    start=(batch_index//accumulation)*accumulation
    return min(accumulation,batches-start)
