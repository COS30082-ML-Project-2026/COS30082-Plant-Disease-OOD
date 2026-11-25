"""Read-only E1 loading, crop prototypes and fixed score fusion."""
import hashlib

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from common import config as cfg
from common.dataset import PlantDataset, get_pairs, get_transforms
from approach3_clip_improvements.approaches.clip_improvement_core import (
    read_checkpoint, extract_state, strict_load, prompt_groups, encode_groups, make_predict, file_hash,
)
from approach3_ood_backbones.adapters import snapshot, versions


def canonical_hash(frame):
    payload = frame[['image','crop','disease','pair_row']].sort_values('image').to_csv(index=False,lineterminator='\n')
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_e1(checkpoint, frames):
    if checkpoint.get('model_name') != 'E1_wise' or checkpoint.get('debug') is not False:
        raise ValueError('Require the formal, non-debug E1_wise checkpoint.')
    hp, selection = checkpoint['hp'], checkpoint['selection']
    if hp['backbone'] != 'ViT-B-16' or hp['pretrained'] != 'openai' or hp['norm'] != 'clip':
        raise ValueError('E1 architecture/pretraining/normalization contract differs.')
    if hp['seed'] != 42 or hp.get('batch_size') != 8 or selection.get('prompt_style') != 'single' or selection.get('alpha') != 0.75:
        raise ValueError('Expected recorded single-prompt E1 alpha=0.75, seed=42.')
    if not torch.equal(checkpoint['pairs'], get_pairs()):
        raise ValueError('E1 candidate order differs from current 333 pair rows.')
    expected = prompt_groups(checkpoint['pairs'], hp, 'single')
    if checkpoint['prompts'] != expected:
        raise ValueError('Stored E1 prompts do not match original verified mappings.')
    for name in ('train','validation'):
        if canonical_hash(frames[name]) != selection['split'][name]:
            raise ValueError(f'{name} image membership differs from E1.')
    if cfg.IMAGE_SIZE != 224 or cfg.RESIZE_SIZE != 256 or cfg.NORM_STATS['clip'] != (
            (0.48145466,0.4578275,0.40821073),(0.26862954,0.26130258,0.27577711)):
        raise ValueError('Original E1 preprocessing geometry changed.')
    return hp


class E1:
    def __init__(self, path, frames, device, expected_sha256):
        import open_clip
        # Reuse the team's loader and strict compatibility checks, never another
        # adapter's zero-shot CLIP weights and never re-interpolate E1.
        self.source_hash = file_hash(path)
        if self.source_hash != expected_sha256:
            raise ValueError('E1 checkpoint differs from the recorded release asset SHA256.')
        checkpoint = read_checkpoint(path)
        self.hp = validate_e1(checkpoint, frames)
        self.model = open_clip.create_model(self.hp['backbone'], force_quick_gelu=True)
        strict_load(self.model, extract_state(checkpoint), 'Frozen E1 WiSE-FT')
        self.model.to(device).eval().requires_grad_(False)
        self.pairs = checkpoint['pairs'].cpu()
        self.prompts = checkpoint['prompts']
        tokenizer = open_clip.get_tokenizer(self.hp['backbone'])
        text = encode_groups(self.model, tokenizer, self.prompts, device, self.hp['text_chunk_size'])
        self.predict = make_predict(self.model, text, self.pairs)
        # E1 used shared Resize256/CenterCrop224, NOT OpenCLIP native Resize224.
        self.transform = get_transforms('clip')[1]
        self.metadata = dict(checkpoint_sha256=self.source_hash, architecture=self.hp['backbone'],
                             prompts=self.prompts, preprocessing=repr(self.transform),
                             candidate_pairs=self.pairs.tolist(), versions=versions(),
                             trainable_parameters=0, mode='eval; float32; no gradient updates')

    def loader(self, frame):
        return DataLoader(PlantDataset(frame,self.transform,'clip'), batch_size=self.hp['batch_size'],
                          shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(42))

    @torch.inference_mode()
    def scores(self, frame):
        return torch.cat([self.predict(images).float().cpu() for images,*_ in self.loader(frame)])


class Dino:
    def __init__(self, config, device, allow_download=False):
        from transformers import AutoImageProcessor, AutoModel
        path, revision = snapshot(config['dino_repository'],config['dino_revision'],allow_download,
                                  ['*.json','*.safetensors','pytorch_model.bin'])
        self.processor = AutoImageProcessor.from_pretrained(path,local_files_only=True,use_fast=False)
        self.model = AutoModel.from_pretrained(path,local_files_only=True).to(device).eval().requires_grad_(False)
        self.device = device
        self.metadata = dict(repository=config['dino_repository'],resolved_revision=revision,
                             preprocessing=self.processor.to_dict(),pooling=config['dino_pooling'],
                             trainable_parameters=0,versions=versions())

    @torch.inference_mode()
    def embeddings(self, frame, batch_size):
        result = []
        for offset in range(0,len(frame),batch_size):
            images=[]
            for name in frame.image.iloc[offset:offset+batch_size]:
                with Image.open(cfg.IMAGE_DIR / name) as image:
                    images.append(image.convert('RGB'))
            inputs=self.processor(images=images,return_tensors='pt').to(self.device)
            features=self.model(**inputs).last_hidden_state[:,0,:].float()
            if not torch.isfinite(features).all() or (features.norm(dim=1) < 1e-8).any():
                raise ValueError('Invalid DINO CLS embedding.')
            result.append(F.normalize(features,dim=1).cpu())
        return torch.cat(result)


def prototypes(embeddings, crop_labels, crop_ids):
    features=F.normalize(torch.as_tensor(embeddings).float(),dim=1)
    labels=torch.as_tensor(crop_labels).long()
    crop_ids=torch.as_tensor(crop_ids).long()
    if features.ndim!=2 or len(features)!=len(labels) or not torch.isfinite(features).all():
        raise ValueError('Invalid prototype features or labels.')
    means=[];counts=[]
    for crop in crop_ids:
        selected=features[labels==crop]
        if not len(selected):
            raise ValueError(f'No training examples for crop {int(crop)}; cannot invent a prototype.')
        mean=selected.mean(0)
        if mean.norm()<1e-8:
            raise ValueError('Degenerate crop prototype.')
        means.append(F.normalize(mean,dim=0)); counts.append(len(selected))
    return torch.stack(means), torch.tensor(counts)


def pair_crop_columns(pairs, crop_ids):
    lookup={int(c):i for i,c in enumerate(crop_ids)}
    try:
        return torch.tensor([lookup[int(c)] for c in torch.as_tensor(pairs)[:,0]])
    except KeyError as error:
        raise ValueError(f'Candidate crop {error} has no training prototype.') from error


def zscore(scores):
    scores=torch.as_tensor(scores).float()
    if not torch.isfinite(scores).all():
        raise ValueError('Nonfinite fusion scores.')
    return (scores-scores.mean(1,keepdim=True))/scores.std(1,unbiased=False,keepdim=True).clamp_min(1e-6)


def fuse(clip_scores,crop_scores,columns,clip_fraction=0.8):
    if clip_fraction != 0.8:
        raise ValueError('This experiment fixes CLIP/DINO weights at 0.8/0.2; no tuning.')
    if clip_scores.shape[0]!=crop_scores.shape[0] or clip_scores.shape[1]!=len(columns):
        raise ValueError('Fusion sample/candidate alignment mismatch.')
    return clip_fraction*zscore(clip_scores)+(1-clip_fraction)*zscore(crop_scores)[:,columns]


def crop_accuracy(scores, labels, crop_ids):
    labels=torch.as_tensor(labels).long()
    crop_ids=torch.as_tensor(crop_ids).long()
    if len(scores)!=len(labels) or not len(labels) or scores.shape[1]!=len(crop_ids):
        raise ValueError('Invalid crop-only evaluation.')
    top=crop_ids[scores.topk(min(5,len(crop_ids)),dim=1).indices]
    return dict(samples=len(labels),crop_top1=round(100*(top[:,0]==labels).float().mean().item(),2),
                crop_top5=round(100*(top==labels[:,None]).any(1).float().mean().item(),2),
                pair_top1=None,disease_acc=None,
                note='Crop-only prototype classification; cannot identify disease or pair by itself.')


def reproduction(actual, reference, tolerance=0.005):
    keys=['seen_top1','seen_top5','unseen_top1','unseen_top5','harmonic_mean','crop_acc','disease_acc']
    checks={k:dict(expected=reference[k],actual=actual[k],
                   passed=abs(actual[k]-reference[k])<=tolerance+1e-9) for k in keys}
    for name, expected in reference.get('robustness',{}).items():
        for metric in ('top1','drop'):
            value=actual['robustness'][name][metric]
            checks['robustness.'+name+'.'+metric]=dict(expected=expected[metric],actual=value,
                                                     passed=abs(value-expected[metric])<=tolerance+1e-9)
    return dict(passed=all(c['passed'] for c in checks.values()),checks=checks,
                tolerance_percentage_points=tolerance,
                scope='Recorded rounded aggregate metrics and robustness; original per-image predictions unavailable.')
