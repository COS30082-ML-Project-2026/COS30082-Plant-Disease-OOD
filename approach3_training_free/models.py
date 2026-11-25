"""Reuse original checkpoint loader and preprocessing. No optimizers/backward."""
import torch
from torch.nn import functional as F
from common.dataset import get_pairs,get_transforms
from approach3_clip_improvements.approaches.clip_improvement_core import (
    read_checkpoint,extract_state,strict_load,prompt_groups,encode_groups,make_predict,file_hash,
)
from approach3_dino_fusion.core import E1,canonical_hash,validate_e1
from approach3_ood_backbones.adapters import versions


class FrozenCLIP(E1):
    def __init__(self,path,frames,device,spec):
        import open_clip
        self.source_hash=file_hash(path)
        if self.source_hash!=spec['sha256']:raise ValueError('Checkpoint release SHA256 mismatch.')
        payload=read_checkpoint(path);self.hp=payload['hp'];selection=payload['selection']
        if spec['name']=='E1_wise':validate_e1(payload,frames)
        if payload['model_name']!=spec['name'] or payload.get('debug') is not False:raise ValueError('Wrong/nonformal model.')
        if self.hp['backbone']!='ViT-B-16' or self.hp['pretrained']!='openai' or self.hp['norm']!='clip' or self.hp['batch_size']!=8 or self.hp['seed']!=42:
            raise ValueError('Original CLIP architecture/normalization/batch contract changed.')
        if not torch.equal(payload['pairs'],get_pairs()):raise ValueError('Candidate order mismatch.')
        for split in ('train','validation'):
            if canonical_hash(frames[split])!=selection['split'][split]:raise ValueError('Original training membership changed.')
        if selection['prompt_style']!='single' or payload['prompts']!=prompt_groups(get_pairs(),self.hp,'single'):
            raise ValueError('Stored original prompts differ.')
        self.prompts=payload['prompts'];self.pairs=payload['pairs'].cpu()
        self.model=open_clip.create_model('ViT-B-16',force_quick_gelu=True)
        strict_load(self.model,extract_state(payload),spec['name'])
        self.model.to(device).eval().requires_grad_(False)
        self.tokenizer=open_clip.get_tokenizer('ViT-B-16')
        self.transform=get_transforms('clip')[1]
        text=encode_groups(self.model,self.tokenizer,self.prompts,device,self.hp['text_chunk_size'])
        self.predict=make_predict(self.model,text,self.pairs)
        self.metadata=dict(checkpoint_sha256=self.source_hash,prompts=self.prompts,preprocessing=repr(self.transform),architecture='ViT-B-16 QuickGELU',versions=versions())

    @torch.inference_mode()
    def tta_scores(self,frame):
        # Identity is exactly the original preprocessing; the second view flips
        # that SAME normalized crop. Raw logit arithmetic mean, fixed two views.
        return torch.cat([((self.predict(images)+self.predict(images.flip(-1)))/2).float().cpu()
                          for images,*_ in self.loader(frame)])

    @torch.inference_mode()
    def prompted_scores(self,frame,prompts):
        device=next(self.model.parameters()).device
        text=encode_groups(self.model,self.tokenizer,[[p] for p in prompts],device,self.hp['text_chunk_size'])
        predict=make_predict(self.model,text,self.pairs)
        return torch.cat([predict(images).float().cpu() for images,*_ in self.loader(frame)])
