"""DPD-specific implementations of the selected trainable models.

CSP vocabulary construction follows Nayak et al. (ICLR 2023). Troika adaptation
implements three semantic paths, two visual residual adapters per transformer
block, learned contexts, visual disentanglers and image-conditioned cross-modal
traction (Huang et al., CVPR 2024). These are OpenCLIP B/16 adaptations, not
claims of reproducing published L/14 benchmark numbers. See RESEARCH_NOTES.md.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from common.dataset import get_crop_and_disease_names
from approach3_ood_backbones.adapters import load_adapter
from approach3_experiment_suite.protocol import class_space


class Bottleneck(nn.Module):
    def __init__(self,width,hidden):
        super().__init__()
        self.down=nn.Linear(width,hidden);self.up=nn.Linear(hidden,width)
        nn.init.zeros_(self.up.weight);nn.init.zeros_(self.up.bias)
    def forward(self,x):return .1*self.up(F.relu(self.down(x)))


class PromptModel(nn.Module):
    def __init__(self,adapter,pairs,config,troika=False,name_maps=None):
        super().__init__()
        import open_clip
        self.base=adapter.model
        self.base.requires_grad_(False)
        self.config=config;self.troika=troika;self.device=next(self.base.parameters()).device
        self.tokenizer=open_clip.get_tokenizer('ViT-B-16')
        crops,diseases,columns=class_space(pairs)
        self.register_buffer('columns',columns.to(self.device))
        crop_names,disease_names=name_maps if name_maps is not None else get_crop_and_disease_names()
        # CSP treats each component as ONE learnable token, initialized from the
        # mean pretrained subword embeddings for its exact verified class name.
        names=[disease_names[d].replace('_',' ') for d in diseases]+[crop_names[c].replace('_',' ') for c in crops]
        tokens=self.tokenizer(names).to(self.device)
        with torch.no_grad():
            embeddings=self.base.token_embedding(tokens)
            initial=torch.stack([row[1:int(token.argmax())].mean(0) for row,token in zip(embeddings,tokens)])
        self.soft=nn.Parameter(initial)
        self.nd=len(diseases);self.nc=len(crops)
        self.text_checkpoint=config['gradient_checkpointing']
        templates=['a photo of X X','a photo of X','a photo of X']
        self.register_buffer('templates',self.tokenizer(templates).to(self.device))
        if troika:
            with torch.no_grad():ctx=self.base.token_embedding(self.templates)[:,1:4].clone()
            self.contexts=nn.Parameter(ctx)
            visual=self.base.visual
            width=visual.conv1.out_channels
            self.adapters=nn.ModuleList([Bottleneck(width,config['adapter_dim']) for _ in range(2*len(visual.transformer.resblocks))])
            dim=initial.shape[1]
            # Text width == embedding dim for the verified B/16 architecture.
            if self.base.text_projection.shape[-1]!=dim:raise ValueError('Inspect CSP/Troika text/embedding projection dimensions.')
            self.disentanglers=nn.ModuleList([nn.Sequential(nn.Linear(dim,dim),nn.BatchNorm1d(dim),nn.ReLU(),nn.Dropout(.5)) for _ in range(2)])
            self.cross=nn.ModuleList([nn.MultiheadAttention(dim,dim//64,batch_first=True,dropout=.1) for _ in range(config['cmt_layers'])])
            self.cross_norm=nn.ModuleList([nn.LayerNorm(dim) for _ in range(config['cmt_layers'])])
            self.cross_mlp=nn.ModuleList([nn.Sequential(nn.Linear(dim,4*dim),nn.GELU(),nn.Dropout(.1),nn.Linear(4*dim,dim)) for _ in range(config['cmt_layers'])])
            self.patch_norm=nn.LayerNorm(dim)
            self.traction_scale=nn.Parameter(torch.full((dim,),config['cmt_initial_scale']))
        self.to(self.device)

    def text_features(self,branch):
        if branch==0:
            # Attribute = disease, object = crop, mapped back to original pair rows.
            components=[self.soft[self.columns[:,1]],self.soft[self.nd+self.columns[:,0]]]
        elif branch==1:components=[self.soft[:self.nd]]
        else:components=[self.soft[self.nd:]]
        count=len(components[0]);token=self.templates[branch].repeat(count,1)
        x=self.base.token_embedding(token).detach().clone()
        eos=int(token[0].argmax())
        for i,component in enumerate(components):x[:,eos-len(components)+i]=F.dropout(component,.0,training=self.training)
        if self.troika:x[:,1:4]=self.contexts[branch]
        def encode(chunk,ids):
            chunk=chunk+self.base.positional_embedding
            chunk=self.base.transformer(chunk,attn_mask=self.base.attn_mask)
            chunk=self.base.ln_final(chunk)
            result=chunk[torch.arange(len(chunk),device=chunk.device),ids.argmax(-1)] @ self.base.text_projection
            return F.normalize(result.float(),dim=-1)
        result=[]
        size=self.config['text_batch_size']
        for start in range(0,count,size):
            xx,tt=x[start:start+size],token[start:start+size]
            result.append(checkpoint(encode,xx,tt,use_reentrant=False) if self.training and self.text_checkpoint else encode(xx,tt))
        return torch.cat(result)

    def image_features(self,pixels):
        if not self.troika:
            with torch.no_grad():return F.normalize(self.base.encode_image(pixels).float(),dim=-1),None
        visual=self.base.visual;x=visual._embeds(pixels)
        blocks=visual.transformer.resblocks;n=len(blocks)
        for i,block in enumerate(blocks):
            def apply(value,block=block,i=i):
                value=value+block.attention(block.ln_1(value))+self.adapters[i](value)
                return value+block.mlp(block.ln_2(value))+self.adapters[n+i](value)
            x=checkpoint(apply,x,use_reentrant=False) if self.training and self.config['gradient_checkpointing'] else apply(x)
        x=visual.ln_post(x) @ visual.proj
        return x[:,0].float(),x.float()

    def forward(self,pixels):
        self.base.eval()
        image,patches=self.image_features(pixels)
        texts=[self.text_features(0)]
        if not self.troika:return [self.base.logit_scale.exp()*image @ texts[0].T,None,None]
        texts += [self.text_features(1),self.text_features(2)]
        visuals=[F.normalize(image,dim=-1)]+[F.normalize(module(image),dim=-1) for module in self.disentanglers]
        logits=[]
        patches=self.patch_norm(patches)
        for visual,text in zip(visuals,texts):
            adapted=text.unsqueeze(0).expand(len(image),-1,-1)
            for attention,norm,mlp in zip(self.cross,self.cross_norm,self.cross_mlp):
                adapted=adapted+attention(adapted,patches,patches,need_weights=False)[0]
                adapted=adapted+mlp(norm(adapted))
            conditioned=F.normalize(text[None]+self.traction_scale*adapted,dim=-1)
            logits.append(self.base.logit_scale.exp()*torch.einsum('bd,bkd->bk',visual,conditioned))
        # Return pair,crop,disease in the shared suite convention.
        return [logits[0],logits[2],logits[1]]


class SiglipModel(nn.Module):
    def __init__(self,adapter,banks,config):
        super().__init__();self.base=adapter.model;self.config=config
        self.base.requires_grad_(False)
        blocks=self.base.vision_model.encoder.layers
        if len(blocks)<config['partial_blocks']:raise ValueError('Too many partial visual blocks.')
        for block in blocks[-config['partial_blocks']:]:block.requires_grad_(True)
        for i,bank in enumerate(banks):self.register_buffer('bank'+str(i),bank.detach())
        if config['gradient_checkpointing']:self.base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        self.teacher=None
        if config['preservation_weight']:
            self.teacher=copy.deepcopy(self.base).eval().requires_grad_(False)
            self.teacher.gradient_checkpointing_disable()
    def forward(self,pixels):
        image=F.normalize(self.base.get_image_features(pixel_values=pixels).float(),dim=-1)
        self.last_features=image
        return [self.base.logit_scale.exp()*image @ getattr(self,'bank'+str(i)).T+self.base.logit_bias for i in range(3)]
    def preservation(self,pixels):
        if self.teacher is None:return self.last_features.new_zeros(())
        with torch.no_grad():target=F.normalize(self.teacher.get_image_features(pixel_values=pixels).float(),dim=-1)
        return (1-(target*self.last_features).sum(1)).mean()


def build_model(method,adapter,banks,pairs,config,name_maps=None):
    return SiglipModel(adapter,banks,config) if method in ('F3','F4') else PromptModel(adapter,pairs,config,troika=method=='R3',name_maps=name_maps)
