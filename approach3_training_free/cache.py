"""Strict row/candidate/provenance checks; reject ambiguous legacy caches."""
import torch
from approach3_dino_fusion.core import canonical_hash


def manifest(frames,pairs,smoke):
    return dict(schema=1,smoke=smoke,pairs=pairs.cpu(),
        frames={k:dict(images=f.image.tolist(),sha256=canonical_hash(f)) for k,f in frames.items()})


def validate_cache(payload,frames,pairs,smoke,kind,source=None):
    if payload.get('schema')!=1 or payload.get('smoke')!=smoke or payload.get('kind')!=kind:
        raise ValueError('Cache schema/kind/smoke scope mismatch. Do not use smoke scores for formal evaluation.')
    if not torch.equal(payload['pairs'].cpu(),pairs.cpu()):raise ValueError('333 candidate ordering changed.')
    for split,frame in frames.items():
        entry=payload['frames'][split]
        if entry['images']!=frame.image.tolist() or entry['sha256']!=canonical_hash(frame):raise ValueError('Cache image order/membership mismatch: '+split)
    if source is not None and payload.get('checkpoint_sha256')!=source:raise ValueError('Wrong pretrained/fine-tuned checkpoint cache.')
    key='scores' if kind=='clip_scores' else 'features'
    for split,value in payload[key].items():
        if split not in frames or len(value)!=len(frames[split]) or value.ndim!=2 or not torch.isfinite(value).all():raise ValueError('Invalid cached tensor.')
        if kind=='clip_scores' and value.shape[1]!=333:raise ValueError('Cache restricts candidate space; require all333.')
    expected={'seen','unseen'} if kind=='clip_scores' else {'train','seen','unseen'}
    if set(payload[key])!=expected:raise ValueError('Missing cache splits.')
    return payload


def save_cache(path,payload):
    with path.open('xb') as stream:torch.save(payload,stream)
