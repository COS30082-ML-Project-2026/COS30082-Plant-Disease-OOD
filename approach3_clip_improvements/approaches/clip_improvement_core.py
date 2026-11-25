"""Member D improvement contracts, checkpoints and prompts; no shared-code edits."""
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import config as cfg
from common.dataset import get_crop_and_disease_names, get_loaders, get_pairs, load_split


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def seed_all(seed):
    cfg.set_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def frame_hash(frame):
    columns = ['image', 'crop', 'disease', 'pair_row']
    payload = frame[columns].sort_values('image').to_csv(index=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def checked_loaders(hp, debug, workers):
    """Reuse team membership/transforms; DEBUG alone may bound the returned subsets."""
    seed_all(hp['seed'])
    loaders = get_loaders(hp['batch_size'], norm='clip', debug=debug)
    signatures = [frame_hash(loader.dataset.frame) for loader in loaders]
    for loader in loaders:
        loader.worker_init_fn = seed_worker
        require(loader.num_workers == workers, 'Unexpected workers: recreate shared loaders.')
        if debug:
            loader.dataset.frame = loader.dataset.frame.iloc[:hp['debug_samples']].reset_index(drop=True)
    return loaders, signatures


def check_data():
    pairs = get_pairs()
    require(tuple(pairs.shape) == (333, 2), f'Expected pairs [333,2], found {pairs.shape}.')
    require(len(set(map(tuple, pairs.tolist()))) == 333, 'Duplicate candidate pairs; check common.dataset.')
    frames = [load_split(p) for p in (cfg.TRAIN_CSV, cfg.TEST_SEEN_CSV, cfg.TEST_UNSEEN_CSV)]
    require([len(f) for f in frames] == [8000, 2000, 1448], 'Incorrect DPD-Small CSV counts.')
    expected = [set(range(323)), set(range(323)), set(range(323, 333))]
    for frame, labels in zip(frames, expected):
        require(set(frame.pair_row) == labels, 'Pair row order/range differs from shared contract.')
        require(all((cfg.IMAGE_DIR / x).is_file() for x in frame.image), 'Missing images under IMAGE_DIR.')
    for index, frame in enumerate(frames):
        require(frame.image.is_unique, 'Duplicate image names in a split.')
        require(all(not set(frame.image) & set(f.image) for f in frames[index+1:]), 'Image leakage across splits.')
    return pairs


def resolve_partial_path(requested):
    if requested:
        return Path(requested).expanduser().resolve()
    paths = [cfg.CHECKPOINT_DIR / 'approach3_clip' / 'approach3_clip_6.pt',
             Path.home() / 'Downloads' / 'approach3_clip_6.pt',
             Path.home() / 'OneDrive' / 'Model' / 'checkpoints' / 'approach3_clip' / 'approach3_clip_6.pt']
    found = next((p for p in paths if p.is_file()), None)
    require(found is not None, 'E1/E2: missing Partial-6 checkpoint. Set CLIP_PARTIAL6 to its local .pt path.')
    return found.resolve()


def read_checkpoint(path):
    require(path.is_file(), f'Missing checkpoint {path}; copy it locally or set CLIP_PARTIAL6.')
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    except (OSError, RuntimeError) as error:
        raise RuntimeError(f'Cannot read {path}. Download the OneDrive file locally. {error}') from error
    require(isinstance(checkpoint, dict), f'Invalid checkpoint structure at {path}.')
    return checkpoint


def extract_state(checkpoint):
    state = checkpoint.get('state_dict', checkpoint)
    require(state and all(isinstance(k, str) and torch.is_tensor(v) for k, v in state.items()),
            'Checkpoint must contain a tensor state_dict.')
    if all(k.startswith('module.') for k in state):
        state = {k[len('module.'):]: v for k, v in state.items()}
    return state


def strict_load(model, state, context):
    expected = model.state_dict()
    missing, unexpected = sorted(expected.keys() - state.keys()), sorted(state.keys() - expected.keys())
    incompatible = [k for k in expected.keys() & state.keys() if expected[k].shape != state[k].shape
                    or expected[k].dtype != state[k].dtype]
    report = dict(missing=missing, unexpected=unexpected, incompatible=incompatible)
    require(not any(report.values()), f'{context}: incompatible checkpoint: {report}. Check architecture.')
    require(all(torch.isfinite(t).all() for t in state.values() if t.is_floating_point()),
            f'{context}: checkpoint has non-finite weights.')
    model.load_state_dict(state, strict=True)
    return report


def interpolate(base, tuned, alpha):
    require(0 <= alpha <= 1, f'WiSE-FT alpha must be in [0,1], got {alpha}.')
    require(base.keys() == tuned.keys(), 'WiSE-FT state keys differ.')
    require(all(base[k].shape == tuned[k].shape and base[k].dtype == tuned[k].dtype for k in base),
            'WiSE-FT tensor shapes or dtypes differ.')
    return {k: base[k].lerp(tuned[k], alpha) if base[k].is_floating_point() else base[k].clone()
            for k in base}


def configure_visual(model, variant):
    model.requires_grad_(False)
    if variant == 'partial6':
        blocks = dict(model.visual.named_modules()).get('transformer.resblocks')
        require(isinstance(blocks, torch.nn.ModuleList) and len(blocks) >= 6,
                'Partial-6: expected visual.transformer.resblocks ModuleList; inspect OpenCLIP.')
        modules = list(blocks)[-6:]
    else:
        require(variant == 'visual_full', f'Unknown trainable variant: {variant}.')
        modules = [model.visual]
    expected = {id(p) for module in modules for p in module.parameters()}
    for module in modules:
        module.requires_grad_(True)
    require({id(p) for p in model.parameters() if p.requires_grad} == expected,
            'Unexpected trainable parameters; text/projection/logit_scale freeze mismatch.')
    return [k for k, p in model.named_parameters() if p.requires_grad]


def prompt_groups(pairs, hp, style):
    crop_names, disease_names = get_crop_and_disease_names()
    groups = []
    for crop_id, disease_id in pairs.tolist():
        crop, disease = crop_names[crop_id].replace('_', ' '), disease_names[disease_id].replace('_', ' ')
        if style == 'single':
            templates = [hp['prompt']]
        elif disease.casefold().strip() == 'healthy':
            templates = hp['healthy_templates']
        else:
            templates = hp['ensemble_templates'][:int(style.removeprefix('ensemble'))]
        groups.append([template.format(crop=crop, disease=disease) for template in templates])
    return groups


@torch.no_grad()
def encode_groups(model, tokenizer, groups, device, chunk_size):
    texts = [text for group in groups for text in group]
    tokens = tokenizer(texts)
    features = [F.normalize(model.encode_text(batch.to(device)).float(), dim=1)
                for batch in tokens.split(chunk_size)]
    features = torch.cat(features)
    result, offset = [], 0
    for group in groups:
        result.append(F.normalize(features[offset:offset+len(group)].mean(dim=0), dim=0))
        offset += len(group)
    return torch.stack(result)


def make_predict(model, text, pairs):
    """The score order is the immutable shared pair table; accepts CPU/GPU images."""
    device = next(model.parameters()).device
    @torch.no_grad()
    def predict(images):
        require(torch.equal(get_pairs(), pairs), 'Candidate order changed after setup; restart run.')
        model.eval()
        features = F.normalize(model.encode_image(images.to(device)).float(), dim=1)
        scores = model.logit_scale.exp().float() * features @ text.T
        require(tuple(scores.shape) == (len(images), 333), f'Expected [B,333], observed {scores.shape}.')
        require(torch.isfinite(scores).all(), 'Non-finite CLIP scores; check weights and precision.')
        return scores
    return predict


def train_manifest(loader, val_loader):
    train, val = loader.dataset.frame, val_loader.dataset.frame
    require(not set(train.image) & set(val.image), 'Train/validation image leakage.')
    return dict(train=frame_hash(train), validation=frame_hash(val),
                train_count=len(train), validation_count=len(val),
                note='Original shared per-pair image split. No proxy split or extra held-out pairs.')


class EvaluationDataset:
    """Wrap the original split to enforce deterministic evaluation without changing membership."""
    def __init__(self, dataset):
        self.dataset = dataset
        self.norm_mean, self.norm_std = dataset.norm_mean, dataset.norm_std
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        return self.dataset[index]


def evaluation_loader(loader):
    # Avoid persistent-worker RNG advancing differently across trials/robustness calls.
    return DataLoader(EvaluationDataset(loader.dataset), batch_size=loader.batch_size, shuffle=False,
                      num_workers=0, generator=torch.Generator().manual_seed(cfg.SEED))
