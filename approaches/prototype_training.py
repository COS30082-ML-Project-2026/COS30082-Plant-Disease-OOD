"""Frozen-feature training and checkpoint helpers using the shared data contract."""
from copy import deepcopy
import json
import platform
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from common import config
from common.dataset import PlantDataset
from approaches.prototype_model import CompositionalPrototypeRetrieval


@torch.no_grad()
def cache_features(model, loader):
    device = next(model.parameters()).device
    batches = [[], [], [], []]
    for images, crops, diseases, pairs in loader:
        values = [model.encode(images.to(device)).cpu(), crops, diseases, pairs]
        for destination, value in zip(batches, values):
            destination.append(value)
    return tuple(torch.cat(values) for values in batches)


def support_loader(train_loader, val_loader):
    """The training subset with deterministic evaluation transforms, never val images."""
    dataset = PlantDataset(train_loader.dataset.frame.copy(),
                           val_loader.dataset.transform, 'imagenet')
    return DataLoader(dataset, batch_size=train_loader.batch_size, shuffle=False,
                      num_workers=config.NUM_WORKERS)


def _epoch(model, loader, optimizer, device):
    model.train()
    total, count = 0., 0
    for features, crops, diseases, _ in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = model.training_loss(features.to(device), crops.to(device), diseases.to(device))
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * len(features)
        count += len(features)
    return total / count


@torch.no_grad()
def _validation(model, cache, device):
    scores = model.score_features(cache[0].to(device))
    return 100 * (scores.argmax(1).cpu() == cache[3]).float().mean().item()


def train_model(model, train_loader, val_loader, hp, debug=False):
    """Cache one augmented view, learn heads, choose epoch by shared validation split."""
    device = next(model.parameters()).device
    support = support_loader(train_loader, val_loader)
    print('Caching augmented training, deterministic training support, and validation features', flush=True)
    augmented, reference, validation = [cache_features(model, loader)
                                       for loader in (train_loader, support, val_loader)]
    _check_support(model, reference, debug)
    batches = DataLoader(TensorDataset(*augmented), batch_size=hp['head_batch_size'],
                         shuffle=True, generator=torch.Generator().manual_seed(config.SEED))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=hp['lr'], weight_decay=hp['weight_decay'])
    return _fit(model, batches, optimizer, reference, validation, hp, debug, device, support)


def _fit(model, batches, optimizer, reference, validation, hp, debug, device, support):
    best_accuracy, best_state, history = -1., None, []
    for epoch in range(1 if debug else hp['epochs']):
        model.fit_prototypes(*(value.to(device) for value in reference[:3]))
        loss = _epoch(model, batches, optimizer, device)
        model.eval().fit_prototypes(*(value.to(device) for value in reference[:3]))
        accuracy = _validation(model, validation, device)
        history.append({'epoch': epoch + 1, 'loss': loss, 'val_top1': accuracy})
        print(history[-1], flush=True)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return history, build_memory(model, reference, support.dataset.frame)


def _check_support(model, reference, debug):
    crop_ids, disease_ids = reference[1].unique(), reference[2].unique()
    unsupported = (~torch.isin(model.pairs[:, 0].cpu(), crop_ids)
                   | ~torch.isin(model.pairs[:, 1].cpu(), disease_ids)).nonzero().flatten()
    if len(unsupported) and not debug:
        raise ValueError(f'No training support for concepts in pair rows {unsupported.tolist()}; '
                         'visual prototypes cannot identify entirely unseen concepts.')
    if len(unsupported):
        print(f'DEBUG subset lacks concepts for {len(unsupported)} pairs; not a benchmark run.')


@torch.no_grad()
def build_memory(model, reference, frame):
    device = next(model.parameters()).device
    crop, disease = model.project(reference[0].to(device))
    return {'crop_embeddings': crop.cpu(), 'disease_embeddings': disease.cpu(),
            'crops': reference[1], 'diseases': reference[2],
            'paths': frame['image'].tolist(), 'source': 'training_subset_only'}


def save_checkpoint(model, memory, history, hp, model_name, debug):
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {'model_name': model_name, 'state_dict': model.state_dict(), 'hp': hp,
                  'memory': memory, 'history': history, 'debug': debug,
                  'environment': {'torch': str(torch.__version__), 'python': platform.python_version()}}
    torch.save(checkpoint, config.CHECKPOINT_DIR / f'{model_name}.pt')
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / f'{model_name}_training.json').write_text(
        json.dumps({'history': history, 'debug': debug, 'hp': hp}, indent=2))


def load_checkpoint(path, device='cpu'):
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    state = checkpoint['state_dict']
    model = CompositionalPrototypeRetrieval(checkpoint['hp'], state['pairs'],
        state['crop_prototypes'].shape[0], state['disease_prototypes'].shape[0], pretrained=False)
    model.load_state_dict(state)
    return model.to(device).eval(), checkpoint['memory'], checkpoint
