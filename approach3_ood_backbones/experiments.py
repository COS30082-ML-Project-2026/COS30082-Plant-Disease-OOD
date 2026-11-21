"""F1/F2 inference and reports. No F3/F4 training is implemented."""
import gc
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from PIL import Image

from common import config as cfg
from common.dataset import get_crop_and_disease_names, get_pairs
from approach3_ood_backbones.adapters import load_adapter, versions
from approach3_ood_backbones.data import shared_frames, compositional_split, signature
from approach3_ood_backbones.factorized import combine, select_weights
from approach3_ood_backbones.ood_evaluation import breakdown, counts, metrics, membership


def write_json(path, data):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)


def new_run(stage, output=None):
    path = Path(output or cfg.RESULTS_DIR / 'ood_backbones') / (
        stage + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:8])
    path.mkdir(parents=True, exist_ok=False)
    return path


def audit(config, output=None):
    frames = shared_frames()
    reduced, holdout, manifest = compositional_split(frames['train'], config['pair_holdout_fraction'], config['seed'])
    path = new_run('B', output)
    for name, frame in [('reduced_train', reduced), ('compositional_validation', holdout)]:
        frame.to_csv(path / (name + '.csv'), index=False)
    record = dict(primary_objective='unseen crop-disease combinations', split=manifest,
                  candidate_pairs=get_pairs().tolist(),
                  shared_signatures={name: signature(frame) for name, frame in frames.items()},
                  official_counts={name: counts(frames[name], frames['train']) for name in ('seen', 'unseen')},
                  holdout_counts=counts(holdout, reduced))
    write_json(path / 'audit.json', record)
    return path, record


def text_bank(adapter, config):
    pairs = get_pairs()
    crops, diseases = get_crop_and_disease_names()
    crop_ids = sorted({int(c) for c in pairs[:, 0]})
    disease_ids = sorted({int(d) for d in pairs[:, 1]})
    crop_column = {c: i for i, c in enumerate(crop_ids)}
    disease_column = {d: i for i, d in enumerate(disease_ids)}
    def clean(name):
        return name.replace('_', ' ')
    groups = [
        [config['prompt'].format(crop=clean(crops[c]), disease=clean(diseases[d])) for c, d in pairs.tolist()],
        [config['crop_prompt'].format(crop=clean(crops[c])) for c in crop_ids],
        [config['disease_prompt'].format(disease=clean(diseases[d])) for d in disease_ids],
    ]
    banks = []
    batch = config['text_batch_size']
    for group in groups:
        banks.append(torch.cat([adapter.encode_text(group[i:i+batch]) for i in range(0, len(group), batch)]))
    columns = torch.tensor([[crop_column[c], disease_column[d]] for c, d in pairs.tolist()])
    return banks, columns, groups


def collect(adapter, banks, frame, batch_size, perturbation=None):
    """PIL images -> each model's native preprocessing. Preserve CSV row order."""
    collected = [[] for _ in banks]
    generator = torch.Generator().manual_seed(42)
    for offset in range(0, len(frame), batch_size):
        images = []
        for name in frame.image.iloc[offset:offset+batch_size]:
            with Image.open(cfg.IMAGE_DIR / name) as image:
                images.append(image.convert('RGB'))
        if perturbation:
            from torchvision.transforms import functional as TF
            changed = []
            for image in images:
                pixels = TF.to_tensor(image)
                if perturbation == 'brightness':
                    pixels = TF.adjust_brightness(pixels, 0.6)
                elif perturbation == 'blur':
                    pixels = TF.gaussian_blur(pixels, 9, 2.0)
                elif perturbation == 'noise':
                    pixels = (pixels + 0.08 * torch.randn(pixels.shape, generator=generator)).clamp(0, 1)
                else:
                    raise ValueError('Unknown perturbation.')
                changed.append(TF.to_pil_image(pixels))
            images = changed
        features = adapter.encode_images(images)
        for target, text in zip(collected, banks):
            target.append(adapter.similarity(features, text).float().cpu())
    return tuple(torch.cat(rows) for rows in collected)


def score_report(seen_scores, unseen_scores, frames, pairs):
    seen = metrics(seen_scores, torch.tensor(frames['seen'].pair_row.to_numpy()), pairs)
    unseen = metrics(unseen_scores, torch.tensor(frames['unseen'].pair_row.to_numpy()), pairs)
    a, b = seen['top1'], unseen['top1']
    combined_frame = pd.concat([frames['seen'], frames['unseen']], ignore_index=True)
    combined_scores = torch.cat([seen_scores, unseen_scores])
    marked = membership(combined_frame, frames['train'])
    subsets = {}
    # These summaries can overlap when components really are unseen. The detailed
    # 8-row breakdown below is mutually exclusive and must be used for totals.
    for name, mask in [('unseen_crop', ~marked.crop_seen),
                       ('unseen_disease', ~marked.disease_seen),
                       ('unseen_combination', marked.crop_seen & marked.disease_seen & ~marked.pair_seen)]:
        mask = torch.tensor(mask.to_numpy())
        subsets[name] = metrics(combined_scores[mask], torch.tensor(combined_frame.pair_row.to_numpy())[mask], pairs)
    summaries = {name + '_' + key: value[key] for name, value in subsets.items() for key in ('samples', 'top1', 'top5')}
    return dict(seen_top1=a, seen_top5=seen['top5'], unseen_top1=b, unseen_top5=unseen['top5'],
                harmonic_mean=round(2*a*b/(a+b), 2) if a+b else 0.0,
                crop_acc=seen['crop_acc'], disease_acc=seen['disease_acc'],
                unseen_split_crop_acc=unseen['crop_acc'], unseen_split_disease_acc=unseen['disease_acc'],
                **summaries,
                seen=seen, unseen=unseen,
                ood_breakdown=breakdown(combined_scores, combined_frame, frames['train'], pairs),
                candidate_space='all 333 original candidates, including 10 official unseen pairs')


def save_confusion(path, name, scores, frame, pairs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matrix = torch.zeros(323, len(pairs), dtype=torch.long)
    labels = torch.tensor(frame.pair_row.to_numpy())
    matrix.index_put_((labels, scores.argmax(1)), torch.ones_like(labels), accumulate=True)
    pd.DataFrame(matrix.numpy()).to_csv(path / (name + '_confusion.csv'), index_label='true_pair_row')
    fig, axis = plt.subplots(figsize=(10, 8))
    plot = axis.imshow((matrix.float() / matrix.sum(1, keepdim=True).clamp_min(1)).numpy(), aspect='auto', cmap='viridis')
    axis.axvline(322.5, color='red', linewidth=0.7)
    axis.set(xlabel='Predicted candidate row (0–332)', ylabel='True seen candidate row (0–322)', title=name)
    fig.colorbar(plot, ax=axis)
    fig.savefig(path / (name + '_confusion.png'), dpi=130, bbox_inches='tight')
    plt.close(fig)


def run(config, stage, model_names, smoke=True, allow_download=False, robust=False, output=None, adapter_factory=load_adapter):
    if stage not in ('F1', 'F2') or (stage == 'F2' and model_names != ['siglip2']):
        raise ValueError('F2 currently requires exactly --models siglip2; F3/F4 are not implemented.')
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    frames = shared_frames()
    reduced, holdout, manifest = compositional_split(frames['train'], config['pair_holdout_fraction'], config['seed'])
    path = new_run(stage + ('_smoke' if smoke else '_full'), output)
    write_json(path / 'config.json', config)
    write_json(path / 'split_manifest.json', dict(**manifest, shared_signatures={k: signature(v) for k,v in frames.items()}))
    reduced.to_csv(path / 'reduced_train.csv', index=False)
    holdout.to_csv(path / 'compositional_validation.csv', index=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not smoke and device != 'cuda':
        raise RuntimeError('Full benchmarks require CUDA. Use smoke mode on this CPU machine.')
    if device == 'cpu':
        torch.set_num_threads(config.get('cpu_threads', 4))
    pairs = get_pairs()
    write_json(path / 'candidate_pairs.json', pairs.tolist())
    reports = []
    for name in model_names:
        adapter = None
        try:
            if name not in config['models']:
                raise ValueError(f'Unknown model {name}.')
            started = time.perf_counter()
            adapter = adapter_factory(config['models'][name], device, allow_download)
            if device == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            banks, columns, prompts = text_bank(adapter, config)
            # Explicit inference preflight, before any full validation/test run.
            preflight = collect(adapter, banks, frames['validation'].iloc[:2], min(config['batch_size'], 2))
            if preflight[0].shape != (2, 333) or not all(torch.isfinite(s).all() for s in preflight):
                raise ValueError('Inference smoke failed: nonfinite or misaligned scores.')
            selection = None
            if stage == 'F2':
                selection_frame = holdout.iloc[:2] if smoke else holdout
                validation_scores = collect(adapter, banks, selection_frame, config['batch_size'])
                selection = select_weights(validation_scores, columns, selection_frame.pair_row.to_numpy(), config['coefficient_grid'])
                selection.update(split=manifest, neural_training_images=0,
                                 note='Zero-shot backbone: holdout is disjoint from designated reduced train; no neural weights trained on either partition.')
                # Seal before accessing official test IMAGES or computing their scores.
                write_json(path / (name + '_selection_frozen.json'), selection)
            # Bound inference images only. Membership must always use the FULL
            # optimization-training partition, including during smoke tests.
            evaluation_frames = {k: (v.iloc[:2] if smoke and k != 'train' else v) for k,v in frames.items()}
            seen_components = collect(adapter, banks, evaluation_frames['seen'], config['batch_size'])
            unseen_components = collect(adapter, banks, evaluation_frames['unseen'], config['batch_size'])
            variants = [('pair_only', None)]
            if stage == 'F2':
                variants += [('calibrated_pair_only', [1, 0, 0]), ('factorized', selection['weights'])]
            for variant, weights in variants:
                score = lambda c: c[0] if weights is None else combine(c, columns, weights)
                seen_scores, unseen_scores = score(seen_components), score(unseen_components)
                result = score_report(seen_scores, unseen_scores, evaluation_frames, pairs)
                report = dict(model=name, variant=variant, stage=stage, status='evaluated', smoke=smoke,
                              neural_training_images=0, retrained_on_7223=False,
                              selection=selection, metadata=adapter.metadata, prompts=prompts,
                              device=device, versions=versions(), **result)
                report['robustness'] = dict(status='not_requested')
                if robust:
                    outcomes = {}
                    for perturbation in ('brightness', 'blur', 'noise'):
                        components = collect(adapter, banks, evaluation_frames['seen'], config['batch_size'], perturbation)
                        value = metrics(score(components), torch.tensor(evaluation_frames['seen'].pair_row.to_numpy()), pairs)['top1']
                        outcomes[perturbation] = dict(top1=value, drop=round(result['seen_top1']-value, 2))
                    report['robustness'] = dict(status='evaluated', values=outcomes,
                        limitation='Perturb original RGB before native preprocessing, with PIL quantization; differs from legacy post-resize tensor perturbations.')
                report.update(seconds_since_model_start=time.perf_counter()-started,
                              peak_gpu_memory_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else None)
                label = name + '_' + variant
                write_json(path / (label + '.json'), report)
                save_confusion(path, label, seen_scores, evaluation_frames['seen'], pairs)
                reports.append(report)
        except Exception as error:
            write_json(path / (name + '_error.json'), dict(error=type(error).__name__, message=str(error), settings_reduced=False))
            raise
        finally:
            del adapter
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
    write_json(path / 'comparison.json', reports)
    keys = ['model', 'variant', 'stage', 'smoke', 'neural_training_images', 'seen_top1', 'seen_top5',
            'unseen_top1', 'unseen_top5', 'harmonic_mean', 'crop_acc', 'disease_acc', 'unseen_split_crop_acc', 'unseen_split_disease_acc']
    keys += [name + '_' + key for name in ('unseen_crop', 'unseen_disease', 'unseen_combination') for key in ('samples','top1','top5')]
    pd.DataFrame([{k:r[k] for k in keys} for r in reports]).to_csv(path / 'comparison.csv', index=False, na_rep='N/A')
    rows = [dict(model=r['model'], variant=r['variant'], **{k:v for k,v in row.items() if k != 'per_class'})
            for r in reports for row in r['ood_breakdown']]
    pd.DataFrame(rows).to_csv(path / 'ood_breakdown.csv', index=False, na_rep='N/A')
    return path, reports
