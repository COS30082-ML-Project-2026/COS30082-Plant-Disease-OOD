"""One training update per CLIP configuration, independent of formal experiments."""
import gc
import json
import time

import torch
import torch.nn.functional as F


def _step(model, tokens, images, labels, optimizer, seen_count, full):
    model.train(full)
    optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(full):
        text = F.normalize(model.encode_text(tokens[:seen_count]).float(), dim=1)
    features = F.normalize(model.encode_image(images).float(), dim=1)
    logits = model.logit_scale.exp() * features @ text.T
    loss = F.cross_entropy(logits, labels)
    if not torch.isfinite(loss):
        raise RuntimeError('Non-finite training loss')
    loss.backward()
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable):
        raise RuntimeError('Missing or non-finite trainable gradients')
    if full and model.token_embedding.weight.grad is None:
        raise RuntimeError('Full fine-tuning did not reach the text encoder')
    changed_parameter = next((p for p in trainable if p.grad.abs().max() > 0), None)
    if changed_parameter is None:
        raise RuntimeError('Every gradient is zero')
    before = changed_parameter.detach().clone()
    optimizer.step()
    if torch.equal(before, changed_parameter.detach()):
        raise RuntimeError('Optimizer did not change the checked parameter')
    return float(loss.detach()), tuple(logits.shape)


def run_training_smoke(model, tokens, train_loader, cfg, hp, freeze_blocks, initial_path):
    """Test 1/3/6/full on one image each; restore initial weights even on failure.

    Run after Model setup and before any training. All four tests use the same
    transformed image and initial pretrained weights. This is a mechanics check,
    not a model comparison or an accuracy measurement.
    """
    saved_flags = [p.requires_grad for p in model.parameters()]
    saved_mode = model.training
    original_cpu_rng = torch.get_rng_state()
    original_cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    reports = []
    images, _, _, labels = next(iter(train_loader))
    images, labels = images[:1].to(cfg.DEVICE), labels[:1].to(cfg.DEVICE)
    optimizer = None
    try:
        for kind in [*hp['partial_depths'], 'full']:
            started = time.perf_counter()
            cfg.set_seed(hp['seed'])
            model.load_state_dict(torch.load(initial_path, map_location='cpu', weights_only=True))
            if kind == 'full':
                model.requires_grad_(True)
            else:
                freeze_blocks(kind)
            count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f'Starting {kind}: 1 image, 1 optimizer update; {count:,} trainable parameters', flush=True)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                          lr=hp['lr'], weight_decay=hp['weight_decay'])
            loss, shape = _step(model, tokens, images, labels, optimizer, cfg.NUM_SEEN_PAIRS, kind == 'full')
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            result = dict(experiment=str(kind), status='PASS', loss=loss, logits_shape=shape,
                          seconds=round(time.perf_counter()-started, 2), trainable_parameters=count,
                          peak_cuda_mb=round(torch.cuda.max_memory_allocated()/1024**2, 1)
                          if torch.cuda.is_available() else None)
            reports.append(result)
            print(result, flush=True)
            model.zero_grad(set_to_none=True)
            optimizer = None
            gc.collect()
    finally:
        optimizer = None
        model.zero_grad(set_to_none=True)
        model.load_state_dict(torch.load(initial_path, map_location='cpu', weights_only=True))
        for parameter, flag in zip(model.parameters(), saved_flags):
            parameter.requires_grad = flag
        model.train(saved_mode)
        torch.set_rng_state(original_cpu_rng)
        if original_cuda_rng is not None:
            torch.cuda.set_rng_state_all(original_cuda_rng)
        gc.collect()
    output = initial_path.parent / 'single_batch_smoke.json'
    output.write_text(json.dumps(reports, indent=2), encoding='utf-8')
    print('All four training smoke tests passed. Initial weights restored.', flush=True)
    return reports
