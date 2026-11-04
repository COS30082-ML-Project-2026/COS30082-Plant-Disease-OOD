"""Controlled CLIP training with a frozen original teacher and shared validation."""
import gc
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from approaches.clip_improvement_core import configure_visual, require, seed_all, strict_load


def validation(model, text, loader, pairs):
    from approaches.clip_improvement_core import make_predict
    predict = make_predict(model, text, pairs)
    correct, count, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for images, _, _, labels in loader:
            scores = predict(images).float().cpu()
            require(((labels >= 0) & (labels < 323)).all(), 'Validation labels must be seen pair rows.')
            loss_sum += F.cross_entropy(scores[:, :323], labels, reduction='sum').item()
            correct += (scores.argmax(1) == labels).sum().item()
            count += len(labels)
    require(count > 0, 'Validation split empty; check CSVs.')
    return dict(top1=round(100 * correct / count, 2), loss=loss_sum / count, samples=count)


def precision_context(device, precision):
    require(precision in ('float32', 'bf16'), f'Unsupported precision {precision}.')
    if precision == 'float32':
        return nullcontext()
    require(device.type == 'cuda' and torch.cuda.is_bf16_supported(),
            'bf16 requires a supported CUDA GPU. Select float32 before starting a new run.')
    return torch.autocast('cuda', dtype=torch.bfloat16)


def check_gradients(model, teacher, names):
    params = dict(model.named_parameters())
    require(all(params[n].grad is not None and torch.isfinite(params[n].grad).all() for n in names),
            'Trainable gradients missing/non-finite; inspect loss, freeze selection or precision.')
    require(all(p.grad is None for p in model.parameters() if not p.requires_grad),
            'A frozen text/logit/projection parameter has gradients.')
    require(all(not p.requires_grad and p.grad is None for p in teacher.parameters()),
            'Teacher is not completely frozen.')


def training_step(engine, images, labels, optimizer, text, regularization, names):
    optimizer.zero_grad(set_to_none=True)
    device, model = engine.device, engine.model
    images, labels = images.to(device), labels.to(device)
    require(((labels >= 0) & (labels < 323)).all(), 'Training target outside 323 seen pairs.')
    with precision_context(device, engine.hp['precision']):
        features = F.normalize(model.encode_image(images).float(), dim=1)
        logits = model.logit_scale.exp().float() * features @ text[:323].T
        ce = F.cross_entropy(logits, labels)
        preservation = preservation_loss(engine.teacher, images, features) if regularization else ce.new_zeros(())
        loss = ce + regularization * preservation
    require(torch.isfinite(loss), 'Training loss is non-finite.')
    loss.backward()
    check_gradients(model, engine.teacher, names)
    optimizer.step()
    return dict(loss=loss.item(), ce=ce.item(), preservation=preservation.item())


def preservation_loss(teacher, images, student_features):
    with torch.no_grad():
        original = F.normalize(teacher.encode_image(images.to(next(teacher.parameters()).device)).float(), dim=1)
    require(original.shape == student_features.shape, 'Teacher/student embedding dimensions differ.')
    return (1 - (student_features * original.to(student_features.device)).sum(1)).mean()


def training_epoch(engine, loader, text, optimizer, regularization, names):
    engine.model.eval()  # Matches the existing Partial-6 frozen-text training convention.
    totals, count = dict(loss=0.0, ce=0.0, preservation=0.0), 0
    for images, _, _, labels in loader:
        values = training_step(engine, images, labels, optimizer, text, regularization, names)
        for key, value in values.items():
            totals[key] += value * len(labels)
        count += len(labels)
    require(count > 0, 'Training split empty; check CSVs.')
    return {key: value / count for key, value in totals.items()}


def fit_candidate(engine, variant, regularization, style, candidate_id):
    seed_all(engine.hp['seed'])
    train, val, _, _ = engine.loaders()
    strict_load(engine.model, engine.base, candidate_id + ' pretrained reset')
    engine.model.zero_grad(set_to_none=True)
    names = configure_visual(engine.model, variant)
    text, history, best, best_state = engine.text(style), [], -1.0, None
    optimizer = torch.optim.AdamW([p for p in engine.model.parameters() if p.requires_grad],
                                 lr=engine.hp['lr'], weight_decay=engine.hp['weight_decay'])
    for epoch in range(engine.hp['epochs']):
        started = time.perf_counter()
        metrics = training_epoch(engine, train, text, optimizer, regularization, names)
        scores = validation(engine.model, text, val, engine.pairs)
        history.append(dict(epoch=epoch+1, **metrics, val_top1=scores['top1'], val_loss=scores['loss'],
                            seconds=time.perf_counter()-started))
        print(candidate_id, history[-1], flush=True)
        if scores['top1'] > best:
            best, best_state = scores['top1'], {k: v.detach().cpu().clone() for k, v in engine.model.state_dict().items()}
    return finish_candidate(engine, history, best, best_state, names, candidate_id, regularization, style)


def finish_candidate(engine, history, best, state, names, candidate_id, regularization, style):
    require(state is not None, f'{candidate_id}: no valid checkpoint selected.')
    selected = max(history, key=lambda row: row['val_top1'])
    record = dict(candidate=candidate_id, lambda_preservation=regularization, prompt_style=style,
                  val_top1=best, best_epoch=selected['epoch'], history=history, trainable_names=names,
                  trainable_parameters=sum(p.numel() for p in engine.model.parameters() if p.requires_grad),
                  frozen_parameters=sum(p.numel() for p in engine.model.parameters() if not p.requires_grad),
                  initialization='original OpenAI CLIP', text_frozen=True, logit_scale_frozen=True,
                  total_seconds=sum(row['seconds'] for row in history))
    engine.model.zero_grad(set_to_none=True)
    gc.collect()
    return record, state
