"""Shared evaluation for test-independent 9625-pair candidate protocol.

The only legitimate source of official unseen identities is the *ground truth*
labels on get_final_test_loaders(), accessed after all development is complete.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision.transforms.functional as TF

from common import config
from common.dataset import get_seen_pair_table

PERTURBATIONS = {
    "brightness": lambda x: TF.adjust_brightness(x, 0.6),
    "blur": lambda x: TF.gaussian_blur(x, kernel_size=9, sigma=2.0),
    "noise": lambda x: (x + 0.08 * torch.randn_like(x)).clamp(0, 1),
}


@torch.no_grad()
def collect_scores(predict, loader, perturb=None) -> tuple:
    """Returns [N,9625] scores and [N] Cartesian IDs (crop * 175 + disease)."""
    if loader is None:
        raise ValueError("Missing loader: did you call get_final_test_loaders()?")
    score_batches, label_batches = [], []
    mean = torch.tensor(loader.dataset.norm_mean, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(loader.dataset.norm_std, dtype=torch.float32).view(1, 3, 1, 1)
    for images, _, _, pair_ids in loader:
        if perturb is not None:
            pixels = (images.float() * std + mean).clamp(0.0, 1.0)
            images = (perturb(pixels).clamp(0.0, 1.0) - mean) / std
        batch_scores = predict(images).detach().float().cpu()
        if batch_scores.ndim != 2 or batch_scores.shape != (len(images), config.NUM_PAIRS):
            raise ValueError(f"predict must return [batch_size, {config.NUM_PAIRS}], "
                             f"got {tuple(batch_scores.shape)}")
        score_batches.append(batch_scores)
        label_batches.append(pair_ids.long().cpu())
    if not score_batches:
        raise ValueError("Empty evaluation loader")
    scores, labels = torch.cat(score_batches), torch.cat(label_batches)
    if not labels.ge(0).all() or not labels.lt(config.NUM_PAIRS).all():
        raise ValueError("Ground-truth Cartesian pair ID out of range")
    return scores, labels


def top_k_accuracy(scores: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    if k < 1 or k > scores.shape[1]:
        raise ValueError(f"Invalid k={k} for {scores.shape[1]} candidates")
    best = scores.topk(k, dim=1).indices
    return round(100.0 * (best == labels[:, None]).any(dim=1).float().mean().item(), 2)


def harmonic_mean(seen_top1: float, unseen_top1: float) -> float:
    denom = seen_top1 + unseen_top1
    return round(2 * seen_top1 * unseen_top1 / denom, 2) if denom else 0.0


def val_accuracy(predict, val_loader) -> float:
    """Seen-development Top-1 (all 9625 candidates), no test CSV reads."""
    if getattr(val_loader.dataset, "role", None) not in ("seen_dev",):
        raise ValueError("val_accuracy requires a training-derived seen-dev loader")
    scores, labels = collect_scores(predict, val_loader)
    return top_k_accuracy(scores, labels, 1)


def development_metrics(predict, seen_val_loader, pseudo_unseen_loader) -> dict:
    """Use only TRAIN-derived development splits for hyperparameter selection."""
    if (getattr(seen_val_loader.dataset, "role", None) != "seen_dev" or
            getattr(pseudo_unseen_loader.dataset, "role", None) != "pseudo_unseen_dev"):
        raise ValueError("Development metrics require train-only seen/pseudo-unseen loaders")
    seen_scores, seen_y = collect_scores(predict, seen_val_loader)
    held_scores, held_y = collect_scores(predict, pseudo_unseen_loader)
    seen, unseen = top_k_accuracy(seen_scores, seen_y, 1), top_k_accuracy(held_scores, held_y, 1)
    return {
        "seen_top1": seen,
        "pseudo_unseen_top1": unseen,
        "pseudo_unseen_top5": top_k_accuracy(held_scores, held_y, 5),
        "harmonic_mean": harmonic_mean(seen, unseen),
        "candidate_protocol": config.CANDIDATE_PROTOCOL,
    }


def concept_accuracy(scores, labels) -> tuple:
    """Crop/disease accuracy of the *top-ranked pair*."""
    predicted = scores.argmax(dim=1)
    crop = (predicted // config.NUM_DISEASES == labels // config.NUM_DISEASES)
    disease = (predicted % config.NUM_DISEASES == labels % config.NUM_DISEASES)
    return round(100 * crop.float().mean().item(), 2), round(100 * disease.float().mean().item(), 2)


def save_confusion(scores, labels, model_name) -> None: