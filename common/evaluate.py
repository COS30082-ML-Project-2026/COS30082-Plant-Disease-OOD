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
    if _role(val_loader) != "seen_dev":
        raise ValueError("val_accuracy requires a training-derived seen-dev loader")
    scores, labels = collect_scores(predict, val_loader)
    return top_k_accuracy(scores, labels, 1)


def development_metrics(predict, seen_val_loader, pseudo_unseen_loader) -> dict:
    """Use only TRAIN-derived development splits for hyperparameter selection."""
    if (_role(seen_val_loader) != "seen_dev" or _role(pseudo_unseen_loader) != "pseudo_unseen_dev"):
        raise ValueError("Development metrics require train-only seen/pseudo-unseen loaders")
    seen_scores, seen_y = collect_scores(predict, seen_val_loader)
    held_scores, held_y = collect_scores(predict, pseudo_unseen_loader)
    seen, unseen = top_k_accuracy(seen_scores, seen_y, 1), top_k_accuracy(held_scores, held_y, 1)
    return {
        "seen_top1": seen,
        "seen_top5": top_k_accuracy(seen_scores, seen_y, 5),
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
    """Save a compact seen confusion plot and exact counts for every prediction."""
    seen_pairs = get_seen_pair_table()["pair_row"].tolist()
    observed = pd.DataFrame({
        "true_pair_id": labels.cpu().tolist(),
        "predicted_pair_id": scores.argmax(dim=1).cpu().tolist(),
    })
    if not observed["true_pair_id"].isin(seen_pairs).all():
        raise ValueError("Seen confusion matrix requires only train-observed ground-truth pairs")
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sparse = observed.value_counts(sort=False).rename("count").reset_index()
    sparse.to_csv(config.RESULTS_DIR / f"{model_name}_confusion_sparse.csv", index=False)

    other = -1
    compact_predicted = observed["predicted_pair_id"].where(
        observed["predicted_pair_id"].isin(seen_pairs), other)
    table = pd.crosstab(observed["true_pair_id"], compact_predicted).reindex(
        index=seen_pairs, columns=[*seen_pairs, other], fill_value=0)
    table = table.rename(columns={other: "outside_training_pairs"})
    table.to_csv(config.RESULTS_DIR / f"{model_name}_confusion.csv")
    row_share = table.div(table.sum(axis=1).clip(lower=1), axis=0)

    figure, axis = plt.subplots(figsize=(12, 10))
    image = axis.imshow(row_share.to_numpy(), cmap="viridis", aspect="auto", vmin=0, vmax=1)
    axis.axvline(len(seen_pairs) - 0.5, color="red", linewidth=0.8)
    axis.set(title=f"{model_name}: seen test confusion (row-normalised)",
             xlabel="Predicted pair (last column = outside training)", ylabel="True train pair")
    figure.colorbar(image, ax=axis)
    figure.savefig(config.RESULTS_DIR / f"{model_name}_confusion.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def _role(loader):
    return getattr(getattr(loader, "dataset", None), "role", None)


def robustness(predict, seen_loader, clean_top1) -> dict:
    """Seen Top-1 under each perturbation using the same noise seed for each model."""
    report = {}
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config.SEED)
        for name, perturb in PERTURBATIONS.items():
            scores, labels = collect_scores(predict, seen_loader, perturb)
            top1 = top_k_accuracy(scores, labels, 1)
            report[name] = {"top1": top1, "drop": round(clean_top1 - top1, 2)}
    return report


def evaluate(predict, seen_loader, unseen_loader, model_name, hp, robust=False) -> dict:
    """Report official results for loaders opened explicitly after model selection."""
    if (_role(seen_loader), _role(unseen_loader)) != ("official_seen", "official_unseen"):
        raise ValueError("Use get_final_test_loaders(..., final_evaluation=True) after locking the model")
    seen_scores, seen_labels = collect_scores(predict, seen_loader)
    unseen_scores, unseen_labels = collect_scores(predict, unseen_loader)
    seen_top1 = top_k_accuracy(seen_scores, seen_labels, 1)
    unseen_top1 = top_k_accuracy(unseen_scores, unseen_labels, 1)
    crop_acc, disease_acc = concept_accuracy(seen_scores, seen_labels)
    unseen_crop_acc, unseen_disease_acc = concept_accuracy(unseen_scores, unseen_labels)
    results = {
        "model": model_name,
        "candidate_protocol": config.CANDIDATE_PROTOCOL,
        "candidate_count": config.NUM_PAIRS,
        "seen_top1": seen_top1,
        "seen_top5": top_k_accuracy(seen_scores, seen_labels, 5),
        "unseen_top1": unseen_top1,
        "unseen_top5": top_k_accuracy(unseen_scores, unseen_labels, 5),
        "harmonic_mean": harmonic_mean(seen_top1, unseen_top1),
        "crop_acc": crop_acc,
        "disease_acc": disease_acc,
        "unseen_crop_acc": unseen_crop_acc,
        "unseen_disease_acc": unseen_disease_acc,
        "hp": hp,
    }
    del unseen_scores
    if robust:
        results["robustness"] = robustness(predict, seen_loader, seen_top1)
    save_confusion(seen_scores, seen_labels, model_name)
    with (config.RESULTS_DIR / f"{model_name}.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print(json.dumps({key: value for key, value in results.items() if key != "hp"}, indent=2))
    return results
