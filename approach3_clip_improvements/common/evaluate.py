"""One evaluation for every model, so all results in the report are comparable."""
import json

import matplotlib
matplotlib.use("Agg")  # save figures without needing a screen
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision.transforms.functional as TF

from common import config
from common.dataset import get_pair_names, get_pairs

# Robustness tests: simulate field photos (bad light, out of focus, low quality sensor)
PERTURBATIONS = {
    "brightness": lambda x: TF.adjust_brightness(x, 0.6),
    "blur": lambda x: TF.gaussian_blur(x, kernel_size=9, sigma=2.0),
    "noise": lambda x: (x + 0.08 * torch.randn_like(x)).clamp(0, 1),
}


@torch.no_grad()
def collect_scores(predict, loader, perturb=None) -> tuple:
    """Run predict over a loader. Returns scores [N, 333] and true pair rows [N]."""
    all_scores, all_pairs = [], []
    mean = torch.tensor(loader.dataset.norm_mean).view(1, 3, 1, 1)
    std = torch.tensor(loader.dataset.norm_std).view(1, 3, 1, 1)
    for images, _, _, pair_labels in loader:
        if perturb is not None:  # perturb in pixel space, then normalise again
            images = (perturb(images * std + mean) - mean) / std
        all_scores.append(predict(images).float().cpu())
        all_pairs.append(pair_labels)
    return torch.cat(all_scores), torch.cat(all_pairs)


def top_k_accuracy(scores: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """Percent of images whose true pair is in the k highest scores."""
    top_k = scores.topk(k, dim=1).indices
    return round(100 * (top_k == labels[:, None]).any(dim=1).float().mean().item(), 2)


def harmonic_mean(seen_top1: float, unseen_top1: float) -> float:
    """Low if either side is low, so a model cannot win on seen pairs alone."""
    if seen_top1 + unseen_top1 == 0:
        return 0.0
    return round(2 * seen_top1 * unseen_top1 / (seen_top1 + unseen_top1), 2)


def val_accuracy(predict, val_loader) -> float:
    """Top-1 over all 333 pairs on the val split. Use this to pick the best epoch."""
    scores, labels = collect_scores(predict, val_loader)
    return top_k_accuracy(scores, labels, 1)


def concept_accuracy(scores, labels) -> tuple:
    """Crop and disease accuracy read from the top pair, so any model can be scored."""
    pairs = get_pairs()
    predicted, true = pairs[scores.argmax(dim=1)], pairs[labels]
    crop_acc = 100 * (predicted[:, 0] == true[:, 0]).float().mean().item()
    disease_acc = 100 * (predicted[:, 1] == true[:, 1]).float().mean().item()
    return round(crop_acc, 2), round(disease_acc, 2)


def save_confusion(scores, labels, model_name) -> None:
    """Rows: 323 true seen pairs. Columns: all 333 predicted pairs, so mistakes into unseen pairs show."""
    names = get_pair_names()
    counts = torch.zeros(config.NUM_SEEN_PAIRS, config.NUM_PAIRS, dtype=torch.long)
    counts.index_put_((labels, scores.argmax(dim=1)), torch.ones_like(labels), accumulate=True)
    table = pd.DataFrame(counts.numpy(), index=names[:config.NUM_SEEN_PAIRS], columns=names)
    table.to_csv(config.RESULTS_DIR / f"{model_name}_confusion.csv")
    row_share = counts.float() / counts.sum(dim=1, keepdim=True).clamp(min=1)  # rows with few images stay visible
    figure, axis = plt.subplots(figsize=(12, 12))
    image = axis.imshow(row_share.numpy(), cmap="viridis", aspect="auto")
    axis.axvline(config.NUM_SEEN_PAIRS - 0.5, color="red", linewidth=0.8)
    axis.set(title=f"{model_name}: seen test confusion (row-normalised)",
             xlabel="Predicted pair (right of red line = unseen)", ylabel="True pair")
    figure.colorbar(image, ax=axis, fraction=0.046)
    figure.savefig(config.RESULTS_DIR / f"{model_name}_confusion.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def robustness(predict, seen_loader, clean_top1) -> dict:
    """Seen Top-1 under each perturbation, and the drop from clean accuracy."""
    torch.manual_seed(config.SEED)  # same noise for every model
    report = {}
    for name, perturb in PERTURBATIONS.items():
        scores, labels = collect_scores(predict, seen_loader, perturb)
        top1 = top_k_accuracy(scores, labels, 1)
        report[name] = {"top1": top1, "drop": round(clean_top1 - top1, 2)}
    return report


def evaluate(predict, seen_loader, unseen_loader, model_name, hp, robust=False) -> dict:
    """Score a model, print the results and write results/<model_name>.json and confusion files."""
    seen_scores, seen_labels = collect_scores(predict, seen_loader)
    unseen_scores, unseen_labels = collect_scores(predict, unseen_loader)
    assert seen_scores.shape[1] == config.NUM_PAIRS, "predict() must return [B, 333]"
    results = {"model": model_name}
    results["seen_top1"] = top_k_accuracy(seen_scores, seen_labels, 1)
    results["seen_top5"] = top_k_accuracy(seen_scores, seen_labels, 5)
    results["unseen_top1"] = top_k_accuracy(unseen_scores, unseen_labels, 1)
    results["unseen_top5"] = top_k_accuracy(unseen_scores, unseen_labels, 5)
    results["harmonic_mean"] = harmonic_mean(results["seen_top1"], results["unseen_top1"])
    results["crop_acc"], results["disease_acc"] = concept_accuracy(seen_scores, seen_labels)
    results["hp"] = hp
    if robust:
        results["robustness"] = robustness(predict, seen_loader, results["seen_top1"])
    config.RESULTS_DIR.mkdir(exist_ok=True)
    with open(config.RESULTS_DIR / f"{model_name}.json", "w") as file:
        json.dump(results, file, indent=2)
    save_confusion(seen_scores, seen_labels, model_name)
    print(json.dumps({k: v for k, v in results.items() if k != "hp"}, indent=2))
    return results
