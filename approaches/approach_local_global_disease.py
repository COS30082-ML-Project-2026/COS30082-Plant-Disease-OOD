"""Local–Global Disease Feature Learning with a frozen shared DINOv2 backbone.

Crop prediction uses the image token. Disease prediction combines the image
token with an attention-weighted summary of patch tokens. Development uses
only train.csv and training-derived candidate pairs. Official tests are read
only by the explicit final-evaluation mode, after model selection is fixed.
"""
import argparse
import json
from pathlib import Path

import timm
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from common import config
from common.dataset import (PlantDataset, get_final_test_loaders, get_pairs,
                            get_transforms, load_split)
from common.evaluate import evaluate, harmonic_mean

MODEL = "vit_base_patch14_dinov2.lvd142m"
NAME = "approach_local_global_disease_dinov2"
PROTOCOL = "train_only_local_global_disease_v1"


def prepare(batch_size):
    """Create held-out pairs using training rows only."""
    frame = load_split(config.TRAIN_CSV)
    table = (frame[["pair_row", "crop", "disease", "pair_name"]]
             .drop_duplicates().sort_values("pair_row").reset_index(drop=True))
    if table.pair_row.tolist() != list(range(config.NUM_SEEN_PAIRS)):
        raise ValueError("Training pair rows do not match development candidates")
    remaining, held = frame.copy(), []
    for index in table.sample(frac=1, random_state=config.SEED).index:
        candidate = remaining[remaining.pair_row == index]
        rest = remaining[remaining.pair_row != index]
        crop, disease = table.loc[index, ["crop", "disease"]]
        if len(candidate) >= 10 and (rest.crop == crop).any() and (rest.disease == disease).any():
            held.append(int(index))
            remaining = rest
        if len(held) == 10:
            break
    if len(held) != 10:
        raise ValueError("Could not create ten training-only held-out pairs")
    familiar = remaining.groupby("pair_row").sample(
        frac=config.VAL_FRACTION, random_state=config.SEED)
    train = remaining.drop(familiar.index)
    novel = frame[frame.pair_row.isin(held)]
    assert set(train.pair_row).isdisjoint(held)
    assert set(familiar.pair_row).isdisjoint(held)
    assert set(novel.crop) <= set(train.crop) and set(novel.disease) <= set(train.disease)
    train_tf, test_tf = get_transforms("imagenet")
    def make_loader(rows, transform, shuffle):
        return DataLoader(PlantDataset(rows, transform, "imagenet"),
                          batch_size=batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=False)
    loaders = (make_loader(train.reset_index(drop=True), train_tf, True),
               make_loader(familiar, test_tf, False),
               make_loader(novel, test_tf, False))
    pairs = get_pairs().to(config.DEVICE)
    assert pairs.shape == (config.NUM_SEEN_PAIRS, 2)
    return loaders, pairs, held, table


class LocalGlobalNet(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.backbone = timm.create_model(MODEL, pretrained=pretrained,
                                          num_classes=0, img_size=224)
        self.backbone.requires_grad_(False)
        width = self.backbone.num_features
        self.crop_head = nn.Linear(width, config.NUM_CROPS)
        self.patch_attention = nn.Linear(width, 1)
        self.disease_head = nn.Linear(width * 2, config.NUM_DISEASES)

    def forward(self, images):
        # The backbone is frozen. Avoid storing its activations for backward.
        with torch.no_grad():
            tokens = self.backbone.forward_features(images)
        if tokens.ndim != 3 or tokens.shape[1] <= 1:
            raise RuntimeError("Expected one image token and multiple DINOv2 patch tokens")
        global_feature = tokens[:, 0]
        patches = tokens[:, 1:]
        attention = self.patch_attention(patches).softmax(dim=1)
        local_feature = (attention * patches).sum(dim=1)
        return (self.crop_head(global_feature),
                self.disease_head(torch.cat((global_feature, local_feature), dim=1)))


@torch.no_grad()
def pair_scores(model, images, pairs):
    model.eval()
    crop, disease = model(images.to(config.DEVICE))
    crop = F.log_softmax(crop.float(), dim=1)
    disease = F.log_softmax(disease.float(), dim=1)
    return crop[:, pairs[:, 0]] + disease[:, pairs[:, 1]]


@torch.no_grad()
def top1(model, loader, pairs):
    correct = total = 0
    for images, _, _, targets in loader:
        scores = pair_scores(model, images, pairs)
        correct += int((scores.argmax(dim=1).cpu() == targets).sum())
        total += len(targets)
    return round(100 * correct / total, 2)


def full_loader(batch_size):
    frame = load_split(config.TRAIN_CSV)
    if set(frame.pair_row) != set(range(config.NUM_SEEN_PAIRS)):
        raise ValueError("Expected all 323 training-derived pairs")
    train_transform, _ = get_transforms("imagenet")
    generator = torch.Generator().manual_seed(config.SEED)
    loader = DataLoader(PlantDataset(frame, train_transform, "imagenet"),
                        batch_size=batch_size, shuffle=True, num_workers=0,
                        pin_memory=False, generator=generator)
    return loader, frame


def train_epoch(model, loader, optimizer, criterion):
    model.eval()  # Frozen backbone remains in evaluation mode.
    model.crop_head.train(); model.patch_attention.train(); model.disease_head.train()
    crop_sum = disease_sum = count = 0
    for images, crops, diseases, _ in loader:
        images = images.to(config.DEVICE)
        crop_logits, disease_logits = model(images)
        crop_loss = criterion(crop_logits, crops.to(config.DEVICE))
        disease_loss = criterion(disease_logits, diseases.to(config.DEVICE))
        optimizer.zero_grad(set_to_none=True)
        (crop_loss + disease_loss).backward()
        optimizer.step()
        batch = len(images)
        crop_sum += crop_loss.item() * batch
        disease_sum += disease_loss.item() * batch
        count += batch
    return round(crop_sum / count, 3), round(disease_sum / count, 3)


def optimizer_for(model):
    return torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                             lr=1e-4, weight_decay=1e-4)


def validate(args):
    config.SEED = args.seed
    config.set_seed(args.seed)
    (train, familiar, held_loader), pairs, held, table = prepare(args.batch_size)
    split = {"seed": args.seed, "source": "training CSV only",
             "held_pair_rows": held,
             "held_pair_names": table.loc[held, "pair_name"].tolist(),
             "training_images": len(train.dataset),
             "familiar_validation_images": len(familiar.dataset),
             "held_pair_validation_images": len(held_loader.dataset)}
    print(json.dumps(split, indent=2), flush=True)
    if args.check:
        return
    output = config.RESULTS_DIR / f"{NAME}_pair_validation_seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "split.json").write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    model = LocalGlobalNet(pretrained=True).to(config.DEVICE)
    optimizer = optimizer_for(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_h, history = -1.0, []
    for epoch in range(args.epochs):
        crop_loss, disease_loss = train_epoch(model, train, optimizer, criterion)
        scheduler.step()
        familiar_acc = top1(model, familiar, pairs)
        held_acc = top1(model, held_loader, pairs)
        row = {"epoch": epoch + 1, "familiar_top1": familiar_acc,
               "held_pair_top1": held_acc,
               "harmonic_mean": harmonic_mean(familiar_acc, held_acc),
               "crop_loss": crop_loss, "disease_loss": disease_loss}
        history.append(row)
        print(f"epoch {epoch + 1}/{args.epochs}: {row}", flush=True)
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        if row["harmonic_mean"] > best_h:
            best_h = row["harmonic_mean"]
            (output / "best.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    print(f"Best training-only comparison: {output / 'best.json'}", flush=True)


def fit(args):
    config.SEED = args.seed
    config.set_seed(args.seed)
    train, frame = full_loader(args.batch_size)
    assert get_pairs().shape == (323, 2)
    print(f"Full training: {len(frame)} images, 323 training pairs, {args.epochs} epochs.", flush=True)
    if args.check:
        return
    model = LocalGlobalNet(pretrained=True).to(config.DEVICE)
    optimizer = optimizer_for(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    for epoch in range(args.epochs):
        crop_loss, disease_loss = train_epoch(model, train, optimizer, criterion)
        scheduler.step()
        print(f"epoch {epoch + 1}/{args.epochs}: crop loss {crop_loss}, disease loss {disease_loss}", flush=True)
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.CHECKPOINT_DIR / f"{NAME}_full.pt"
    hp = {"backbone": MODEL, "input_size": 224, "backbone_frozen": True,
          "shared_backbone": True, "local_patch_attention": True,
          "batch_size": args.batch_size, "epochs": args.epochs, "seed": args.seed,
          "lr": 1e-4, "weight_decay": 1e-4, "label_smoothing": 0.1,
          "protocol": PROTOCOL}
    torch.save({"model_name": NAME, "state_dict": model.state_dict(), "hp": hp}, path)
    print(f"Saved {path}. Official tests require a separate final-evaluation command.", flush=True)


def final_evaluation(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    hp = checkpoint["hp"]
    if (hp.get("protocol") != PROTOCOL or hp.get("backbone") != MODEL
            or not hp.get("local_patch_attention") or not hp.get("shared_backbone")):
        raise ValueError("Expected a full-training Local–Global DINOv2 checkpoint")
    config.set_seed(hp["seed"])
    model = LocalGlobalNet(pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(config.DEVICE).eval()
    pairs = get_pairs(final_evaluation=True).to(config.DEVICE)
    seen, unseen = get_final_test_loaders(hp["batch_size"], final_evaluation=True)
    evaluate(lambda images: pair_scores(model, images, pairs), seen, unseen,
             NAME + "_final", hp, robust=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("validate", "full-train", "final-evaluation"),
                        default="validate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--check", action="store_true", help="Check the split without training")
    args = parser.parse_args()
    if args.seed < 0 or args.epochs < 1 or args.batch_size < 1:
        parser.error("Seed must be nonnegative; epochs and batch size must be positive")
    config.NUM_WORKERS = 0
    if args.mode == "final-evaluation":
        if args.check or args.checkpoint is None or not args.checkpoint.is_file():
            parser.error("Final evaluation requires --checkpoint and no --check")
        final_evaluation(args.checkpoint)
    elif args.checkpoint is not None:
        parser.error("--checkpoint is only for final-evaluation")
    elif args.mode == "full-train":
        fit(args)
    else:
        validate(args)


if __name__ == "__main__":
    main()
