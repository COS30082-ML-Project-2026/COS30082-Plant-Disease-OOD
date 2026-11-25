"""DPD-Small datasets, with a strict training / final-test boundary.

NO official test CSV (including test_seen.csv) is read by:
  get_pairs, get_pair_names, get_crop_and_disease_names, load_split(TRAIN_CSV),
  get_train_val_loaders, or get_compositional_dev_loaders.

Use get_final_test_loaders(..., final_evaluation=True) *only after* the model,
checkpoint, hyperparameters and prediction rule are locked.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from common import config

COLUMNS = ["image", "crop", "disease", "pair_label", "crop_name", "disease_name", "pair_name"]


def read_csv(csv_path, *, final_evaluation: bool = False) -> pd.DataFrame:
    """Read metadata, requiring explicit final evaluation for either official CSV."""
    source = Path(csv_path).resolve()
    official_paths = {Path(config.TEST_SEEN_CSV).resolve(), Path(config.TEST_UNSEEN_CSV).resolve()}
    if source in official_paths and not final_evaluation:
        raise ValueError("Official metadata requires final_evaluation=True after models are locked.")
    frame = pd.read_csv(csv_path, header=None, names=COLUMNS, dtype=str)
    if frame.empty:
        raise ValueError(f"Empty metadata file: {csv_path}")
    label_columns = ("crop", "disease", "pair_label")
    if all(not str(frame.iloc[0][column]).strip().lstrip("+-").isdigit()
           for column in label_columns):
        frame = frame.iloc[1:].copy()
    if frame.empty:
        raise ValueError(f"No records in metadata file: {csv_path}")
    for column in label_columns:
        values = pd.to_numeric(frame[column], errors="raise")
        if not (values.notna() & (values % 1 == 0)).all():
            raise ValueError(f"{csv_path}: {column} must contain integer IDs")
        frame[column] = values.astype(np.int64)
    frame["image"] = frame["image"].str.strip().str.replace("\\", "/", regex=False)
    if frame["image"].isna().any() or frame["image"].eq("").any():
        raise ValueError(f"{csv_path}: missing image path")
    frame["image"] = frame["image"].map(lambda value: Path(value).as_posix())
    if frame["image"].duplicated().any():
        raise ValueError(f"{csv_path}: duplicate image path")
    return frame.reset_index(drop=True)


def _check_concepts(frame: pd.DataFrame) -> None:
    if not frame["crop"].between(0, config.NUM_CROPS - 1).all():
        raise ValueError("Crop ID outside 0..54")
    if not frame["disease"].between(0, config.NUM_DISEASES - 1).all():
        raise ValueError("Disease ID outside 0..174")


def pair_id(crop_id: int, disease_id: int) -> int:
    """Stable, test-independent 0..9624 mapping."""
    if not (0 <= int(crop_id) < config.NUM_CROPS and
            0 <= int(disease_id) < config.NUM_DISEASES):
        raise ValueError(f"Invalid crop/disease IDs: {(crop_id, disease_id)}")
    return int(crop_id) * config.NUM_DISEASES + int(disease_id)


def get_pairs() -> torch.LongTensor:
    """[9625, 2]; all Cartesian candidates, built WITHOUT opening test CSVs."""
    return torch.cartesian_prod(
        torch.arange(config.NUM_CROPS, dtype=torch.long),
        torch.arange(config.NUM_DISEASES, dtype=torch.long),
    )


@lru_cache(maxsize=1)
def get_seen_pair_table() -> pd.DataFrame:
    """The 323 *training* pair identities, sorted by their CSV pair label."""
    frame = read_csv(config.TRAIN_CSV)
    _check_concepts(frame)
    seen = (frame[COLUMNS[1:]].drop_duplicates(["crop", "disease"])
            .sort_values(["pair_label", "crop", "disease"]).reset_index(drop=True))
    if len(seen) != config.NUM_SEEN_PAIRS:
        raise ValueError(
            f"Training contains {len(seen)} unique pairs, expected {config.NUM_SEEN_PAIRS}"
        )
    seen["pair_row"] = seen["crop"] * config.NUM_DISEASES + seen["disease"]
    return seen


@lru_cache(maxsize=1)
def get_crop_and_disease_names() -> tuple:
    """Concept names from *train.csv only*, never official test metadata.

    IDs not named in train.csv receive neutral fallback tokens, e.g. 'crop_12'.
    These fallbacks must not be silently replaced using test-unseen labels.
    """
    frame = read_csv(config.TRAIN_CSV)
    _check_concepts(frame)
    crop_names = dict(zip(frame["crop"], frame["crop_name"]))
    disease_names = dict(zip(frame["disease"], frame["disease_name"]))
    return (
        {i: str(crop_names.get(i, f"crop_{i}")) for i in range(config.NUM_CROPS)},
        {i: str(disease_names.get(i, f"disease_{i}")) for i in range(config.NUM_DISEASES)},
    )


def get_pair_names() -> list:
    """9625 pair names, constructed from *training-only* concept names."""
    crops, diseases = get_crop_and_disease_names()
    return [f"{crops[c]} + {diseases[d]}"
            for c in range(config.NUM_CROPS)
            for d in range(config.NUM_DISEASES)]


def load_split(csv_path, *, final_evaluation: bool = False) -> pd.DataFrame:
    """Attach crop-major score-column IDs, independent of CSV composition labels."""
    frame = read_csv(csv_path, final_evaluation=final_evaluation)
    _check_concepts(frame)
    return frame.assign(pair_row=frame["crop"] * config.NUM_DISEASES + frame["disease"])


def get_transforms(norm: str = "imagenet") -> tuple:
    """Baseline image preprocessing with the requested channel normalization."""
    normalise = transforms.Normalize(*config.NORM_STATS[norm])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        normalise,
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(config.RESIZE_SIZE),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToTensor(),
        normalise,
    ])
    return train_transform, eval_transform


class PlantDataset(Dataset):
    """Return (image, crop ID, disease ID, Cartesian pair ID) for each image."""

    def __init__(self, frame: pd.DataFrame, transform, norm: str, *, role=None):
        self.frame = frame.reset_index(drop=True)
        _check_concepts(self.frame)
        self.transform = transform
        self.norm_mean, self.norm_std = config.NORM_STATS[norm]
        self.role = role
        self.targets = self.frame[["crop", "disease"]].to_numpy(dtype=np.int64)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        filename = self.frame.at[index, "image"]
        path = config.IMAGE_DIR / filename
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        crop, disease = self.targets[index]
        return tensor, int(crop), int(disease), pair_id(crop, disease)


def _split_train_val(train_frame: pd.DataFrame, *, seed: int | None = None) -> tuple:
    """Hold out seen images while keeping a fit example of every composition."""
    if not 0 < config.VAL_FRACTION < 1:
        raise ValueError("VAL_FRACTION must be between 0 and 1")
    rng = np.random.default_rng(config.SEED if seed is None else seed)
    val_indices = []
    for _, group in train_frame.groupby("pair_row", sort=True):
        count = min(len(group) - 1, max(1, round(len(group) * config.VAL_FRACTION)))
        if count:
            val_indices.extend(rng.choice(group.index.to_numpy(), count, replace=False))
    return train_frame.drop(index=val_indices), train_frame.loc[val_indices]


def _subset(frame, size):
    return frame.sample(min(size, len(frame)), random_state=config.SEED)


def _make_loader(frame, transform, norm, batch_size, shuffle, *, role=None):
    if frame.empty:
        raise ValueError(f"No images available for the {role or 'requested'} split")
    generator = torch.Generator().manual_seed(config.SEED)
    return DataLoader(
        PlantDataset(frame, transform, norm, role=role),
        batch_size=batch_size, shuffle=shuffle, num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(), generator=generator,
        persistent_workers=config.NUM_WORKERS > 0,
    )


def get_train_val_loaders(batch_size: int, norm: str = "imagenet", debug: bool = False) -> tuple:
    """Return fit and seen validation loaders from the training CSV only."""
    train_transform, eval_transform = get_transforms(norm)
    fit, seen_val = _split_train_val(load_split(config.TRAIN_CSV))
    if debug:
        fit = _subset(fit, config.DEBUG_TRAIN_SIZE)
        seen_val = _subset(seen_val[seen_val["pair_row"].isin(fit["pair_row"])], config.DEBUG_VAL_SIZE)
    return (
        _make_loader(fit, train_transform, norm, batch_size, True, role="train"),
        _make_loader(seen_val, eval_transform, norm, batch_size, False, role="seen_dev"),
    )


def get_compositional_dev_loaders(batch_size: int, norm: str = "imagenet", *,
                                  holdout_pairs: int = 10, min_images: int = 5,
                                  seed: int | None = None) -> tuple:
    """Return fit, seen-dev and proxy-unseen-dev loaders from the training CSV."""
    if holdout_pairs < 1 or min_images < 1:
        raise ValueError("holdout_pairs and min_images must both be positive")
    seed = config.SEED if seed is None else seed
    frame = load_split(config.TRAIN_CSV)
    pairs = frame.groupby(["pair_row", "crop", "disease"], sort=True).size().rename("images").reset_index()
    eligible = list(pairs.loc[pairs["images"] >= min_images].itertuples(index=False))
    crop_counts = pairs.groupby("crop").size().to_dict()
    disease_counts = pairs.groupby("disease").size().to_dict()
    rng = np.random.default_rng(seed)
    for _ in range(64):
        crops, diseases = crop_counts.copy(), disease_counts.copy()
        held = []
        for index in rng.permutation(len(eligible)):
            pair = eligible[index]
            if crops[pair.crop] > 1 and diseases[pair.disease] > 1:
                held.append(pair.pair_row)
                crops[pair.crop] -= 1
                diseases[pair.disease] -= 1
                if len(held) == holdout_pairs:
                    break
        if len(held) == holdout_pairs:
            break
    else:
        raise ValueError("Could not construct the requested holdout while keeping both concepts in fit")

    held_mask = frame["pair_row"].isin(held)
    proxy_unseen = frame[held_mask]
    fit, seen_val = _split_train_val(frame[~held_mask], seed=seed)
    train_transform, eval_transform = get_transforms(norm)
    return (
        _make_loader(fit, train_transform, norm, batch_size, True, role="train"),
        _make_loader(seen_val, eval_transform, norm, batch_size, False, role="seen_dev"),
        _make_loader(proxy_unseen, eval_transform, norm, batch_size, False, role="pseudo_unseen_dev"),
    )


def get_final_test_loaders(batch_size: int, norm: str = "imagenet", debug: bool = False, *,
                           final_evaluation: bool = False) -> tuple:
    """Open the two official tests only on an explicit post-lock evaluation call."""
    if not final_evaluation:
        raise ValueError("Lock the model/checkpoint first, then pass final_evaluation=True")
    fit = load_split(config.TRAIN_CSV)
    seen = load_split(config.TEST_SEEN_CSV, final_evaluation=True)
    unseen = load_split(config.TEST_UNSEEN_CSV, final_evaluation=True)
    fit_pairs = set(fit["pair_row"])
    if not set(seen["pair_row"]) <= fit_pairs or set(unseen["pair_row"]) & fit_pairs:
        raise ValueError("Official seen/unseen pair labels do not match the training split")
    fit_images, seen_images, unseen_images = (
        set(frame["image"]) for frame in (fit, seen, unseen)
    )
    if fit_images & (seen_images | unseen_images) or seen_images & unseen_images:
        raise ValueError("Image paths overlap between training and/or official test splits")
    if debug:
        seen, unseen = (_subset(frame, config.DEBUG_TEST_SIZE) for frame in (seen, unseen))
    _, eval_transform = get_transforms(norm)
    return (
        _make_loader(seen, eval_transform, norm, batch_size, False, role="official_seen"),
        _make_loader(unseen, eval_transform, norm, batch_size, False, role="official_unseen"),
    )


def get_loaders(batch_size: int, norm: str = "imagenet", debug: bool = False) -> tuple:
    """Legacy tuple shape with training/validation only; use the explicit test helper for final."""
    fit, val = get_train_val_loaders(batch_size, norm, debug)
    return fit, val, None, None
