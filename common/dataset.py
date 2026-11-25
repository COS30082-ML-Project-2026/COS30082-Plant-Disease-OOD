"""DPD-Small datasets, with a strict training / final-test boundary.

NO official test CSV (including test_seen.csv) is read by:
  get_pairs, get_pair_names, get_crop_and_disease_names, load_split(TRAIN_CSV),
  get_train_val_loaders, or get_compositional_dev_loaders.

Use get_final_test_loaders(..., final_evaluation=True) *only after* the model,
checkpoint, hyperparameters and prediction rule are locked.
"""
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from common import config

COLUMNS = ["image", "crop", "disease", "pair_label", "crop_name", "disease_name", "pair_name"]


def read_csv(csv_path) -> pd.DataFrame:
    """Read an explicitly specified CSV. Call with test paths only in final eval."""
    frame = pd.read_csv(csv_path, header=None, names=COLUMNS, dtype=str)
    if frame.empty:
        raise ValueError(f"Empty metadata file: {csv_path}")
    if not str(frame.iloc[0]["crop"]).strip().isdigit():
        frame = frame.iloc[1:]
    for column in ("crop", "disease", "pair_label"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
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