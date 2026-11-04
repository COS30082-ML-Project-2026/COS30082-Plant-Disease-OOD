"""Data loading for DPD-Small. The only way notebooks should read images or labels."""
from functools import lru_cache

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from common import config

# CSV columns by position, so header spelling in the files does not matter
COLUMNS = ["image", "crop", "disease", "pair_label", "crop_name", "disease_name", "pair_name"]


def read_csv(csv_path) -> pd.DataFrame:
    """Read a metadata CSV and drop the header row if the file has one."""
    frame = pd.read_csv(csv_path, header=None, names=COLUMNS, dtype=str)
    if not frame.loc[0, "crop"].strip().isdigit():
        frame = frame.iloc[1:]
    for column in ["crop", "disease", "pair_label"]:
        frame[column] = frame[column].astype(int)
    return frame.reset_index(drop=True)


@lru_cache(maxsize=1)
def _pair_table() -> pd.DataFrame:
    """One row per pair: 323 seen pairs (sorted by CSV pair label), then 10 unseen pairs."""
    seen = pd.concat([read_csv(config.TRAIN_CSV), read_csv(config.TEST_SEEN_CSV)])
    unseen = read_csv(config.TEST_UNSEEN_CSV)
    keep = ["crop", "disease", "pair_name", "pair_label"]
    seen_pairs = seen[keep].drop_duplicates(["crop", "disease"]).sort_values("pair_label")
    unseen_pairs = unseen[keep].drop_duplicates(["crop", "disease"]).sort_values("pair_label")
    table = pd.concat([seen_pairs, unseen_pairs]).reset_index(drop=True)
    # Unseen CSV labels may reuse seen numbers, so pairs are keyed by (crop, disease)
    assert len(seen_pairs) == config.NUM_SEEN_PAIRS, f"{len(seen_pairs)} seen pairs found"
    assert len(unseen_pairs) == config.NUM_UNSEEN_PAIRS, f"{len(unseen_pairs)} unseen pairs found"
    return table


def get_pairs() -> torch.LongTensor:
    """[333, 2] tensor. Row i = (crop_id, disease_id) of pair i."""
    return torch.tensor(_pair_table()[["crop", "disease"]].values, dtype=torch.long)


def get_pair_names() -> list:
    """Pair names in the same order as get_pairs()."""
    return _pair_table()["pair_name"].tolist()


def get_crop_and_disease_names() -> tuple:
    """Two dicts: crop_id -> crop name, disease_id -> disease name (all three CSVs)."""
    frames = [read_csv(path) for path in (config.TRAIN_CSV, config.TEST_SEEN_CSV, config.TEST_UNSEEN_CSV)]
    every_row = pd.concat(frames)
    crop_names = dict(zip(every_row["crop"].tolist(), every_row["crop_name"]))
    disease_names = dict(zip(every_row["disease"].tolist(), every_row["disease_name"]))
    return crop_names, disease_names


def load_split(csv_path) -> pd.DataFrame:
    """Metadata plus pair_row, the row of each image's pair in get_pairs()."""
    frame = read_csv(csv_path)
    row_of_pair = {(c, d): i for i, (c, d) in enumerate(get_pairs().tolist())}
    frame["pair_row"] = [row_of_pair[(c, d)] for c, d in zip(frame["crop"], frame["disease"])]
    return frame


def get_transforms(norm: str = "imagenet") -> tuple:
    """Train and test transforms. norm picks ImageNet or CLIP mean and std."""
    normalise = transforms.Normalize(*config.NORM_STATS[norm])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        normalise,
    ])
    test_transform = transforms.Compose([
        transforms.Resize(config.RESIZE_SIZE),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToTensor(),
        normalise,
    ])
    return train_transform, test_transform


class PlantDataset(Dataset):
    """Returns (image, crop_label, disease_label, pair_label) for one row of a split."""

    def __init__(self, frame: pd.DataFrame, transform, norm: str):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.norm_mean, self.norm_std = config.NORM_STATS[norm]  # evaluate() needs these for robust mode

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image = Image.open(config.IMAGE_DIR / row["image"]).convert("RGB")
        return self.transform(image), row["crop"], row["disease"], row["pair_row"]


def _split_train_val(train_frame: pd.DataFrame) -> tuple:
    """10% of each pair goes to val. Pairs with very few images stay in train."""
    val_frame = train_frame.groupby("pair_row").sample(frac=config.VAL_FRACTION, random_state=config.SEED)
    return train_frame.drop(val_frame.index), val_frame


def _subset(frame, size):
    return frame.sample(min(size, len(frame)), random_state=config.SEED)


def _make_loader(frame, transform, norm, batch_size, shuffle):
    generator = torch.Generator().manual_seed(config.SEED)
    return DataLoader(
        PlantDataset(frame, transform, norm), batch_size=batch_size, shuffle=shuffle,
        num_workers=config.NUM_WORKERS, pin_memory=torch.cuda.is_available(), generator=generator,
        persistent_workers=config.NUM_WORKERS > 0,
    )


def get_loaders(batch_size: int, norm: str = "imagenet", debug: bool = False) -> tuple:
    """train_loader, val_loader, seen_loader, unseen_loader. debug=True uses small subsets."""
    train_transform, test_transform = get_transforms(norm)
    train_frame, val_frame = _split_train_val(load_split(config.TRAIN_CSV))
    seen_frame = load_split(config.TEST_SEEN_CSV)
    unseen_frame = load_split(config.TEST_UNSEEN_CSV)
    if debug:
        train_frame = _subset(train_frame, config.DEBUG_TRAIN_SIZE)
        val_frame = _subset(val_frame, config.DEBUG_VAL_SIZE)
        seen_frame = _subset(seen_frame, config.DEBUG_TEST_SIZE)
        unseen_frame = _subset(unseen_frame, config.DEBUG_TEST_SIZE)
    return (
        _make_loader(train_frame, train_transform, norm, batch_size, shuffle=True),
        _make_loader(val_frame, test_transform, norm, batch_size, shuffle=False),
        _make_loader(seen_frame, test_transform, norm, batch_size, shuffle=False),
        _make_loader(unseen_frame, test_transform, norm, batch_size, shuffle=False),
    )
