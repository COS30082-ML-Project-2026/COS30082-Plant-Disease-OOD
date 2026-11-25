"""Data loading for DPD-Small. The only way notebooks should read images or labels."""
from functools import lru_cache
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from common import config

# CSV columns by position, so header spelling in the files does not matter
COLUMNS = ["image", "crop", "disease", "pair_label", "crop_name", "disease_name", "pair_name"]


def read_csv(csv_path, *, final_evaluation=False) -> pd.DataFrame:
    """Read a metadata CSV and drop the header row if the file has one."""
    if Path(csv_path).resolve() in {Path(config.TEST_SEEN_CSV).resolve(), Path(config.TEST_UNSEEN_CSV).resolve()} and not final_evaluation:
        raise ValueError('Official test metadata is available only with final_evaluation=True after selection is frozen.')
    frame = pd.read_csv(csv_path, header=None, names=COLUMNS, dtype=str)
    if not frame.loc[0, "crop"].strip().isdigit():
        frame = frame.iloc[1:]
    for column in ["crop", "disease", "pair_label"]:
        frame[column] = frame[column].astype(int)
    return frame.reset_index(drop=True)


@lru_cache(maxsize=2)
def _pair_table(final_evaluation=False) -> pd.DataFrame:
    """323 training-derived rows; append 10 unseen rows only for final evaluation."""
    seen = read_csv(config.TRAIN_CSV)
    keep = ["crop", "disease", "pair_name", "pair_label"]
    seen_pairs = seen[keep].drop_duplicates(["crop", "disease"]).sort_values("pair_label")
    # Unseen CSV labels may reuse seen numbers, so pairs are keyed by (crop, disease)
    assert len(seen_pairs) == config.NUM_SEEN_PAIRS, f"{len(seen_pairs)} seen pairs found"
    if not final_evaluation:
        return seen_pairs.reset_index(drop=True)
    unseen = read_csv(config.TEST_UNSEEN_CSV, final_evaluation=True)
    unseen_pairs = unseen[keep].drop_duplicates(["crop", "disease"]).sort_values("pair_label")
    assert len(unseen_pairs) == config.NUM_UNSEEN_PAIRS, f"{len(unseen_pairs)} unseen pairs found"
    assert not set(zip(seen_pairs.crop, seen_pairs.disease)) & set(zip(unseen_pairs.crop, unseen_pairs.disease)), 'Unseen pairs overlap training.'
    table = pd.concat([seen_pairs, unseen_pairs]).reset_index(drop=True)
    return table


def get_pairs(*, final_evaluation=False) -> torch.LongTensor:
    """[323, 2] for development; [333, 2] only on explicit final evaluation."""
    return torch.tensor(_pair_table(final_evaluation)[["crop", "disease"]].values, dtype=torch.long)


def get_pair_names(*, final_evaluation=False) -> list:
    """Pair names in the same order as get_pairs()."""
    return _pair_table(final_evaluation)["pair_name"].tolist()


def get_crop_and_disease_names() -> tuple:
    """Concept names from training metadata only, including during final evaluation."""
    every_row = read_csv(config.TRAIN_CSV)
    crop_names = dict(zip(every_row["crop"].tolist(), every_row["crop_name"]))
    disease_names = dict(zip(every_row["disease"].tolist(), every_row["disease_name"]))
    return crop_names, disease_names


def load_split(csv_path, *, final_evaluation=False) -> pd.DataFrame:
    """Metadata plus pair_row, the row of each image's pair in get_pairs()."""
    frame = read_csv(csv_path, final_evaluation=final_evaluation)
    row_of_pair = {(c, d): i for i, (c, d) in enumerate(get_pairs(final_evaluation=final_evaluation).tolist())}
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


def _make_loader(frame, transform, norm, batch_size, shuffle, role=None):
    generator = torch.Generator().manual_seed(config.SEED)
    dataset = PlantDataset(frame, transform, norm)
    dataset.role = role
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=config.NUM_WORKERS, pin_memory=torch.cuda.is_available(), generator=generator,
        persistent_workers=config.NUM_WORKERS > 0,
    )


def get_loaders(batch_size: int, norm: str = "imagenet", debug: bool = False) -> tuple:
    """Train/validation only; legacy tuple ends with None, None to avoid test access."""
    train_transform, test_transform = get_transforms(norm)
    train_frame, val_frame = _split_train_val(load_split(config.TRAIN_CSV))
    if debug:
        train_frame = _subset(train_frame, config.DEBUG_TRAIN_SIZE)
        val_frame = _subset(val_frame, config.DEBUG_VAL_SIZE)
    return (
        _make_loader(train_frame, train_transform, norm, batch_size, shuffle=True, role='train'),
        _make_loader(val_frame, test_transform, norm, batch_size, shuffle=False, role='seen_dev'),
        None, None,
    )


def get_final_test_loaders(batch_size, norm='imagenet', debug=False, *, final_evaluation=False):
    """Explicit final-only access, after checkpoint and prediction rule are frozen."""
    if not final_evaluation:
        raise ValueError('Freeze selection first, then pass final_evaluation=True.')
    train = load_split(config.TRAIN_CSV)
    seen = load_split(config.TEST_SEEN_CSV, final_evaluation=True)
    unseen = load_split(config.TEST_UNSEEN_CSV, final_evaluation=True)
    frames = [train, seen, unseen]
    for i, first in enumerate(frames):
        if not first.image.is_unique:
            raise ValueError('Duplicate image filename within a split.')
        for second in frames[i+1:]:
            if set(first.image) & set(second.image):
                raise ValueError('Image filename overlap between train and official tests.')
    if not set(seen.pair_row) <= set(train.pair_row):
        raise ValueError('Seen test has a combination absent from training.')
    if debug:
        seen, unseen = _subset(seen, config.DEBUG_TEST_SIZE), _subset(unseen, config.DEBUG_TEST_SIZE)
    _, transform = get_transforms(norm)
    return (_make_loader(seen, transform, norm, batch_size, False, role='official_seen'),
            _make_loader(unseen, transform, norm, batch_size, False, role='official_unseen'))
