"""Run once after training a model:
    python build_index.py                         (every model in checkpoints/)
    python build_index.py baseline1_independent   (one model)

Saves one feature vector per training image, plus the pair table and names, for each model.
The app uses this file for "most similar training images", and it does not need the CSVs afterwards.
"""
import sys

import torch
from torch.utils.data import DataLoader

from app_model import MODELS, available_models, image_features, index_path, load_model
from common.config import NUM_SEEN_PAIRS, TRAIN_CSV
from common.dataset import (PlantDataset, get_crop_and_disease_names, get_pair_names, get_pairs,
                            get_transforms, load_split)

BATCH_SIZE = 64


def build(name):
    print(f"Building index for {name}")
    norm = MODELS[name][2]
    model = load_model(name)
    _, test_transform = get_transforms(norm)  # same transform as testing, no random augmentation
    train_frame = load_split(TRAIN_CSV)
    loader = DataLoader(PlantDataset(train_frame, test_transform, norm), batch_size=BATCH_SIZE)
    features = []
    for batch_number, (images, _, _, _) in enumerate(loader):
        features.append(image_features(model, images))
        print(f"batch {batch_number + 1}/{len(loader)}", end="\r")
    index = {
        "features": torch.cat(features).half(),  # half precision halves the file size
        "image_files": train_frame["image"].tolist(),
        "pair_rows": torch.tensor(train_frame["pair_row"].values),
        "pairs": get_pairs(),
        "pair_names": get_pair_names(),
        "crop_and_disease_names": get_crop_and_disease_names(),
        "num_seen_pairs": NUM_SEEN_PAIRS,
    }
    torch.save(index, index_path(name))
    print(f"\nSaved {len(train_frame)} features to {index_path(name)}")


def main():
    names = sys.argv[1:] or available_models()
    if not names:
        print("No checkpoints found in checkpoints/. Train a model first.")
    for name in names:
        build(name)


if __name__ == "__main__":
    main()