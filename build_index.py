"""Run once after training: python build_index.py

Saves one feature vector per training image, plus the pair table and names.
The app uses this file for "most similar training images", and it does not need the CSVs afterwards.
"""
import torch
from torch.utils.data import DataLoader

from app_model import INDEX_PATH, NORM, image_features, load_model
from common.config import NUM_SEEN_PAIRS, TRAIN_CSV
from common.dataset import (PlantDataset, get_crop_and_disease_names, get_pair_names, get_pairs,
                            get_transforms, load_split)

BATCH_SIZE = 64


def main():
    model = load_model()
    _, test_transform = get_transforms(NORM)  # same transform as testing, no random augmentation
    train_frame = load_split(TRAIN_CSV)
    loader = DataLoader(PlantDataset(train_frame, test_transform, NORM), batch_size=BATCH_SIZE)
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
    INDEX_PATH.parent.mkdir(exist_ok=True)
    torch.save(index, INDEX_PATH)
    print(f"\nSaved {len(train_frame)} features to {INDEX_PATH}")


if __name__ == "__main__":
    main()
