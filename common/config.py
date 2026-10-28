"""Shared settings. Every notebook imports from here so results stay comparable."""
import os
import random
from pathlib import Path

import numpy as np
import torch

# Paths. Change only these lines if your data lives somewhere else.
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
IMAGE_DIR = DATA_DIR / "DPD_Small_dataset"  # image file names in the CSVs are relative to this folder
TRAIN_CSV = DATA_DIR / "DPD_Small_train.csv"
TEST_SEEN_CSV = DATA_DIR / "DPD_Small_test_seen.csv"
TEST_UNSEEN_CSV = DATA_DIR / "DPD_Small_test_unseen.csv"
RESULTS_DIR = ROOT_DIR / "results"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"

# Fixed by the dataset
NUM_CROPS = 55
NUM_DISEASES = 175
NUM_SEEN_PAIRS = 323
NUM_UNSEEN_PAIRS = 10
NUM_PAIRS = NUM_SEEN_PAIRS + NUM_UNSEEN_PAIRS  # 333

# Fixed by the team
IMAGE_SIZE = 224
RESIZE_SIZE = 256
SEED = 42
VAL_FRACTION = 0.1
DEBUG_TRAIN_SIZE = 200
DEBUG_VAL_SIZE = 50
DEBUG_TEST_SIZE = 100
NUM_WORKERS = min(4, os.cpu_count() or 1)

NORM_STATS = {
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "clip": ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = SEED) -> None:
    """Fix every random source so two runs with the same HP give the same result."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
