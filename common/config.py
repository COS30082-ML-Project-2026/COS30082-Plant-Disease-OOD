"""COS30082 / DPD-Small configuration.

Protocol: official unseen crop-disease pair identities must not enter training,
development, checkpoint selection, prompt construction or candidate generation.
The unseen CSV may be opened only by the explicit final-evaluation loader.
"""
import os
import random
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
IMAGE_DIR = DATA_DIR / "DPD_Small_dataset"
TRAIN_CSV = DATA_DIR / "DPD_Small_train.csv"
TEST_SEEN_CSV = DATA_DIR / "DPD_Small_test_seen.csv"
TEST_UNSEEN_CSV = DATA_DIR / "DPD_Small_test_unseen.csv"
RESULTS_DIR = ROOT_DIR / "results"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"

# Dataset concept vocabularies; IDs are zero-based.
NUM_CROPS = 55
NUM_DISEASES = 175
NUM_SEEN_PAIRS = 323       # number of combinations observed in training
NUM_UNSEEN_PAIRS = 10      # for final reporting only; never a candidate list

# Candidate list is determined without inspecting *any* official test labels.
NUM_CANDIDATE_PAIRS = NUM_CROPS * NUM_DISEASES  # 9625
NUM_PAIRS = NUM_CANDIDATE_PAIRS  # compatibility alias for score-matrix width
CANDIDATE_PROTOCOL = "full_cartesian_train_independent"

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
    "clip": ((0.48145466, 0.4578275, 0.40821073),
             (0.26862954, 0.26130258, 0.27577711)),
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
