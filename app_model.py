"""The model the app uses. Copy the model class and predict() from the best notebook here.

Right now this is Baseline 2 (multi-task CNN). After the 15 Nov cutoff, swap in the best model:
1. Replace MultiTaskNet with the best notebook's model class.
2. Change BEST_MODEL_NAME.
3. Make predict_scores() and image_features() match that model.
4. Run `python build_index.py` again, because the features change with the model.
"""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.config import CHECKPOINT_DIR, DEVICE, NUM_CROPS, NUM_DISEASES

BEST_MODEL_NAME = "baseline2_multitask"
NORM = "imagenet"  # use "clip" if the best model is CLIP
CHECKPOINT_PATH = CHECKPOINT_DIR / f"{BEST_MODEL_NAME}.pt"
INDEX_PATH = CHECKPOINT_DIR / "train_index.pt"


class MultiTaskNet(nn.Module):
    """Copied from baselines/baseline2_multitask.ipynb."""

    def __init__(self, backbone_name, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        self.crop_head = nn.Linear(self.backbone.num_features, NUM_CROPS)
        self.disease_head = nn.Linear(self.backbone.num_features, NUM_DISEASES)

    def forward(self, images):
        features = self.backbone(images)
        return self.crop_head(features), self.disease_head(features)


def load_model() -> nn.Module:
    """Build the model from the checkpoint's own HP, so settings always match training."""
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    model = MultiTaskNet(checkpoint["hp"]["backbone"])  # no pretrained download, weights come next
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(DEVICE).eval()


@torch.no_grad()
def predict_scores(model, images, pairs) -> torch.Tensor:
    """Scores [B, 333] in get_pairs() order. Same maths as the notebook's predict()."""
    crop_logits, disease_logits = model(images.to(DEVICE))
    crop_log_prob = F.log_softmax(crop_logits.float(), dim=1)
    disease_log_prob = F.log_softmax(disease_logits.float(), dim=1)
    pairs = pairs.to(DEVICE)
    return (crop_log_prob[:, pairs[:, 0]] + disease_log_prob[:, pairs[:, 1]]).cpu()


@torch.no_grad()
def image_features(model, images) -> torch.Tensor:
    """Unit-length backbone features, so a dot product is the cosine similarity."""
    return F.normalize(model.backbone(images.to(DEVICE)).float(), dim=1).cpu()
