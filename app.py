"""Every model the app can use. The app shows a dropdown of the ones whose checkpoint exists.

To add a model (for example an approach after training):
1. Copy its model class from the notebook. It must return (crop_logits, disease_logits)
   from forward() and have a features() method for "most similar training images".
2. Add a builder function and one line to MODELS below.
3. Run `python build_index.py <model name>` once.
"""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.config import CHECKPOINT_DIR, DEVICE, NUM_CROPS, NUM_DISEASES


class MultiTaskNet(nn.Module):
    """Baseline 2. Copied from baselines/baseline2_multitask.ipynb."""

    def __init__(self, backbone_name, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        self.crop_head = nn.Linear(self.backbone.num_features, NUM_CROPS)
        self.disease_head = nn.Linear(self.backbone.num_features, NUM_DISEASES)

    def forward(self, images):
        features = self.backbone(images)
        return self.crop_head(features), self.disease_head(features)

    def features(self, images):
        return self.backbone(images)


class IndependentNets(nn.Module):
    """Baseline 1. Two separate classifiers, as in baselines/baseline1_independent.ipynb."""

    def __init__(self, backbone_name):
        super().__init__()
        self.crop_model = timm.create_model(backbone_name, pretrained=False, num_classes=NUM_CROPS)
        self.disease_model = timm.create_model(backbone_name, pretrained=False, num_classes=NUM_DISEASES)

    def forward(self, images):
        return self.crop_model(images), self.disease_model(images)

    def features(self, images):
        """Both models' pooled features side by side, each scaled to unit length."""
        parts = [F.normalize(m.forward_head(m.forward_features(images), pre_logits=True), dim=1)
                 for m in (self.crop_model, self.disease_model)]
        return torch.cat(parts, dim=1)


def build_baseline1(checkpoint):
    model = IndependentNets(checkpoint["hp"]["backbone"])
    model.crop_model.load_state_dict(checkpoint["state_dict"]["crop"])
    model.disease_model.load_state_dict(checkpoint["state_dict"]["disease"])
    return model


def build_baseline2(checkpoint):
    model = MultiTaskNet(checkpoint["hp"]["backbone"])
    model.load_state_dict(checkpoint["state_dict"])
    return model


# Model name (same as MODEL_NAME in the notebook) -> (label in the app, builder, image normalisation)
MODELS = {
    "baseline2_multitask": ("Baseline 2: multi-task CNN", build_baseline2, "imagenet"),
    "baseline1_independent": ("Baseline 1: two independent CNNs", build_baseline1, "imagenet"),
}


def checkpoint_path(name):
    """Full run if it exists, otherwise the debug run (notebooks save <name>_debug.pt when DEBUG=True)."""
    for run_name in (name, f"{name}_debug"):
        path = CHECKPOINT_DIR / f"{run_name}.pt"
        if path.exists():
            return path
    return None


def index_path(name):
    """Each model has its own index, because each model's features are different."""
    return CHECKPOINT_DIR / f"{checkpoint_path(name).stem}_index.pt"


def available_models():
    """Model names whose checkpoint is in checkpoints/."""
    return [name for name in MODELS if checkpoint_path(name) is not None]


def load_model(name) -> nn.Module:
    """Build the model from the checkpoint's own HP, so settings always match training."""
    checkpoint = torch.load(checkpoint_path(name), map_location="cpu", weights_only=True)
    model = MODELS[name][1](checkpoint)
    return model.to(DEVICE).eval()


@torch.no_grad()
def predict_scores(model, images, pairs) -> torch.Tensor:
    """Scores [B, 333] in get_pairs() order. Same maths as the notebooks' predict()."""
    crop_logits, disease_logits = model(images.to(DEVICE))
    crop_log_prob = F.log_softmax(crop_logits.float(), dim=1)
    disease_log_prob = F.log_softmax(disease_logits.float(), dim=1)
    pairs = pairs.to(DEVICE)
    return (crop_log_prob[:, pairs[:, 0]] + disease_log_prob[:, pairs[:, 1]]).cpu()


@torch.no_grad()
def image_features(model, images) -> torch.Tensor:
    """Unit-length features, so a dot product is the cosine similarity."""
    return F.normalize(model.features(images.to(DEVICE)).float(), dim=1).cpu()