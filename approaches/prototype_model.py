"""DINOv2 compositional prototypes; no test images enter the support memory."""
import torch
from torch import nn
from torch.nn import functional as F


def balanced_prototypes(embeddings, labels, partners, count):
    """Give each observed partner equal weight, rather than each photograph."""
    prototypes = embeddings.new_zeros(count, embeddings.shape[1])
    present = torch.zeros(count, dtype=torch.bool, device=embeddings.device)
    for label in labels.unique():
        selected = labels == label
        means = [embeddings[selected & (partners == p)].mean(0)
                 for p in partners[selected].unique()]
        prototypes[label] = torch.stack(means).mean(0)
        present[label] = True
    return F.normalize(prototypes, dim=1), present


class CompositionalPrototypeRetrieval(nn.Module):
    """Frozen ViT-B/14, learned concept projections, balanced visual prototypes."""

    def __init__(self, hp, pairs, num_crops, num_diseases, pretrained=True, backbone=None):
        super().__init__()
        if backbone is None:
            import timm
            backbone = timm.create_model(hp['backbone'], pretrained=pretrained,
                                         num_classes=0, img_size=hp['image_size'])
        self.backbone = backbone.requires_grad_(False).eval()
        self.hp = dict(hp)
        width, dim = backbone.num_features, hp['embedding_dim']
        self.crop_head = nn.Linear(width, dim)
        self.disease_head = nn.Linear(width, dim)
        self.crop_classifier = nn.Linear(dim, num_crops, bias=False)
        self.disease_classifier = nn.Linear(dim, num_diseases, bias=False)
        self.register_buffer('pairs', pairs.long().clone())
        for name, count in [('crop', num_crops), ('disease', num_diseases)]:
            self.register_buffer(name + '_prototypes', torch.zeros(count, dim))
            self.register_buffer(name + '_present', torch.zeros(count, dtype=torch.bool))

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def encode(self, images):
        return F.normalize(self.backbone(images).float(), dim=1)

    def project(self, features):
        return (F.normalize(self.crop_head(features), dim=1),
                F.normalize(self.disease_head(features), dim=1))

    @torch.no_grad()
    def fit_prototypes(self, features, crops, diseases):
        crop_emb, disease_emb = self.project(features)
        crop_proto, crop_present = balanced_prototypes(
            crop_emb, crops, diseases, self.crop_prototypes.shape[0])
        disease_proto, disease_present = balanced_prototypes(
            disease_emb, diseases, crops, self.disease_prototypes.shape[0])
        self.crop_prototypes.copy_(crop_proto)
        self.disease_prototypes.copy_(disease_proto)
        self.crop_present.copy_(crop_present)
        self.disease_present.copy_(disease_present)

    def concept_scores(self, embeddings):
        crop_emb, disease_emb = embeddings
        crop = (crop_emb @ self.crop_prototypes.T) / self.hp['temperature']
        disease = (disease_emb @ self.disease_prototypes.T) / self.hp['temperature']
        return (crop.masked_fill(~self.crop_present, -1e4),
                disease.masked_fill(~self.disease_present, -1e4))

    def score_features(self, features):
        crop, disease = self.concept_scores(self.project(features))
        weight = self.hp['crop_weight']
        scores = weight * crop[:, self.pairs[:, 0]] + (1 - weight) * disease[:, self.pairs[:, 1]]
        scores[:, :self.hp['num_seen_pairs']] -= self.hp['seen_bias']
        return scores

    def forward(self, images):
        return self.score_features(self.encode(images))

    def training_loss(self, features, crops, diseases):
        embeddings = self.project(features)
        crop, disease = self.concept_scores(embeddings)
        auxiliary = (F.cross_entropy(self.crop_classifier(embeddings[0]), crops)
                     + F.cross_entropy(self.disease_classifier(embeddings[1]), diseases))
        prototype = F.cross_entropy(crop, crops) + F.cross_entropy(disease, diseases)
        return auxiliary + self.hp['prototype_loss_weight'] * prototype


@torch.no_grad()
def retrieve(model, images, memory, k=5):
    """Return separate crop/disease exemplars plus joint matches from training only."""
    device = next(model.parameters()).device
    features = model.encode(images.to(device))
    crop, disease = model.project(features)
    crop_sim = crop @ memory['crop_embeddings'].to(device).T
    disease_sim = disease @ memory['disease_embeddings'].to(device).T
    scores = model.score_features(features)
    pairs = model.pairs[scores.argmax(1)]
    outputs = []
    for index, pair in enumerate(pairs):
        crop_mask = memory['crops'].to(device) == pair[0]
        disease_mask = memory['diseases'].to(device) == pair[1]
        outputs.append({'pair': pair.cpu().tolist(),
                        'crop': _matches(crop_sim[index], crop_mask, memory, k),
                        'disease': _matches(disease_sim[index], disease_mask, memory, k),
                        'joint': _matches((crop_sim[index] + disease_sim[index]) / 2,
                                          torch.ones_like(crop_mask), memory, k)})
    return outputs


def _matches(scores, mask, memory, k):
    valid = mask.nonzero().flatten()
    chosen = valid[scores[valid].topk(min(k, len(valid))).indices]
    return [{'image': memory['paths'][i], 'similarity': float(scores[i]),
             'crop': int(memory['crops'][i]), 'disease': int(memory['diseases'][i])}
            for i in chosen.cpu().tolist()]
