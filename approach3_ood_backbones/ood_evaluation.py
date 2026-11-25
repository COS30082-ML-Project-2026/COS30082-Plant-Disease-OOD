"""Membership comes from training, independently of official test names."""
import torch
from common.evaluate import top_k_accuracy


def top1_predictions(scores):
    """Match common.evaluate.top_k_accuracy(k=1), including PyTorch ties.

    Do not substitute argmax or the first column from topk(5). PyTorch does not
    promise the same tied index across those operations or across devices/versions.
    Like shared collect_scores, formal evaluation uses float32 CPU scores.
    """
    return scores.topk(1, dim=1).indices[:, 0]


def membership(frame, train):
    result = frame.copy()
    result['crop_seen'] = result.crop.isin(set(train.crop))
    result['disease_seen'] = result.disease.isin(set(train.disease))
    seen_pairs = set(zip(train.crop, train.disease))
    result['pair_seen'] = [(c, d) in seen_pairs for c, d in zip(result.crop, result.disease)]
    return result


def metrics(scores, labels, pairs):
    scores, labels = torch.as_tensor(scores).float().cpu(), torch.as_tensor(labels).long().cpu()
    pairs = torch.as_tensor(pairs).long().cpu()
    if scores.shape != (len(labels), len(pairs)) or not torch.isfinite(scores).all():
        raise ValueError('Scores must be finite and aligned to candidate rows.')
    if len(labels) and ((labels < 0).any() or (labels >= len(pairs)).any()):
        raise ValueError('Label outside candidate space.')
    if not len(labels):
        return dict(samples=0, top1=None, top5=None, crop_acc=None, disease_acc=None,
                    per_class=[], macro_pair_accuracy=None)
    top1 = top1_predictions(scores) == labels
    # The original evaluator intentionally uses argmax for concept accuracy and
    # confusion, even though its pair Top-1 metric uses topk(1). Preserve both.
    predicted, true = pairs[scores.argmax(dim=1)], pairs[labels]
    per_class = []
    for label in labels.unique(sorted=True):
        mask = labels == label
        per_class.append(dict(pair_row=int(label), samples=int(mask.sum()),
                              top1=round(100 * top1[mask].float().mean().item(), 2),
                              descriptive_only=int(mask.sum()) < 5))
    return dict(samples=len(labels), top1=top_k_accuracy(scores, labels, 1),
                top5=top_k_accuracy(scores, labels, min(5, len(pairs))),
                crop_acc=round(100 * (predicted[:, 0] == true[:, 0]).float().mean().item(), 2),
                disease_acc=round(100 * (predicted[:, 1] == true[:, 1]).float().mean().item(), 2),
                per_class=per_class,
                macro_pair_accuracy=round(sum(r['top1'] for r in per_class)/len(per_class), 2))


def breakdown(scores, frame, train, pairs):
    marked = membership(frame, train).reset_index(drop=True)
    scores = torch.as_tensor(scores).cpu()
    labels = torch.tensor(marked.pair_row.to_numpy())
    if len(scores) != len(frame):
        raise ValueError('Prediction/frame length mismatch.')
    rows = []
    for crop_seen in (True, False):
        for disease_seen in (True, False):
            component = (marked.crop_seen == crop_seen) & (marked.disease_seen == disease_seen)
            for pair_seen in (True, False):
                mask = torch.tensor((component & (marked.pair_seen == pair_seen)).to_numpy())
                rows.append(dict(crop_seen=crop_seen, disease_seen=disease_seen, pair_seen=pair_seen,
                                 **metrics(scores[mask], labels[mask], pairs)))
    return rows


def counts(frame, train):
    marked = membership(frame, train)
    return [dict(crop_seen=c, disease_seen=d, pair_seen=p,
                 samples=int(((marked.crop_seen == c) & (marked.disease_seen == d) & (marked.pair_seen == p)).sum()))
            for c in (True, False) for d in (True, False) for p in (True, False)]
