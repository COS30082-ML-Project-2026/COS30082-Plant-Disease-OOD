"""Per-image component calibration and validation-only coefficient selection."""
import torch


def standardize(scores):
    scores = torch.as_tensor(scores).float()
    return (scores - scores.mean(-1, keepdim=True)) / scores.std(-1, keepdim=True, unbiased=False).clamp_min(1e-6)


def combine(components, pair_columns, weights):
    if len(weights) != 3 or min(weights) < 0 or abs(sum(weights)-1) > 1e-6:
        raise ValueError('Three nonnegative weights must sum to one.')
    pair, crop, disease = components
    columns = torch.as_tensor(pair_columns).long()
    if columns.shape != (pair.shape[1], 2):
        raise ValueError('Pair column mapping mismatch.')
    # Normalize over each UNIQUE label space before gathering candidate columns.
    return (weights[0] * standardize(pair) + weights[1] * standardize(crop)[:, columns[:, 0]]
            + weights[2] * standardize(disease)[:, columns[:, 1]])


def select_weights(components, pair_columns, labels, grid):
    """Accept only a validation score bundle; callers never supply test scores."""
    labels = torch.as_tensor(labels).long()
    if not len(labels):
        raise ValueError('Empty coefficient-selection validation set.')
    history = []
    for weights in grid:
        scores = combine(components, pair_columns, weights)
        if not torch.isfinite(scores).all() or len(scores) != len(labels):
            raise ValueError('Invalid validation scores.')
        history.append(dict(weights=list(weights), correct=int((scores.argmax(1) == labels).sum()), samples=len(labels)))
    if not history:
        raise ValueError('Coefficient grid is empty.')
    # Config order is the deterministic tie breaker, with pair-only first.
    chosen = max(history, key=lambda row: row['correct'])
    return dict(weights=chosen['weights'], history=history,
                calibration='per image z-score over each unique label space; no test-fitted statistics',
                metric='compositional validation pair Top-1 over all 333 candidates')
