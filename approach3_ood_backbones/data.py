"""Immutable shared membership and training-only compositional holdout."""
import hashlib
import random
from collections import Counter

from common import config as cfg
from common.dataset import _split_train_val, get_pairs, load_split


def signature(frame):
    return hashlib.sha256(frame[['image', 'crop', 'disease', 'pair_row']]
                          .sort_values('image').to_csv(index=False).encode()).hexdigest()


def shared_frames():
    train, validation = _split_train_val(load_split(cfg.TRAIN_CSV))
    frames = dict(train=train, validation=validation,
                  seen=load_split(cfg.TEST_SEEN_CSV), unseen=load_split(cfg.TEST_UNSEEN_CSV))
    pairs = get_pairs()
    if pairs.shape != (333, 2) or len(set(map(tuple, pairs.tolist()))) != 333:
        raise ValueError('Expected the unchanged 333 candidate pairs.')
    if [len(frames[k]) for k in frames] != [7223, 777, 2000, 1448]:
        raise ValueError('Shared split counts changed; audit before running.')
    for i, first in enumerate(frames.values()):
        if not first.image.is_unique:
            raise ValueError('Duplicate image within split.')
        for second in list(frames.values())[i+1:]:
            if set(first.image) & set(second.image):
                raise ValueError('Image leakage between shared splits.')
    return frames


def compositional_split(train, fraction=0.1, seed=42):
    """Greedy seeded pair holdout; preserve EVERY crop/disease in reduced train.

    Uses only supplied optimization-training metadata, never original validation
    or official tests. Target is a fraction of pair classes, not image count.
    Greedy selection need not maximize the feasible number of held-out pairs.
    """
    if not 0 < fraction < 1:
        raise ValueError('Pair fraction must be between zero and one.')
    table = train[['pair_row', 'crop', 'disease']].drop_duplicates()
    if table.pair_row.duplicated().any():
        raise ValueError('A pair row maps to multiple components.')
    records = list(table.sort_values('pair_row').itertuples(index=False, name=None))
    crops, diseases = Counter(r[1] for r in records), Counter(r[2] for r in records)
    random.Random(seed).shuffle(records)
    target = max(1, round(len(records) * fraction))
    held = []
    for pair, crop, disease in records:
        if crops[crop] > 1 and diseases[disease] > 1:
            held.append(pair)
            crops[crop] -= 1
            diseases[disease] -= 1
            if len(held) == target:
                break
    if not held:
        raise ValueError('No pair can be held out with both components retained.')
    mask = train.pair_row.isin(held)
    reduced, validation = train.loc[~mask].copy(), train.loc[mask].copy()
    if set(reduced.pair_row) & set(validation.pair_row):
        raise AssertionError('Holdout pair leakage.')
    if not set(validation.crop) <= set(reduced.crop) or not set(validation.disease) <= set(reduced.disease):
        raise AssertionError('Held-out components missing from reduced train.')
    manifest = dict(seed=seed, pair_fraction=fraction, requested_pairs=target,
                    held_pairs=sorted(held), achieved_pairs=len(held),
                    original_train_images=len(train), reduced_train_images=len(reduced),
                    compositional_validation_images=len(validation),
                    original_train_sha256=signature(train), reduced_train_sha256=signature(reduced),
                    compositional_validation_sha256=signature(validation),
                    all_crop_components_retained=True, all_disease_components_retained=True,
                    selection='seeded greedy pair holdout; not a maximum matching',
                    source='original optimization-training partition only')
    return reduced, validation, manifest
