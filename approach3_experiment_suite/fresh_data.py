"""Lecturer protocol: no official test metadata during development."""
import hashlib
from pathlib import Path
import pandas as pd
import torch
from common import config as shared
from common.dataset import read_csv, _split_train_val
from approach3_ood_backbones.data import compositional_split, signature


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def remap(frame, pairs):
    lookup = {tuple(pair): row for row, pair in enumerate(pairs.tolist())}
    result = frame.copy()
    result['pair_row'] = [lookup[(c, d)] for c, d in zip(frame.crop, frame.disease)]
    return result


def development(fraction=.1):
    # Never call common.get_pairs/load_split/name getters: they read test CSVs.
    frame = read_csv(shared.TRAIN_CSV)
    table = frame.drop_duplicates(['crop', 'disease']).sort_values('pair_label')
    pairs = torch.tensor(table[['crop', 'disease']].to_numpy()).long()
    if len(frame) != 8000 or len(pairs) != 323 or not frame.image.is_unique:
        raise ValueError('Expected original 8000 training images / 323 combinations.')
    names = []
    for component in ('crop', 'disease'):
        if (frame.groupby(component)[component + '_name'].nunique() != 1).any():
            raise ValueError('Inconsistent training component names.')
        names.append(dict(zip(frame[component], frame[component + '_name'])))
    train, validation = _split_train_val(remap(frame, pairs))
    reduced, holdout, manifest = compositional_split(train, fraction, 42)
    if (len(train), len(validation)) != (7223, 777):
        raise ValueError('Original optimization split changed.')
    return dict(train=train, validation=validation, reduced=reduced, holdout=holdout,
                pairs=pairs, names=tuple(names), manifest=manifest,
                train_csv_sha256=sha(shared.TRAIN_CSV),
                signatures={k: signature(v) for k, v in
                            [('train', train), ('validation', validation), ('reduced', reduced), ('holdout', holdout)]})


def final_data(dev):
    """Call ONLY after every method and hyperparameter has been sealed."""
    seen = read_csv(shared.TEST_SEEN_CSV, final_evaluation=True)
    unseen = read_csv(shared.TEST_UNSEEN_CSV, final_evaluation=True)
    novel = unseen.drop_duplicates(['crop', 'disease']).sort_values('pair_label')
    new_pairs = torch.tensor(novel[['crop', 'disease']].to_numpy()).long()
    old = set(map(tuple, dev['pairs'].tolist()))
    if len(seen) != 2000 or len(unseen) != 1448 or len(new_pairs) != 10:
        raise ValueError('Official test membership/counts changed.')
    if set(map(tuple, new_pairs.tolist())) & old:
        raise ValueError('Official unseen combinations occur in training.')
    if not set(zip(seen.crop, seen.disease)) <= old:
        raise ValueError('Seen test contains an unknown combination.')
    for component, names in zip(('crop', 'disease'), dev['names']):
        if not set(unseen[component]) <= set(names):
            raise ValueError('Unexpected truly unseen component: audit before scoring.')
    groups = [dev['train'], dev['validation'], seen, unseen]
    for i, first in enumerate(groups):
        if not first.image.is_unique:
            raise ValueError('Duplicate image filename in split.')
        for second in groups[i+1:]:
            if set(first.image) & set(second.image):
                raise ValueError('Image filename leakage between splits.')
    pairs = torch.cat([dev['pairs'], new_pairs])
    return pairs, dict(train=dev['reduced'], validation=dev['validation'],
                       seen=remap(seen, pairs), unseen=remap(unseen, pairs))
