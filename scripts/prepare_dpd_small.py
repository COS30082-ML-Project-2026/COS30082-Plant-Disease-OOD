"""Extract the supplied DPD-Small archive into the shared loader's layout."""
import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil


def extract_archive(archive, destination):
    import py7zr
    with py7zr.SevenZipFile(archive) as source:
        for item in source.list():
            path = PurePosixPath(item.filename.replace('\\', '/'))
            if path.is_absolute() or '..' in path.parts or ':' in item.filename:
                raise ValueError(f'Unsafe archive path: {item.filename}')
            if getattr(item, 'is_symlink', False):
                raise ValueError(f'Archive links are not allowed: {item.filename}')
        source.extractall(destination)


def organise(extracted, target):
    target.mkdir(parents=True, exist_ok=True)
    for split in ['train', 'test_seen', 'test_unseen']:
        matches = list(extracted.rglob(f'DPD_Small_{split}.csv'))
        if len(matches) != 1:
            raise ValueError(f'Expected one metadata file for {split}: {matches}')
        shutil.copy2(matches[0], target / f'{split}.csv')
    image_dirs = [path for path in extracted.rglob('DPD_Small_dataset') if path.is_dir()]
    if len(image_dirs) != 1:
        raise ValueError('Could not identify the DPD_Small_dataset image folder')
    shutil.copytree(image_dirs[0], target / 'images', dirs_exist_ok=True)
    return validate(target)


def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.reader(stream))
    if not rows[0][1].strip().isdigit():
        rows = rows[1:]
    if any(len(row) != 7 for row in rows):
        raise ValueError(f'Expected seven columns: {path}')
    return rows


def validate(target):
    frames = {split: read_rows(target / f'{split}.csv')
              for split in ['train', 'test_seen', 'test_unseen']}
    expected = {'train': 8000, 'test_seen': 2000, 'test_unseen': 1448}
    for split, rows in frames.items():
        if len(rows) != expected[split]:
            raise ValueError(f'{split}: expected {expected[split]} images, got {len(rows)}')
        for row in rows:
            image = (target / 'images' / row[0]).resolve()
            if not image.is_relative_to((target / 'images').resolve()) or not image.is_file():
                raise ValueError(f'Missing or unsafe image path: {row[0]}')
    image_sets = {split: {r[0] for r in rows} for split, rows in frames.items()}
    for a, b in [('train', 'test_seen'), ('train', 'test_unseen'), ('test_seen', 'test_unseen')]:
        if image_sets[a] & image_sets[b]:
            raise ValueError(f'Overlapping image filenames: {a}, {b}')
    _validate_pairs(frames)
    return {'images': {k: len(v) for k, v in frames.items()}, 'validation': 'passed'}


def _validate_pairs(frames):
    pairs = {split: {(int(r[1]), int(r[2])) for r in rows} for split, rows in frames.items()}
    if len(pairs['train']) != 323 or len(pairs['test_unseen']) != 10:
        raise ValueError('Unexpected number of seen/unseen pairs')
    if pairs['test_seen'] != pairs['train'] or pairs['train'] & pairs['test_unseen']:
        raise ValueError('Invalid compositional zero-shot split')
    for column, count in [(1, 55), (2, 175)]:
        concepts = {int(row[column]) for row in frames['train']}
        if concepts != set(range(count)):
            raise ValueError('Training set does not cover the expected concepts')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--extract-dir', type=Path, default=Path('data/extracted'))
    args = parser.parse_args()
    args.extract_dir.mkdir(parents=True, exist_ok=True)
    extract_archive(args.archive, args.extract_dir)
    report = organise(args.extract_dir, args.data_dir)
    with args.archive.open('rb') as stream:
        report['archive_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
    (args.data_dir / 'dataset_manifest.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
