"""Fetch the required formal Partial-6 checkpoint; verify SHA256, never overwrite."""
import hashlib
import os
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
NAME = 'approach3_clip_6.pt'
SHA256 = '5c99701e2abfaa1706adf5761efb2cb290537b7f350f5f994a8480fe0c9cf403'
URL = ('https://github.com/COS30082-ML-Project-2026/COS30082-Plant-Disease-OOD/'
       'releases/download/clip-improvements-inputs-v1/' + NAME)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    target = ROOT / 'checkpoints' / 'approach3_clip' / NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if digest(target) != SHA256:
            raise RuntimeError(f'Existing {target} has a different hash; preserve it and inspect manually.')
        print('Verified existing checkpoint:', target)
        return
    temporary = target.with_suffix('.pt.downloading')
    try:
        with urlopen(URL, timeout=120) as response, temporary.open('xb') as stream:
            total, count = int(response.headers.get('Content-Length', 0)), 0
            for block in iter(lambda: response.read(4 * 1024 * 1024), b''):
                stream.write(block)
                count += len(block)
                print(f'\rModel: {count/1e6:.1f}/{total/1e6:.1f} MB', end='', flush=True)
        if digest(temporary) != SHA256:
            raise RuntimeError('Downloaded model hash mismatch; checkpoint was not installed.')
        os.rename(temporary, target)
        print('\nVerified checkpoint:', target)
    except Exception:
        print(f'\nDownload failed. Inspect/remove only the partial download {temporary} before retrying.')
        raise


if __name__ == '__main__':
    main()
