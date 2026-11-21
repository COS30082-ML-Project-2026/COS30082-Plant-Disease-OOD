"""Download the existing E1 release asset without overwriting any checkpoint."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

from approach3_clip_improvements.approaches.clip_improvement_core import file_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,default=Path('checkpoints/dino_fusion_inputs/E1_wise.pt'))
    args=parser.parse_args()
    config=json.loads((Path(__file__).parent/'config.json').read_text(encoding='utf-8'))
    target=args.destination
    if target.exists():
        if file_hash(target)!=config['e1_sha256']:
            raise ValueError('Existing file differs; refusing overwrite.')
        print('Verified existing E1:',target);return
    gh=shutil.which('gh')
    if not gh:
        raise RuntimeError('Install/authenticate GitHub CLI, or copy your existing E1_wise.pt manually.')
    target.parent.mkdir(parents=True,exist_ok=True)
    partial=target.with_suffix('.pt.downloading')
    print('Downloading existing E1 checkpoint (~599 MB). Draft release requires authorized gh login.')
    with partial.open('xb') as stream:
        subprocess.run([gh,'api','repos/COS30082-ML-Project-2026/COS30082-Plant-Disease-OOD/releases/assets/'+str(config['e1_asset_id']),
                        '-H','Accept: application/octet-stream'],stdout=stream,check=True)
    if partial.stat().st_size!=config['e1_bytes'] or file_hash(partial)!=config['e1_sha256']:
        raise ValueError('Downloaded E1 checksum/size mismatch; partial file retained for inspection.')
    # Hard-link creation is atomic and refuses an existing destination.
    import os
    os.link(partial,target)
    partial.unlink()
    print('Verified:',target)


if __name__=='__main__':main()
