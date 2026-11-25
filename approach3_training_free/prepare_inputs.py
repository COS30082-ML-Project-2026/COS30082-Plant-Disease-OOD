"""Explicit download of existing E1/E3/E4 assets; never trains or overwrites."""
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from approach3_clip_improvements.approaches.clip_improvement_core import file_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',nargs='+',choices=['E1','E3','E4'],default=['E1'])
    parser.add_argument('--destination',type=Path,default=Path('checkpoints/training_free_inputs'))
    args=parser.parse_args();config=json.loads((Path(__file__).parent/'config.json').read_text())
    gh=shutil.which('gh')
    if gh is None:raise RuntimeError('Authenticate GitHub CLI or copy the existing release checkpoints manually.')
    args.destination.mkdir(parents=True,exist_ok=True)
    for key in args.models:
        spec=config['models'][key];target=args.destination/(spec['name']+'.pt')
        if target.exists():
            if file_hash(target)!=spec['sha256']:raise ValueError('Wrong existing file; refusing overwrite: '+str(target))
            print('Verified existing:',target);continue
        partial=target.with_suffix('.pt.downloading')
        print('Downloading existing',key,'asset, approximately599MB. Draft release requires authorized gh login.')
        with partial.open('xb') as stream:
            subprocess.run([gh,'api','repos/COS30082-ML-Project-2026/COS30082-Plant-Disease-OOD/releases/assets/'+str(spec['asset_id']),
                            '-H','Accept: application/octet-stream'],stdout=stream,check=True)
        if partial.stat().st_size!=spec['bytes'] or file_hash(partial)!=spec['sha256']:raise ValueError('Download size/SHA mismatch; partial retained for inspection.')
        os.link(partial,target);partial.unlink();print('Verified:',target)


if __name__=='__main__':main()
