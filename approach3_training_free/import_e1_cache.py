"""Import full E1 scores from our prior DINO-fusion run, with strict provenance."""
import argparse
import json
from pathlib import Path

import torch

from common.dataset import get_pairs,get_transforms
from approach3_ood_backbones.data import shared_frames
from approach3_ood_backbones.experiments import score_report,write_json,new_run
from approach3_clip_improvements.approaches.clip_improvement_core import file_hash
from approach3_dino_fusion.core import canonical_hash,reproduction
from approach3_training_free.cache import manifest,validate_cache,save_cache


def convert(source,output):
    source=Path(source);output=Path(output)
    config=json.loads((Path(__file__).parent/'config.json').read_text())
    frames=shared_frames();pairs=get_pairs();hashes={}
    def load_json(name):
        path=source/name;hashes[name]=file_hash(path)
        return json.loads(path.read_text())
    protocol=load_json('protocol_frozen.json');metadata=load_json('e1_metadata.json');gate=load_json('reproduction.json')
    if protocol['stage']!='full' or gate.get('passed') is not True:raise ValueError('Require a completed formal E1 reproduction gate.')
    if metadata['checkpoint_sha256']!=config['models']['E1']['sha256']:raise ValueError('Wrong E1 checkpoint provenance.')
    if metadata['candidate_pairs']!=pairs.tolist():raise ValueError('Source candidate ordering differs.')
    if metadata['preprocessing']!=repr(get_transforms('clip')[1]):raise ValueError('Source E1 preprocessing differs.')
    for split,frame in frames.items():
        if protocol['splits'][split]['sha256']!=canonical_hash(frame):raise ValueError('Source split membership differs.')
    payload=dict(**manifest(frames,pairs,False),kind='clip_scores',checkpoint_sha256=metadata['checkpoint_sha256'],
                 model_metadata=metadata,scores={},imported_from=str(source))
    for split in ('seen','unseen'):
        name=split+'_scores.pt';path=source/name;hashes[name]=file_hash(path)
        cached=torch.load(path,map_location='cpu',weights_only=True)
        if cached['image_names']!=frames[split].image.tolist():raise ValueError('Source image order differs.')
        if not torch.equal(cached['pair_labels'],torch.tensor(frames[split].pair_row.to_numpy())):raise ValueError('Source labels differ.')
        if not torch.equal(cached['pairs'],pairs):raise ValueError('Source cached candidate rows differ.')
        payload['scores'][split]=cached['clip_pair_scores'].float().cpu()
    validate_cache(payload,frames,pairs,False,'clip_scores',config['models']['E1']['sha256'])
    reference=json.loads((Path(__file__).parent/'reference_results/E1.json').read_text())
    clean={k:v for k,v in reference.items() if k!='robustness'}
    check=reproduction(score_report(payload['scores']['seen'],payload['scores']['unseen'],frames,pairs),clean,.005)
    if not check['passed']:raise ValueError('Imported E1 scores do not reproduce original clean metrics.')
    payload['import_provenance']=dict(source_hashes=hashes,clean_reproduction=check)
    for name,value in hashes.items():
        if file_hash(source/name)!=value:raise RuntimeError('Source artifact changed during import.')
    output.parent.mkdir(parents=True,exist_ok=True);save_cache(output,payload)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    output=args.output or new_run('import',Path('results/training_free_cache_import'))/'E1_scores.pt'
    print('Validated imported cache:',convert(args.run_dir,output))


if __name__=='__main__':main()
