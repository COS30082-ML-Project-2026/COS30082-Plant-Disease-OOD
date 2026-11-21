"""Run from repository root: python -m approach3_ood_backbones.run --help."""
import argparse
import json
from pathlib import Path

from approach3_ood_backbones.experiments import audit, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['B', 'F1', 'F2'], default='B')
    parser.add_argument('--models', nargs='+', default=['clip_b16', 'clip_l14', 'siglip2', 'bioclip'])
    parser.add_argument('--config', type=Path, default=Path(__file__).parent / 'configs/default.json')
    parser.add_argument('--full', action='store_true', help='Full inference, only after smoke passes on this machine.')
    parser.add_argument('--allow-download', action='store_true', help='Explicitly permit multi-GB model downloads.')
    parser.add_argument('--robust', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if config['seed'] != 42:
        parser.error('Keep shared seed 42.')
    if args.stage == 'B':
        path, record = audit(config, args.output)
        print(json.dumps(record['split'], indent=2))
    else:
        path, _ = run(config, args.stage, args.models, not args.full, args.allow_download, args.robust, args.output)
    print('Saved:', path)


if __name__ == '__main__':
    main()
