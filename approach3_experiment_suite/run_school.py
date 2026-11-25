"""python -m approach3_experiment_suite.run_school --help"""
import argparse
import json
from pathlib import Path
from approach3_experiment_suite.suite import Suite, final_evaluate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--methods',nargs='+')
    parser.add_argument('--mode',choices=['smoke','full'],default='smoke')
    parser.add_argument('--allow-download',action='store_true')
    parser.add_argument('--config',type=Path,default=Path(__file__).parent/'config.json')
    parser.add_argument('--stage',choices=['develop','final'],default='develop')
    parser.add_argument('--development',type=Path)
    args=parser.parse_args()
    if args.stage=='final':
        if args.development is None:parser.error('--stage final requires --development PATH')
        if args.methods:parser.error('Final evaluation uses the sealed method list; do not pass --methods')
        print('Saved:',final_evaluate(args.development,args.allow_download));return
    if args.development is not None:parser.error('--development is only for --stage final')
    config=json.loads(args.config.read_text())
    if args.methods:config['selected']=args.methods
    suite=Suite(config,args.mode,args.allow_download)
    print(suite.catalog().to_string(index=False));print('Saved:',suite.run_selected())


if __name__=='__main__':main()
