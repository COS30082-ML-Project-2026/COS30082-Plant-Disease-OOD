"""python -m approach3_experiment_suite.run_school --help"""
import argparse
import json
from pathlib import Path
from approach3_experiment_suite.suite import Suite


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--methods',nargs='+',default=['D3'])
    parser.add_argument('--mode',choices=['smoke','full'],default='smoke')
    parser.add_argument('--allow-download',action='store_true')
    parser.add_argument('--config',type=Path,default=Path(__file__).parent/'config.json')
    parser.add_argument('--e1',type=str)
    args=parser.parse_args();config=json.loads(args.config.read_text())
    config['selected']=args.methods
    if args.e1:config['e1_path']=args.e1
    suite=Suite(config,args.mode,args.allow_download)
    print(suite.catalog().to_string(index=False));print('Saved:',suite.run_selected())


if __name__=='__main__':main()
