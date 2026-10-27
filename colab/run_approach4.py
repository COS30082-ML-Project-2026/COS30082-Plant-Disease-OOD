"""Run the six-section approach notebook in an already prepared repository."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', action='store_true', help='30 epochs and all test images')
    args = parser.parse_args()
    os.environ['CPR_DEBUG'] = '0' if args.full else '1'
    path = Path('approaches/approach4_compositional_prototype_retrieval.ipynb')
    notebook = json.loads(path.read_text())
    namespace = {'__name__': '__main__'}
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            exec(compile(''.join(cell['source']), str(path), 'exec'), namespace)


if __name__ == '__main__':
    main()
