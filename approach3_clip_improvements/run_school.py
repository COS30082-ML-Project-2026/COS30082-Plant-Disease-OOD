"""Run the standalone notebook using this Python interpreter, without editing it."""
import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true', help='Full data and training; default is DEBUG.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    partial = root / 'checkpoints/approach3_clip/approach3_clip_6.pt'
    if not partial.is_file():
        raise SystemExit('Run python prepare_models.py first.')
    os.environ['CLIP_PARTIAL6'] = str(partial)
    os.environ['CLIP_BASELINE_RESULTS'] = str(root / 'reference_results/approach3_clip_comparison.json')
    notebook = nbformat.read(root / 'approach3_clip_improvements.ipynb', as_version=4)
    if args.full:
        cell = next(c for c in notebook.cells if c.cell_type == 'code' and 'DEBUG = True' in c.source)
        cell.source = cell.source.replace('DEBUG = True', 'DEBUG = False', 1)
    output = root / 'checkpoints/executions'
    output.mkdir(parents=True, exist_ok=True)
    name = ('full_' if args.full else 'debug_') + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    path = output / (name + '_' + uuid4().hex[:8] + '.ipynb')
    manager = KernelManager(kernel_name='python3')
    manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    client = NotebookClient(notebook, km=manager, timeout=None, resources={'metadata': {'path': str(root)}})
    print('Interpreter:', sys.executable, '\nMode:', 'FULL' if args.full else 'DEBUG', flush=True)
    try:
        client.execute()
    finally:
        with path.open('x', encoding='utf-8') as stream:
            nbformat.write(notebook, stream)
        print('Execution log:', path, flush=True)
    print('Done. Inspect results/clip_improvements/ and back up full-run checkpoints.')


if __name__ == '__main__':
    main()
