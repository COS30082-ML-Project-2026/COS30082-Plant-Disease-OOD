"""Colab alternative when a Drive mount is unavailable; exports a downloadable bundle."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import traceback


def set_stage(output, name):
    (output / 'status.json').write_text(json.dumps({'stage': name}))
    print(name, flush=True)


def run(output, archive, drive_id):
    import gdown
    if not archive.exists():
        set_stage(output, 'Downloading user-provided DPD-Small Drive archive')
        gdown.download(id=drive_id, output=str(archive), quiet=False)
    set_stage(output, 'Extracting and validating DPD-Small')
    subprocess.run(['python', 'scripts/prepare_dpd_small.py', str(archive)], check=True)
    shutil.copy2('data/dataset_manifest.json', output / 'dataset_manifest.json')
    os.environ['CPR_OUTPUT_DIR'] = str(output)
    for full in [False, True]:
        name = 'full' if full else 'debug'
        set_stage(output, 'Training and evaluating ' + name)
        command = ['python', '-u', '-m', 'colab.run_approach4'] + (['--full'] if full else [])
        with (output / (name + '_run.log')).open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    set_stage(output, 'Completed DEBUG and full runs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--drive-id', default='18PqhdsjCGwLbWBBcpcs_1O-8zVsY7Ofk')
    parser.add_argument('--archive', type=Path, default=Path('/content/DPD_Small.7z'))
    parser.add_argument('--output', type=Path, default=Path('/content/CPR-output'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        run(args.output, args.archive, args.drive_id)
    except Exception:
        (args.output / 'error.txt').write_text(traceback.format_exc())
        set_stage(args.output, 'Stopped: inspect error.txt and run logs')
        raise
    finally:
        bundle = shutil.make_archive('/content/CPR-results', 'zip', args.output)
        print('Download bundle:', bundle, flush=True)


if __name__ == '__main__':
    main()
