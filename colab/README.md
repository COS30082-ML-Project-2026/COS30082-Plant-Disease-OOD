# Colab training for approach 4

Use a T4 GPU. Keep Colab's CUDA-compatible torch/torchvision installation;
the repository's RTX-5090-oriented requirements are not blindly installed here.
Install the pinned extra tools below. `py7zr` is an archive preparation dependency,
not a new model dependency. Download the user-provided archive to Drive first.

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
import os, subprocess
from pathlib import Path
repo = Path('/content/COS30082-Plant-Disease-OOD')
if not repo.exists():
    subprocess.run(['git', 'clone', '--branch', 'approach4-compositional-prototype-retrieval',
                    'https://github.com/COS30082-ML-Project-2026/COS30082-Plant-Disease-OOD.git', str(repo)], check=True)
os.chdir(repo)
subprocess.run(['pip', 'install', '-q', 'timm==1.0.30', 'py7zr==1.1.4'], check=True)
archive = Path('/content/drive/MyDrive/DPD_Small.7z')
assert archive.is_file(), 'Locate the uploaded archive in Drive and update archive above.'
subprocess.run(['python', 'scripts/prepare_dpd_small.py', str(archive)], check=True)
os.environ['CPR_OUTPUT_DIR'] = '/content/drive/MyDrive/COS30082-CPR'
subprocess.run(['python', '-m', 'colab.run_approach4'], check=True)
subprocess.run(['python', '-m', 'colab.run_approach4', '--full'], check=True)
```

The first run is a DEBUG smoke test. The second trains 30 head epochs and evaluates
the complete test sets, using the validation-selected epoch. Outputs persist under
the Drive folder specified by `CPR_OUTPUT_DIR`. GPU disconnects can interrupt a
run; the checkpoint is saved after training and before evaluation begins.

Archive source: project PDF's SharePoint link. User-provided Drive copy:
`18PqhdsjCGwLbWBBcpcs_1O-8zVsY7Ofk`. Expected archive layout:
`DPD_Small_dataset/` and `DPD_Small_{train,test_seen,test_unseen}.csv`.
The preparation script checks counts, candidate support, split overlap and paths,
and records archive SHA-256. Do not commit data, weights or cached features.

Local verification: `python -m unittest discover -s tests -v`.
