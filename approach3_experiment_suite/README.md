# Fresh Approach 3 experiment suite

Use `approach3_all_recommended.ipynb` from a clone of the **whole repository**.
This version implements the lecturer boundary: development reads ONLY train.csv
and uses its 323 observed combinations. Official test CSVs are opened only in a
separate final stage after every selected method is sealed. Final prediction uses
all 333 candidates in the original order. Shared `common/` is not modified here.

## Models

| ID | Method | New gradient training |
|---|---|---|
| R1 | Pristine CLIP seen/novel bias calibration | No |
| F2 | Pristine SigLIP2 pair vs calibrated factorized scoring | No |
| R5 | Pristine CLIP symptom prompts, six sourced descriptions | No |
| D3 | Pristine CLIP + frozen DINO crop prototypes, fixed 0.8/0.2 | No |
| R4 | CSP adaptation: learn component embeddings | Yes |
| R3 | Troika adaptation: prompts, adapters and three paths | Yes |
| F3 | SigLIP2 partial visual fine-tuning | Yes |
| F4 | SigLIP2 crop/disease/pair supervision + factorized scoring | Yes |

**No old E1/E3/E4, previous user checkpoint, score cache or resume input is
accepted.** Initialization means public pretrained weights, not random training
from scratch. D3 is now a new pristine-CLIP fusion experiment, NOT the old E1
experiment. CSP/Troika remain DPD B/16 adaptations, not exact paper reproductions.

## School computer setup

Use Python **3.11** in a fresh environment. In the repository root (PowerShell):

```powershell
py -3.11 -m venv .venv-suite
.\.venv-suite\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r approach3_experiment_suite/requirements.txt
python -m pip check
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python -m ipykernel install --user --name cos30082-suite --display-name "COS30082 Fresh Suite"
```

If using conda instead: `conda create -n cos30082-suite python=3.11 -y`, then
`conda activate cos30082-suite` and run the pip commands above. On Linux, use
`python3.11 -m venv .venv-suite` and `source .venv-suite/bin/activate`.
Expect CUDA=True on the RTX 5090. The provided defaults use BF16 for formal
training; unsupported devices fail rather than silently changing precision.
In VS Code select the **COS30082 Fresh Suite** notebook kernel.

Keep the original data layout:

```text
data/DPD_Small_train.csv
data/DPD_Small_test_seen.csv
data/DPD_Small_test_unseen.csv
data/DPD_Small_dataset/<image files>
```

Do not overwrite an existing dataset with the downloader. During development,
only the training CSV and its referenced images are used. Model downloads need
internet and local disk space. No old trained weights need copying.

## Run in order

First verify the interfaces, without downloading any weights:

```powershell
python -m unittest discover -s common/tests -v
python -m unittest discover -s approach3_experiment_suite/tests -v
```

These use tiny/random backbones and real training-image preprocessing to test
mechanics. Their accuracy is **not research evidence**.

Run a small real-weight development smoke before expensive training:

```powershell
python -m approach3_experiment_suite.run_school --stage develop --mode smoke --methods R1 --allow-download
```

For all methods, replace `--methods R1` with
`--methods R1 F2 R5 D3 R4 R3 F3 F4`. Smoke uses two training-only proxy validation
images; trainable methods perform one update epoch on two training images.
D3 smoke uses one training image per crop. Smoke never opens official tests.
Run one method at a time initially to verify GPU memory.

For the formal RTX 5090 development run:

```powershell
python -m approach3_experiment_suite.run_school --stage develop --mode full --methods R1 F2 R5 D3 R4 R3 F3 F4 --allow-download
```

This prints a new `results/approach3_suite_fresh/fresh_full_<id>` directory.
All selected methods must finish before `development_sealed.json` is written.
Settings, chosen weights, checkpoint hashes and public model identities are
frozen. Never modify a sealed folder. There is no automatic test run.

Only after ALL model-development decisions are finished:

```powershell
python -m approach3_experiment_suite.run_school --stage final --development "results/approach3_suite_fresh/fresh_full_<id>"
```

Replace the path with the exact directory printed by development. Final verifies
sealed hashes, expands queries to 333 candidates using the fixed rules, loads
only that run's newly trained checkpoints, and performs no gradient updates or
hyperparameter selection. Final on a `fresh_smoke` directory uses training-only
proxy images and 323 candidates: it is NOT official evaluation.

## Fairness and saving

The original 7223 optimization / 777 validation split is preserved. A seeded,
pair-disjoint holdout is made ONLY from the 7223 images. Default 10% pair holdout
produces 6119 reduced-training / 1104 proxy validation images; the saved manifest
is authoritative. Both component vocabularies remain represented. All trainable
methods and formal DINO prototypes use that reduced partition; trainable smoke
uses two images. No final retraining on all 7223 is performed or implied.

R1 uses original-validation images of reduced-training pairs plus the proxy
holdout. F2/F4 scoring weights and trainable checkpoints are chosen on proxy
validation only, with Top-1 topk(1) tie handling. R5 descriptions/template and
D3 0.8/0.2 fusion are fixed before official test access. True unseen crop/disease
categories remain N/A when empty. Official Seen includes pairs held out during
reduced training; OOD reports correctly use the actual reduced membership.

Historical official-test exposure still exists; fresh initialization does not
make these new experiments blind. Report them as exploratory comparisons and
disclose different backbones/inference costs. Robustness for these variants is
not implemented: clean scores alone do not establish field robustness.

Save the **entire fresh run folder**, including best.pt, prepared.json, manifests,
config and development_sealed.json. Final outputs are in its `evaluations/`
subfolder: all_comparisons.csv, per-method reports, prediction scores, labels,
image ordering, final prompts, and seen confusion CSV/PNG. Keep this on a school
persistent drive, then copy it to OneDrive/external storage before logging out.
Failed runs preserve error.json; create a NEW development session to retry. This
version does not resume partial training. Never tune after viewing final scores.
