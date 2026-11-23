# One notebook for the recommended Approach 3 experiments

Open **`approach3_experiment_suite/approach3_all_recommended.ipynb`** after cloning
the **entire repository**. It follows Setup → Data → Model → Training → Evaluation
→ Save. Select one experiment or a list; unselected methods never run. Default
`RUN=False` is audit-only. Default selection is D3; it does not train everything.

| ID | Implementation | Gradient training | Inputs |
|---|---|---|---|
| D3 | Original frozen E1 + frozen DINO crop prototypes | No | E1_wise.pt + DINO weights |
| R1 | Seen/novel score bias calibration | No | Pristine CLIP B/16 weights |
| F2 | SigLIP2 pair/crop/disease factorized scoring | No | SigLIP2 weights |
| R5 | Six sourced symptom descriptions; name-only fallback | No | Pristine CLIP B/16 weights |
| R4 | CSP learned disease/crop vocabulary | Yes, soft embeddings only | Pristine CLIP B/16 weights |
| R3 | Troika adaptation with adapters, 3 paths and CMT | Yes, new adapters/prompts/modules | Pristine CLIP B/16 weights |
| F3 | SigLIP2 final visual blocks + optional preservation | Yes | Pristine SigLIP2 weights |
| F4 | SigLIP2 pair/crop/disease loss + factorized inference | Yes | Pristine SigLIP2 weights |

F1/FF-CLIP/D4/D5/R2/R8 are outside this selected list. Existing code is untouched.
See RESEARCH_NOTES.md for citations and exact CSP/Troika adaptations. Do not label
these B/16 DPD adaptations as reproductions of published L/14 benchmark results.

## 1. Get the code on the school computer

Use Ubuntu/WSL2 on the RTX5090 school machine. In a writable local disk folder
(prefer the machine's local SSD, not a network/OneDrive filesystem):

```bash
git clone --branch member-d https://github.com/COS30082-ML-Project-2026/COS30082-Plant-Disease-OOD.git
cd COS30082-Plant-Disease-OOD
```

If already cloned, save your local work before switching/pulling:

```bash
git switch member-d
git pull origin member-d
```

## 2. Create an isolated environment

```bash
python3.11 -m venv .venv-suite
source .venv-suite/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r approach3_experiment_suite/requirements.txt
python -m pip check
python -c "import torch; print(torch.__version__,torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python -m ipykernel install --user --name cos30082-suite --display-name "COS30082 Experiment Suite"
```

Expect CUDA=True and the RTX5090. Root requirements install CUDA12.8 PyTorch2.9.1;
additional requirements pin Transformers4.52.2. Do not replace the original
working E1 environment. If pip check fails, resolve it before running experiments.
No experiment requires a separate clone of upstream CSP/Troika or Python3.7.

With VSCode, open the whole repo folder and select the **COS30082 Experiment
Suite** kernel. With JupyterLab in your existing Docker/WSL environment, install
these requirements into that container and select its matching kernel. Mount
the whole repo and dataset; keep results on a persistent mounted volume. A host
Python installation does not install packages inside a container.

## 3. Prepare data and weights

Dataset must be at the same repo-root paths as the original project:

```text
data/DPD_Small_train.csv
data/DPD_Small_test_seen.csv
data/DPD_Small_test_unseen.csv
data/DPD_Small_dataset/<image files>
```

If missing, run `python download_data.py` once from the root. Do not run it over
already-prepared data unnecessarily. No dataset is bundled in the new folder.

For D3, copy your existing E1_wise.pt and set E1_PATH in the notebook. Or download
the exact existing release asset into a new inputs folder:

```bash
gh auth status
python -m approach3_dino_fusion.prepare_inputs
```

The E1 asset is in a GitHub draft release, so gh needs an authorized account.
Do not paste credentials into the notebook. The helper checks its SHA256 and
refuses overwrite. E1 is reused without retraining, re-interpolation or prompt
changes. Full D3 aborts unless recorded E1 clean/robustness metrics reproduce.

All other backbones are cached in the standard Hugging Face/OpenCLIP cache.
Downloads are **off by default**. If you authorize the required storage/network
use, enable ALLOW_DOWNLOAD in the notebook or add `--allow-download` to a CLI
smoke command. CLIP B/16 ~0.6GB, DINO base ~0.35GB and SigLIP2 ~1.5GB; allow
several GB for caches and tens of GB for new training checkpoints/runs. No need
to download CLIP L/14, BioCLIP or other models excluded from this selected list.

## 4. Run offline tests, then smoke

```bash
python -m unittest discover -s approach3_experiment_suite/tests -v
python -m unittest discover -s approach3_ood_backbones/tests -v
python -m unittest discover -s approach3_dino_fusion/tests -v
```

In the notebook: keep MODE="smoke", RUN=False first and run all cells for the
audit. Check counts/candidate order and config. Then set RUN=True, choose one ID
in SELECTED, restart/re-run to create a fresh session, and run its smoke.

Equivalent commands (each makes a new session directory):

```bash
python -m approach3_experiment_suite.run_school --methods D3 --mode smoke --allow-download
python -m approach3_experiment_suite.run_school --methods R1 R5 --mode smoke --allow-download
python -m approach3_experiment_suite.run_school --methods F2 --mode smoke --allow-download
python -m approach3_experiment_suite.run_school --methods R4 R3 --mode smoke --allow-download
python -m approach3_experiment_suite.run_school --methods F3 F4 --mode smoke --allow-download
```

Omit --allow-download if weights are already cached. No downloads happen during
offline tests. CPU tests use tiny/random/mock models and are mechanics checks,
not real checkpoint/model-accuracy validation. Smoke mode explicitly uses two
images, one epoch, batch2, accumulation1 and float32 for new methods. D3 retains
the original E1 batch8 and its own one-image-per-crop smoke. Every native full
training run also performs a real gradient preflight, then restores initial
weights/optimizer/RNG before its formal epochs.

## 5. Run formal experiments, one at a time

After the corresponding GPU smoke passes, use MODE="full", RUN=True in a NEW
notebook session or the commands below. Defaults: seed42, 10epochs, batch8,
accumulation4, BF16 where supported, visual6 blocks, visual lr1e-5, prompt lr5e-5,
AdamW, weight decay0.05. Read config.json before committing to long runs.

```bash
python -m approach3_experiment_suite.run_school --methods D3 --mode full
python -m approach3_experiment_suite.run_school --methods R1 --mode full
python -m approach3_experiment_suite.run_school --methods F2 --mode full
python -m approach3_experiment_suite.run_school --methods R5 --mode full
python -m approach3_experiment_suite.run_school --methods R4 --mode full
python -m approach3_experiment_suite.run_school --methods R3 --mode full
python -m approach3_experiment_suite.run_school --methods F3 --mode full
python -m approach3_experiment_suite.run_school --methods F4 --mode full
```

To run the selected order sequentially after smoke checks:

```bash
python -m approach3_experiment_suite.run_school --methods D3 R1 F2 R5 R4 R3 F3 F4 --mode full
```

Full mode requires CUDA. Do not leave the full list selected if you only want
D3. Errors stop the session, preserve diagnostics, and do not silently reduce
epochs/batch/precision. Save completed results before correcting settings and
starting a new session. Keep the school PC awake and plugged in. Use a terminal
session that survives browser disconnects (e.g. your school's approved tmux setup)
for long training; stopping a notebook kernel stops its experiment.

## 6. Fairness, validation and interpretation

- Official split remains 7223train/777validation/2000seen/1448unseen images,
  333 fixed candidate rows. Primary OOD is unseen combinations; truly unseen
  crops/diseases have no samples and are N/A.
- New trainable R3/R4/F3/F4 use **6119 training images**, holding out all images
  of32 pair classes (**1104 images**) for compositional validation. Every crop
  and disease stays represented in training. CE negatives contain only the291
  reduced-training pairs. Evaluation still scores all333 pairs.
- OOD subset membership for these trained models is derived from reduced train,
  so the32 held-out pairs also count as unseen combinations when their images
  occur in the official Seen test. Original Seen/Unseen Top1/5 columns retain the
  original test-file definitions for comparison. The OOD breakdown is therefore
  a separate, explicitly model-training-relative view. Smoke uses the intended
  full partition as membership reference rather than its two-image mechanics subset.
- These models start from pristine pretrained weights, never E1. Select their
  best epoch on the compositional holdout, not official test. F4 then selects
  inference weights on that same holdout. Pair/crop/disease loss weights are
  fixed a priori. No final retraining on7223 is performed by this suite.
- New training uses deterministic native backbone preprocessing, unlike the
  original baseline's random training augmentation. Batch8 with accumulation4
  also differs from the original effective batch8. Report these data/augmentation/
  compute differences instead of claiming a strictly identical-budget ablation.
- R1 also uses pristine CLIP. Selecting a bias for E1 on that holdout would leak:
  E1 already optimized on all7223, including its32 pairs. R1 selects the bias
  using proxy harmonic mean, transfers the scalar from the291-pair development
  mask to the323-pair official mask and discloses that transfer assumption.
- D3 deliberately preserves the original E1/full7223 prototype protocol. Its
  data budget differs from the6119-image models. Do not hide that difference
  in the report. Its0.8/0.2 weight is fixed, not selected on tests.
- R5 currently describes **6 of175 disease labels**. Other labels fall back to
  names; saved coverage identifies affected pairs. This is not an authoritative
  full symptom catalogue. Compare symptom vs name-only predictions on the same
  images; do not add descriptions after seeing which test classes fail.
- CSP/Troika are explicitly documented DPD adaptations. New clean reports use
  native backbone preprocessing. D3 preserves original shared CLIP preprocessing.
  New trainable models do not yet run robustness tests; their reports say so.
  D3 still verifies the original E1 robustness metrics.
- BaseCLIP/text parameters stay frozen for CSP/Troika. Troika learns visual
  adapters/prompts/disentanglers/CMT and uses BatchNorm stored statistics for
  singleton last batches; no sample is dropped. SigLIP2 freezes its text tower
  and logit scale; optional frozen teacher remains on GPU. BF16 autocast keeps
  FP32 master parameters. Checkpoints record trainable parameter counts, history,
  exact pair mapping and partition size; selected checkpoints are round-trip loaded.

## Results: where to look and what to take home

Each session creates `results/approach3_suite/suite_<mode>_<UTC>_<id>/`. Nothing
overwrites an older session. Each method has its own subfolder. D3/F2 have an
additional nested folder because the existing runners are reused unchanged.

- `all_comparisons.csv`: combined table, with method ID/source path/mode.
- `crop_acc`/`disease_acc` are inferred from the highest-ranked pair, preserving
  the existing shared metric definition. DINO-only crop classification remains
  a separate explicitly named column; it has no disease/pair accuracy.
- `session_index.json`: exact output path and completion status per method.
- Each method: comparison.csv/json or report.json, selection_frozen.json,
  OOD breakdown, confusion CSV/PNG, config/model/split metadata.
- R3/R4/F3/F4: best.pt selected on validation, training_preflight.json and epoch
  loss/validation/duration history. Full SigLIP2 checkpoints can be several GB
  because frozen backbone/teacher tensors are retained for strict round-trip.
  An immutable checkpoint is saved immediately at each validation improvement;
  best_checkpoint.json identifies the selected epoch. Reserve substantial disk
  space (preferably100GB for the full multi-method suite). An interrupted run
  retains improvement checkpoints, but automatic optimizer-state resume is not
  implemented; start a new session rather than claiming an exact resumed run.
- D3: E1 reproduction gate, crop prototypes/features, per-image scores and predictions.

Copy the **entire new session directory** to your own drive before leaving the
school machine. Do not commit datasets, caches or checkpoint weights to Git.
No CUDA training, full E1 reproduction or formal benchmark result has been
performed by the local CPU verification of this package. Full selected-method
GPU smoke checks and measured results remain necessary at school.
