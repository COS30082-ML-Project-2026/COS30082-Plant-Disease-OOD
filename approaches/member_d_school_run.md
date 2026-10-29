# Member D: school computer run

Use branch `member-d`, Python 3.11 and the team's pinned requirements. This notebook implements E0 zero-shot, E1/E2/E3 partial last 1/3/6 visual blocks and E4 full fine-tuning. The expanded experiment checks need more than 100 code lines but stay below 150; flag the soft-limit exception to Member E during review.

## Setup (from repository root)

```powershell
git clone --branch member-d https://github.com/COS30082-ML-Project-2026/COS30082-Plant-Disease-OOD.git
cd COS30082-Plant-Disease-OOD
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -c "import torch, open_clip; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python -m jupyter lab
```

On Linux use `python3.11` and `source .venv/bin/activate`. A school-managed environment can replace the venv if its versions match. Requirements request CUDA 12.8 wheels; check the displayed CUDA device before the full run. Pretrained OpenAI CLIP weights download on first model load, so initial internet access and sufficient storage are needed.

Copy the real DPD-Small metadata and images locally:

```text
data/DPD_Small_train.csv
data/DPD_Small_test_seen.csv
data/DPD_Small_test_unseen.csv
data/DPD_Small_dataset/<filenames matching CSV image column>
```

The three CSVs must contain 8,000 / 2,000 / 1,448 images. Shared code defines 323 seen + 10 unseen pairs. Do not change mappings or split code in the notebook to bypass assertions.

## DEBUG before formal experiments

For a laptop check, run Setup, Data and Model, then only the optional local training smoke-test cell. It uses `approaches/clip_smoke.py` to test one image and one optimizer update for Partial-1/3/6 and Full, checking finite gradients and changed weights. Each test starts from initial pretrained weights, which are restored afterward. Results are saved to ignored `checkpoints/approach3_clip_debug/single_batch_smoke.json`. Stop before the large Training cell if you only want this check. A smoke test proves training mechanics, not full-epoch performance or accuracy.

Open `approaches/approach3_clip.ipynb`. Keep DEBUG=True; restart kernel and Run All. The notebook checks real metadata, prompts, scores and freeze selection, trains each configuration for one epoch, selects a partial winner on validation, then evaluates the three final candidates. Verify finite losses/metrics and generated checkpoints/results. This is a complete small run, not a fabricated sample result.

For a headless DEBUG run from the repository root:

```powershell
python -m jupyter nbconvert --execute --to notebook --ExecutePreprocessor.timeout=-1 --output approach3_clip_debug_executed.ipynb --output-dir checkpoints approaches/approach3_clip.ipynb
```

If memory is insufficient, reduce HP batch_size before running all configurations. For full experiments, fix the same batch size and settings for all runs. Do not tune settings on final seen/unseen scores. Windows notebook DataLoader worker behavior must be checked in DEBUG; report any worker failure to Member E rather than changing common files silently.

## Formal run and handoff

After DEBUG passes, set DEBUG=False, restart kernel and Run All. Training uses validation Top-1 for epochs and partial depth; smaller depth wins ties. Final test runs happen after selection.json is saved. Full text encoder gradients are checked during training. Every experiment resets the pretrained state and seed. Do not rerun final tests while choosing hyperparameters.

Retain all files under checkpoints/approach3_clip/ and results/approach3_clip*. The comparison JSON records all partial histories and the final Zero-shot / Best Partial / Full metrics. Shared evaluation writes confusion CSV/PNG and simulated robustness scores. Keep a separate secure copy of checkpoints before logging off; Git does not back them up. Prepare the one-page report section from real results, including one failure and the limits of seen-only validation and simulated field conditions.

Clear notebook outputs before pushing:

```powershell
python -m nbstripout approaches/approach3_clip.ipynb
git status --short
```

Stage only intended source/results; inspect staged paths before commit. The latest team master already contains tracked bytecode caches; ignore rules do not remove those historical tracked files. This Member D change leaves common/, other notebooks and the app untouched.

Current verification: source checks only. The authoring machine has no torch/OpenCLIP or dataset, so no actual model, GPU or real-data smoke test has passed yet. School DEBUG is required before any full training claim.
