# Approach 3 CLIP improvements — standalone school bundle

Everything needed to run the five improvement experiments is here: the notebook, helper modules, a frozen snapshot of common/, pinned requirements, data/model download scripts and historical baseline results. Dataset and binary weights are not committed. Required formal Partial-6 weights are downloaded from the versioned GitHub Release with SHA256 verification; the original OpenAI CLIP weights are downloaded by OpenCLIP on first use.

## School setup (Windows PowerShell, Python 3.11 + Git installed)

```powershell
git clone --branch member-d https://github.com/COS30082-ML-Project-2026/COS30082-Plant-Disease-OOD.git
cd COS30082-Plant-Disease-OOD/approach3_clip_improvements
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe prepare_models.py
.\.venv\Scripts\python.exe download_data.py
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

If data is already on the school machine, skip download_data.py and copy the three DPD_Small_*.csv files and DPD_Small_dataset/ into this folder's data/. The downloader replaces existing matching data, so do not rerun it unnecessarily. No dataset is included in Git. CUDA must be available before formal school training. Pinned PyTorch CUDA 12.8 wheels support the school's target GPU; driver compatibility must be verified on the actual machine.

## Run

```powershell
# First: complete small DEBUG on the school GPU.
.\.venv\Scripts\python.exe run_school.py

# After DEBUG succeeds: full training/evaluation.
.\.venv\Scripts\python.exe run_school.py --full
```

The runner uses the exact current Python executable for its notebook kernel. It saves an executed notebook even on failure; it never edits the source notebook. Alternatively open the notebook in VS Code, select .venv's interpreter, keep DEBUG=True first, then restart/run all with DEBUG=False after successful DEBUG.

Full mode retains all selected weights and requires at least 4 GB free before starting (allow substantially more for the source weights, dataset, environment and downloads). DEBUG checks real checkpoint save/reload and then removes its own disposable weight files, retaining selected states in CPU RAM until evaluation. It requires at least 768 MB free plus sufficient host RAM. DEBUG metrics are mechanics checks, not research results.

## Outputs and backup

- Final comparison table: `results/clip_improvements/<run_id>/comparison.csv`.
- Full metrics, selections and provenance: `comparison.json`, with per-model confusion PNG/CSV beneath that run.
- Formal checkpoints: `checkpoints/clip_improvements/<run_id>/`.
- Executed notebook/log: `checkpoints/executions/`.

Copy the full-run results and checkpoints to your own durable storage before logging off. These output directories are ignored by Git. Each run has a fresh ID, so previous results are preserved.

Only the formal Partial-6 checkpoint is mandatory for the five improvements. Full baseline metrics are included as historical records in reference_results/. To re-evaluate its weights, place your formal approach3_clip_full.pt beside the downloaded Partial-6 checkpoint. Its absence is labelled explicitly and does not substitute DEBUG weights. Zero-shot is reconstructed from the original pretrained model.

The reference notebook under approaches/ exists for provenance/hash checks; do not execute or modify it. This bundle's common/ snapshot keeps the old shared train/validation split: 7,223/777, with the official 2,000/1,448 seen/unseen tests. No extra pairs are held out. All new selection uses validation only. See EXPERIMENT_NOTES.md and notebook markdown for search grids, limitations and the five experimental hypotheses.
