# CLIP improvements (clone the whole repository)

This folder contains only Member D's notebook, experiment helpers, tests, model preparation and run instructions. It uses the repository-root common/, requirements.txt, download_data.py, dataset and checkpoints. No shared pipeline or original notebook is duplicated here.

Run these commands from the repository root using Python 3.11:

```powershell
git clone --branch member-d https://github.com/COS30082-ML-Project-2026/COS30082-Plant-Disease-OOD.git
cd COS30082-Plant-Disease-OOD
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe approach3_clip_improvements/prepare_models.py
.\.venv\Scripts\python.exe download_data.py
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
.\.venv\Scripts\python.exe approach3_clip_improvements/run_school.py
```

Skip the dataset downloader if the repository-root data/ is already prepared. Verify CUDA=True and complete DEBUG before formal training:

```powershell
.\.venv\Scripts\python.exe approach3_clip_improvements/run_school.py --full
```

Alternatively open this folder's approach3_clip_improvements.ipynb and select the repository .venv kernel.

Required formal Partial-6 weights download from GitHub Release into root checkpoints/approach3_clip/ with SHA256 verification. Optional formal Full weights go beside it; otherwise historical Full metrics remain explicitly labelled. Original OpenAI weights download on first use.

Outputs go to repository-root results/clip_improvements/<run_id>/ and checkpoints/clip_improvements/<run_id>/. The comparison table is comparison.csv; execution logs are in checkpoints/executions/. Back up formal outputs before leaving school. Full mode requires at least 4 GB free. DEBUG checkpoint files are temporary after verified save/reload.

The old shared train/validation split is retained; no extra pairs are held out. All selection uses validation and is sealed before final tests. Experimental design details remain in the notebook. Formal GPU execution must be verified at school.
