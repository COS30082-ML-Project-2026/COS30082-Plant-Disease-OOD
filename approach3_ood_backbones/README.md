# Compositional OOD: Phase B, F1 and F2

Run from the **whole repository root**, using the same Python environment as
Jupyter. This package does not modify common/, E1–E5, original CSVs or checkpoints.
F3/F4 are intentionally not implemented. These are zero-shot experiments: **zero
neural training images**, no fine-tuned weights and no retraining on 7,223 images.

## Protocol

- Keep the shared seed 42 and 333 candidate rows: 323 official seen + 10 unseen.
- Shared image split: 7,223 optimization train, 777 original validation, 2,000
  official seen test, 1,448 official unseen test.
- All official unseen test crops and diseases already occur in optimization train.
  The primary OOD objective is **unseen crop–disease combinations**. Unseen-component
  groups are empty and have JSON null / CSV N/A metrics, never zero accuracy.
- OOD breakdown crosses four crop/disease membership categories with exact-pair
  membership, relative to optimization train. Pair Top-1/5 and crop/disease
  accuracy are distinct. Per-pair accuracy is descriptive; n<5 is explicitly
  marked. No confidence or statistical-significance claims are made.
- The additional validation split uses only the 7,223 optimization-training
  images. Seeded greedy selection holds out 32 of 323 pair classes (target 10%
  of pair classes). All images of each selected pair are removed. Every crop and
  disease remains in reduced train. With current files: **6,119 reduced train +
  1,104 compositional validation**. Original 777 validation and official tests
  are unchanged. Greedy selection is deterministic, not a maximum feasible search.
- F2 selects coefficients exclusively on those 1,104 held-out images over the
  unchanged 333 candidates. No official test images/scores are used before
  writing selection_frozen.json. Reading known candidate-label metadata from
  shared mappings is part of the existing zero-shot protocol, not test tuning.
- The zero-shot backbone never optimizes on either training partition. The split
  is a development partition for coefficient selection; it does not prove that
  a pretrained model never encountered related images or concepts. For future
  F3/F4, separately label models trained on 6,119 vs 7,223 images and any final
  retraining. Never reuse a model trained on all 7,223 as a clean held-pair model.
- Prior E1–E5 official test results were already viewed. These further experiments
  are exploratory comparisons, not a newly blind confirmatory test. Lock grids
  before official evaluation; don't revise them repeatedly from test outcomes.

## Model adapters

| Name | Checkpoint | Inference |
|---|---|---|
| clip_b16 | OpenAI ViT-B-16 | OpenCLIP native tokenizer and eval transform, QuickGELU |
| clip_l14 | OpenAI ViT-L-14 | OpenCLIP native tokenizer and eval transform, QuickGELU |
| siglip2 | google/siglip2-base-patch16-224 | Transformers 4.52.2 AutoProcessor/AutoModel, 64-token lowercase padded text, native image processor |
| bioclip | imageomics/bioclip | Published OpenCLIP architecture/preprocessing, official B/16 SimpleTokenizer convention |

Each adapter normalizes embeddings, applies its logit scale and optional bias.
SigLIP2 uses raw similarity logits (sigmoid is unnecessary for ranking), not a
CLIP softmax probability interpretation. Model identifiers, resolved Hub commit
or OpenAI content revision, preprocessing, parameters and versions are saved.
Hub config revisions default to main but the resolved commit is recorded; pin
the recorded SHA in your config for exact repeats. OpenAI weights are SHA-checked;
a cached OpenCLIP HF conversion is also supported and its file hash recorded.

BioCLIP pretraining is biological taxonomy, not specifically plant disease.
Architecture, pretraining data and compute differ across models: this is not a
controlled same-pretraining backbone ablation. No accuracy improvement is promised.

F1 uses the existing single pair prompt. F2 adds separate crop/disease prompts,
normalizes each unique component-label score vector per image using z-scores,
then gathers columns in the original 333 pair ordering. Only valid pairs can be
predicted. Grid weights are nonnegative and sum to one; ties use config order,
with pair-only first. Reports include raw pair-only, calibrated pair-only and
calibrated factorized scoring. No training-time factorized supervision is used.

## School setup (Ubuntu / WSL2, RTX 5090)

In the repository root, in a fresh environment:

```bash
python3.11 -m venv .venv-ood
source .venv-ood/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r approach3_ood_backbones/requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python -m pip check
```

If data is missing, use the existing repo-root `download_data.py`. Do not rerun
it over prepared data unnecessarily. This package does not download datasets.
Register this environment as a notebook kernel if needed:

```bash
python -m ipykernel install --user --name cos30082-ood --display-name "COS30082 OOD"
```

## Offline verification and Phase B

```bash
python -m unittest discover -s approach3_ood_backbones/tests -v
python -m approach3_ood_backbones.run --stage B
```

Tests use actual metadata, synthetic scores, fake adapters and small image
batches. Synthetic accuracy is **not** model performance evidence.

## F1: smoke before full evaluation

By default, missing weights fail without downloading. Checkpoint storage can be
several GB across four models (SigLIP2 alone roughly 1.5 GB). Review space, network
and school permissions before explicitly adding `--allow-download`. This flag
permits large downloads; no weights are bundled or automatically fetched by tests.

```bash
# Cached weights only; two validation images and two images per test split.
python -m approach3_ood_backbones.run --stage F1 --models clip_b16
python -m approach3_ood_backbones.run --stage F1 --models clip_l14
python -m approach3_ood_backbones.run --stage F1 --models siglip2
python -m approach3_ood_backbones.run --stage F1 --models bioclip

# If weights are missing and you authorize the download, repeat the needed
# command with --allow-download. Example:
python -m approach3_ood_backbones.run --stage F1 --models siglip2 --allow-download

# Run only after each corresponding smoke passes on the school machine.
python -m approach3_ood_backbones.run --stage F1 --models clip_b16 clip_l14 siglip2 bioclip --full --robust
```

Full commands also run an inference preflight before reading all test images.
OOM or incompatible APIs produce an error record and stop, never silently
shrinking batch size/settings. Edit a copied config and start a new run if needed.

## F2: select first, test afterwards

```bash
python -m approach3_ood_backbones.run --stage F2 --models siglip2
python -m approach3_ood_backbones.run --stage F2 --models siglip2 --full --robust
```

Smoke mode uses only two holdout images, so its coefficient selection and metrics
are mechanics checks. Full mode uses the entire 1,104-image compositional holdout.
The original seen-only validation is used for inference preflight, not coefficient
selection. No coefficient statistics are fitted on official tests.

## Outputs

Every command creates a unique directory under `results/ood_backbones/`:
config, split manifest and exported derivative partition CSVs; candidate order;
selection_frozen.json for F2; model JSONs; comparison.csv/json; OOD breakdown CSV;
seen confusion CSV/PNG. The original files are never overwritten. JSON null and
CSV N/A mean an empty category. Inspect `comparison.csv` for the main table.

Robustness is optional and covers the original brightness 0.6, blur kernel9/sigma2,
noise0.08 definitions, with fixed noise seed. It perturbs full RGB before native
preprocessing with PIL conversion. This differs from legacy post-resize tensor
perturbations, so do not claim identical robustness protocols across old/new
reports. Clean metrics retain the same candidate space and sample membership,
while intentionally preserving model-specific preprocessing.

GPU peak allocated memory and elapsed time are saved. There are no epochs or
training checkpoints in B/F1/F2. Model weights stay in their standard local caches.

The notebook follows **Setup → Data → Model → Training → Evaluation → Save**.
The Training step explicitly explains that F1/F2 have no neural training; F2
coefficient selection is performed and frozen inside its evaluation runner.
Notebook: `approach3_ood_backbones/approach3_ood_backbones.ipynb`. Select the new
environment kernel. It defaults to audit-only; explicitly enable weight-dependent
cells after setup and smoke checks. F3/F4 require separate approval.

Official references:
- https://github.com/mlfoundations/open_clip
- https://huggingface.co/docs/transformers/v4.52.2/en/model_doc/siglip2
- https://huggingface.co/google/siglip2-base-patch16-224
- https://huggingface.co/imageomics/bioclip
