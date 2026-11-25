# Fair, training-free E1 experiment sequence

Open **`approach3_training_free/approach3_training_free.ipynb`** after cloning the
whole repository. This package performs **no additional gradient training**.
E1/E3/E4 themselves were previously fine-tuned on7223 images; using those
checkpoints is not a claim of zero-shot learning or of an entirely untrained
system. Crop/disease prototypes use training labels, without neural updates.

## Predeclared sequence

| Step | Variant | Fixed recipe |
|---|---|---|
| T1 | E1 diagnostics + bias correction | Inspect scores without labels; subtract0.5 from standardized seen-pair logits |
| T2 | E1 + DINO crop prototypes | 0.8 standardized E1 pair score +0.2 standardized crop cosine |
| T3 | E1/E3/E4 ensemble | Equal mean of each model's per-image standardized333-class logits |
| T4 | E1 test-time augmentation | Arithmetic mean of raw logits for identity + horizontal flip |
| T5a | E1 symptom-aware prompts | Frozen E1 text tower, six existing sourced descriptions, original prompt fallback |
| T5b | E1 + DINO disease prototypes | 0.8 E1 +0.2 disease cosine after per-image standardization |
| T5c | Symptom E1 + disease prototypes | Same fixed0.8/0.2, a predeclared combined ablation |

No grid search, threshold search, validation fitting or test-driven coefficient
selection occurs. Bias0.5, weights, views and symptom file are frozen in a new
protocol record before test scoring. Do not change them after seeing results.
All variants are reported, not selected/ranked by code as a final "best model".

## Fairness rules

1. **All333 candidate pairs**, in unchanged shared `get_pairs()` / `pair_row`
   order. No unseen-only10-way task, ground-truth-crop filtering or Cartesian
   product of labels. DINO adds a soft component score to every valid candidate.
2. Formal evaluation uses the same2000 seen and1448 unseen test images. The
   shared metric functions (including topk(1)/topk(5) tie handling and rounded
   harmonic mean) are reused through the corrected extended evaluator. Shared
   crop/disease/confusion metrics retain their original argmax behavior.
3. E1/E3/E4 use original weights, stored prompts, B/16 QuickGELU, FP32, batch8,
   Resize256/CenterCrop224 and original CLIP normalization. Load through the
   existing read_checkpoint/extract_state/strict_load functions. Expected release
   hashes and exact model names prevent accidentally loading zero-shot or DEBUG
   weights. Formal clean baseline metrics must reproduce the original seven
   recorded metrics before combining a model. Never loosen the tolerance.
4. T2/T5 additionally require original E1 brightness/blur/noise reproduction,
   preserving the existing D3 gate. They need E1 weights even if scores are cached.
   A failure aborts; it does not trigger weight/preprocessing/tolerance changes.
5. Prototypes use **only the original7223 optimization-training images**; not
   validation777, official tests or all8000 CSV images. Unit-normalize DINO CLS
   embeddings, average per component, normalize the resulting prototype. Every
   crop/disease must have training examples. No missing labels are invented.
6. E1/E3/E4 already saw all7223 images, including the pairs of our later1104-image
   proxy holdout. That holdout is contaminated for these checkpoints. It is NOT
   used here for OOD tuning. Fixed prior recipes avoid a new tuning leak, but
   **do not undo earlier exposure to official test results**. This is exploratory
   research, not a newly blind confirmatory evaluation.
7. Label-free score diagnostics are separated from post-hoc error-overlap
   descriptions. Test labels only score frozen predictions. They never select
   bias, weights, views or descriptions. Candidate-label metadata is public task
   vocabulary, not per-image ground-truth conditioning.
8. Crop/disease-only prototype accuracies are separate JSON diagnostics, with
   pair accuracy N/A. They are not compared as333-way pair accuracy. True unseen
   crop/disease groups remain N/A with this dataset.
9. Preserve inference-cost differences: ensemble requires three CLIP backbones,
   TTA two views instead of one, and DINO requires an extra backbone and training
   feature pass. Same images/candidates/metrics do NOT mean identical compute.
10. Preserve all old code, checkpoints, CSV splits and results. Each sequence
    creates a unique new directory; input cache hashes are verified read-only.

## School / Colab / laptop setup

Use the existing isolated suite environment, or create it as described in
`approach3_experiment_suite/README.md`. From the full repository root:

```bash
git switch member-d
git pull origin member-d
source .venv-suite/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r approach3_experiment_suite/requirements.txt
python -m pip check
python -m unittest discover -s approach3_training_free/tests -v
```

For Colab, install inside the remote runtime and clone the whole repo there.
Your Windows files are not automatically visible to a remote Colab kernel. Use
the runtime's own Python rather than a local terminal. These FP32 frozen methods
do not require BF16. DINO batch4 is the explicit default suitable for initial
T4/6GB-laptop smoke checks; fit still needs to be verified on your actual GPU.
Full neural inference requires CUDA. T1/T3 from complete validated score caches
can run on CPU without loading any weights.

Prepare DPD-Small using the existing root downloader only if data is missing.
Keep the original shared paths. Copy your existing checkpoints into a NEW folder:

```text
checkpoints/training_free_inputs/E1_wise.pt
checkpoints/training_free_inputs/E3_preserve.pt
checkpoints/training_free_inputs/E4_visual_full.pt
```

Or explicitly download the existing release assets (~599MB each):

```bash
gh auth status
python -m approach3_training_free.prepare_inputs --models E1 E3 E4
```

The release is currently a draft: authorized gh access or manual copying is
needed. No credential belongs in the notebook. The helper checks bytes/SHA256
and refuses overwrite. DINO downloads are off unless --allow-download is passed.
No experiment downloads E1/E3/E4 automatically.

## Run smoke first, then the frozen formal recipe

In the notebook select STEPS=["T1"] initially, keep MODE="smoke", RUN=False for
the metadata audit. Set RUN=True only when data/inputs are ready. Smoke uses two
test images per split and one training image per original pair (323 images) for
DINO so that every crop AND disease has a prototype. It is not formal accuracy
or checkpoint reproduction. Formal uses7223 prototype images and all test images.

Equivalent terminal commands:

```bash
python -m approach3_training_free.run --steps T1 --mode smoke
python -m approach3_training_free.run --steps T2 --mode smoke --allow-download
python -m approach3_training_free.run --steps T3 T4 T5 --mode smoke
python -m approach3_training_free.run --steps T1 T2 T3 T4 T5 --mode full
```

Run one step at a time if resources/time are limited. A single complete sequence
reuses its E1 scores and DINO features between steps. New sessions can reuse
prior generated caches via an explicitly copied config:

```json
{
  "score_caches": {"E1":"/path/to/previous/E1_scores.pt", "E3":null, "E4":null},
  "dino_cache":"/path/to/previous/dino_features.pt"
}
```

These are example edits to the full config.json, not a complete standalone
config. Copy config.json, edit those fields, then pass `--config path/to/copy.json`.
Keep the frozen recipe values unchanged. Cache reuse checks exact image names
AND row order, split membership hashes, all333 pairs, model SHA and smoke/full
scope. Legacy R1/R5 zero-shot caches are NOT E1 caches and will be rejected.
Do not guess labels/order or pad10-column unseen scores into a333-column array.
T1 is fastest once an E1 cache exists; creating that cache requires E1 inference
once. Cache-only full runs still verify the recorded clean baseline metrics.

If you already ran the earlier **full** `approach3_dino_fusion` experiment, its
per-split score files contain the original E1 logits. Import them without any
new GPU inference:

```bash
python -m approach3_training_free.import_e1_cache --run-dir /path/to/previous/full/dino/run --output /path/to/new/E1_scores.pt
```

The importer requires that run's formal reproduction gate, exact checkpoint SHA,
preprocessing, split hashes, stored image order/labels and all333 candidate rows.
It rechecks clean E1 metrics from the cached logits and writes a NEW cache file;
it never rewrites the old run. A smoke run, incomplete run, anonymous score
matrix or10-class-only output is rejected. Set score_caches.E1 to the new file.

## Outputs

`results/training_free/sequence_<smoke|full>_<UTC>_<id>/` contains:

- protocol_frozen.json + symptoms_frozen.json, model/checkpoint/version metadata.
- E1/E3/E4 score caches with explicit image order/candidate provenance; clean
  reproduction checks for formal runs and optional source-cache hashes.
- DINO features once; crop/disease prototypes, IDs, counts and training filenames.
- comparison.csv/json for all predeclared pair-classification variants.
- Per-variant333-column scores, labels, image names, confusion PNG/CSV and OOD
  breakdown. Read comparison.csv for the main table.
- Label-free score_diagnostics.json, and descriptive error_overlap.json after
  predictions are frozen. Neither file feeds back into the recipes.
- E1_robust_reproduction.json for formal DINO steps. Full variant robustness
  comparisons are not implemented; primary comparisons use clean test images.

Copy the entire new output folder off the school/Colab machine. Put results on
mounted persistent storage with --output if appropriate; do not assume /content
survives Colab disconnection. No pretrained weights or dataset are committed.

Local validation uses mocks/tiny synthetic scores only. Full E1/E3/E4 score
caches and the checkpoint files are not present locally, so **no real new
performance result is claimed**. Formal reproduction and GPU inference remain
pending. Existing original metric records are copied solely as references.
