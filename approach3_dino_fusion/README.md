# Frozen E1 WiSE-FT + DINOv2 crop-prototype fusion

Uses your **existing E1_wise.pt**, not a new zero-shot CLIP and not a retrained
CLIP. No gradient-based training, optimizer, CLIP interpolation or weight updates.
The notebook follows Setup → Data → Model → Training → Evaluation → Save;
Training means only constructing class prototypes by averaging frozen features.

## Fixed protocol

1. Use the existing `read_checkpoint`, `extract_state`, `strict_load`,
   `encode_groups` and `make_predict` functions. Require the formal E1 asset SHA256
   `d4d8fca674acb5f3be06b9790a8d8f62c368ba2b85426d13232874c92427392f`.
2. Preserve ViT-B/16 + QuickGELU, FP32, original stored single prompts, batch8,
   CLIP normalization, **shared Resize256 → CenterCrop224** and all 333 candidate
   rows. Do not substitute the native OpenCLIP zero-shot resize transform.
3. Verify full E1 results against the copied original reference JSON:
   seen Top1/5=68.10/89.70, unseen Top1/5=15.06/56.35, H=24.67,
   crop/disease=90.75/70.55. Also verify original brightness/blur/noise results.
   Rounded metrics must match within 0.005 percentage points (effectively exact
   at recorded two decimals). Write a pass/fail reproduction.json. **Abort fusion
   on mismatch.** Do not loosen tolerance or modify weights to force a pass.
4. Load official `facebook/dinov2-base` (ViT-B/14), native AutoImageProcessor,
   CLS embeddings, frozen/eval/FP32. Save resolved revision and preprocessing.
5. Use only the original **7,223 optimization-training images**, not all 8,000
   CSV images and not the reduced F2 split. Original validation777/test2000+1448
   remain untouched. Cross-platform split hashes use LF line endings to match
   the original school Linux run. Every crop must have a training prototype;
   missing crops raise an error rather than inventing values.
6. Normalize each image embedding, average within crop, normalize that mean.
   Compute cosine similarities to all 55 crop prototypes. Gather crop columns
   using the actual crop IDs of shared `get_pairs()` rows: no CSV pair-label
   arithmetic, no disease classifier, no Cartesian-product candidates.
7. **Fixed a priori:**

   `fusion = 0.80 * zscore(E1_pair_logits) + 0.20 * zscore(DINO_crop_cosines)[pair_crop_columns]`

   Population z-score is computed separately for each image over 333 CLIP pair
   logits and 55 unique DINO crop cosines BEFORE gathering candidate columns.
   Standard deviations are clamped at 1e-6. No validation/test-fitted statistics
   or coefficient search. Freeze this protocol before reproduction/test scoring.
   The 0.20 contribution is a documented hypothesis, not an optimized weight.
8. Compare E1 pair prediction, DINO **crop-only** Top1/5, and fused pair prediction.
   DINO cannot distinguish diseases of the same crop, so its pair/disease
   accuracy is N/A. Report crop accuracy on seen and unseen-combination tests;
   genuine unseen-crop/disease groups remain N/A under the existing dataset.

## School setup (RTX 5090 / Ubuntu / WSL2)

Run from the whole repository root in the dedicated OOD environment:

```bash
source .venv-ood/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r approach3_ood_backbones/requirements.txt
python -m pip check
python -c "import torch; print(torch.__version__,torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python -m unittest discover -s approach3_dino_fusion/tests -v
```

If this environment does not exist, follow the OOD package README to create it.
No new dependency beyond that separate environment is required. Do not upgrade
your working E1 environment blindly; preserve torch2.9.1+cu128/OpenCLIP3.3.0 for
reproduction. Verification and full runs require CUDA. Local CPU smoke is allowed.

Copy your existing E1 checkpoint locally and use its path. Alternatively, the
helper downloads the existing ~599 MB release asset into a NEW inputs folder:

```bash
gh auth status
python -m approach3_dino_fusion.prepare_inputs
```

The E1 release is currently a draft, so `gh` needs an authorized account. Do not
publish the draft just to download it. The helper checks exact bytes/SHA256 and
refuses overwrite. An interrupted `.downloading` file is retained for inspection.

```bash
# First do tiny inference/prototype mechanics checks with cached DINO weights.
python -m approach3_dino_fusion.run --stage smoke --e1 checkpoints/dino_fusion_inputs/E1_wise.pt

# If DINO weights are missing, explicitly authorize its download (~350 MB):
python -m approach3_dino_fusion.run --stage smoke --e1 checkpoints/dino_fusion_inputs/E1_wise.pt --allow-download

# Full E1 reproduction gate only, on the school GPU:
python -m approach3_dino_fusion.run --stage verify --e1 checkpoints/dino_fusion_inputs/E1_wise.pt

# Complete experiment. Repeats E1 reproduction; fusion proceeds only on a pass.
python -m approach3_dino_fusion.run --stage full --e1 checkpoints/dino_fusion_inputs/E1_wise.pt
```

Smoke uses one training image per crop and two test images per split. It writes
`NOT_VERIFIED` reproduction status and never claims formal E1 reproduction or
full prototype accuracy. Full extraction uses all 7,223 images. No settings are
silently shrunk after OOM: start a new run with a declared DINO batch size change.
Keep E1 batch8 for reproduction. Do not change the fixed fusion coefficient.

## Outputs and limitations

Each run creates a unique `results/dino_e1_fusion/dino_e1_<stage>_<time>_<id>/`:
frozen protocol, model metadata, E1 reproduction checks, normalized prototypes,
training embeddings/image membership, per-sample scores and predictions,
comparison.csv/json, OOD breakdown and seen confusion PNG/CSV. All artifacts
are new files; original checkpoints/results/splits remain unchanged.

Inspect comparison.csv for the table; DINO crop-only columns must not be compared
as pair accuracy. The embedding/prototype `.pt` files are data-derived artifacts,
not trained backbone checkpoints. No CLIP or DINO weights are saved or modified.

Reference is a copy of E1_wise.json from release asset708985581. Original
per-image predictions were not recorded there, so reproduction establishes
agreement with recorded aggregate metrics and robustness, not bitwise per-image
identity. New per-image predictions/scores are saved for future exact checks.
E1 robustness uses the original evaluator. Fusion robustness is not implemented;
the requested fusion comparison is clean inference under unchanged splits.

Local machine has CPU-only PyTorch and lacks the E1 and DINO weights. Full E1
reproduction, DINO feature extraction and fusion results must be executed and
verified on the school GPU; no successful performance result is claimed locally.

Official model/implementation references:
- https://huggingface.co/facebook/dinov2-base
- https://huggingface.co/docs/transformers/v4.52.2/en/model_doc/dinov2
- https://github.com/facebookresearch/dinov2
