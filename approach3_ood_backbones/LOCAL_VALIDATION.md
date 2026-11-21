# Local validation — 21 November 2026

Scope: Phase B, F1 and F2 implementation only. No F3/F4, neural training,
new model downloads or full benchmark runs.

- Actual shared metadata inspected: train 7,223, original validation 777,
  seen test 2,000, unseen test 1,448; 333 candidate rows unchanged.
- Training-only compositional split: 32 held-out pairs / 1,104 images;
  reduced train 6,119 images. Every crop and disease remains represented.
- Unit tests exercise disjointness, mappings, empty subsets, nonperfect Top-1/5,
  crop/disease accuracy, score alignment, calibration, coefficient selection,
  adapter dispatch, offline download defaults, SigLIP2 tokenization, scale/bias,
  full-run CUDA guard and F2 selection freeze before test inference.
- Synthetic CPU pipeline tests cover all four F1 model slots and F2 report
  generation. They validate mechanics, not pretrained accuracy.
- Cached pretrained OpenAI CLIP B/16 passed actual CPU inference and optional
  brightness/blur/noise smoke. Latest verified run:
  `results/ood_backbones/F1_smoke_20261121T114645Z_26c56268/`.
  Two validation preflight images and two images per official test split only.
- Smoke membership always uses the full 7,223-image training partition.
  Earlier local smoke folders were development runs and are superseded by the
  run above; do not use their category summaries as evidence.
- Six-step notebook executed in default audit-only mode: six code cells, zero
  execution errors; F1/F2 weight-dependent cells deliberately skipped.
  Executed copy: `results/ood_backbones/notebook_six_steps_2156286a/audit_executed.ipynb`.
- Confusion PNG generation was visually checked. Smoke confusion matrices have
  only two test examples and are not performance evidence.
- Nineteen existing source/split files were hash-checked against
  `protected_hashes.json`; no changes. No writes to previous checkpoints,
  results or original CSVs. New files are local and have not been pushed.

Remaining validation: real CLIP L/14, SigLIP2 and BioCLIP weight loading/inference;
real SigLIP2 F2 inference and full coefficient selection; GPU memory/performance;
full CUDA benchmark reports. Transformers is not installed in the local base
environment. Use the separate school environment in README; no dependency
changes were made to the existing environment.

Robustness currently perturbs RGB before native preprocessing, unlike legacy
post-resize tensor perturbations. Compare these as explicitly different
robustness protocols. Clean inference also intentionally preserves each model's
native geometry/normalization while keeping membership and candidate order fixed.
