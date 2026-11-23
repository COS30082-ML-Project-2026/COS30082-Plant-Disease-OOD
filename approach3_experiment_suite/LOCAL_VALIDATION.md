# Local validation — 23 November 2026

- New suite: 13 tests passed. Existing OOD/F2 package: 17 passed. Existing
  frozen E1/DINO package: 15 passed. **45 tests passed in total.**
- Actual tiny OpenCLIP transformer forward/backward tests cover both CSP and
  Troika paths, soft vocabulary gradients, frozen base towers, cross-attention,
  visual adapters, training targets and checkpoint round-trip. These models are
  randomly initialized tiny mechanics fixtures, not performance evidence.
- SigLIP2 training contract tested with a tiny PyTorch stand-in: final visual
  block selection, frozen text/logit parameters, frozen teacher and preservation.
  Real SigLIP2 model loading/CUDA gradients still need school GPU validation.
- End-to-end native CSP smoke tested with actual metadata/images and a tiny
  random CLIP fixture: checkpointing, selection freeze, OOD reports and tables.
- Real **cached pretrained CLIP B/16** CPU smoke completed for R1 and R5:
  `results/approach3_suite/suite_smoke_20261123T155038Z_cbb3262f/`.
  Two validation/test images per category only; no formal accuracy claims.
  Both methods completed and generated `all_comparisons.csv` and detailed reports.
- Master six-step notebook executed in default audit-only mode: six code cells,
  zero execution errors, no weights downloaded and no model training triggered.
  Executed copy: `results/approach3_suite/notebook_audit_7ecab857/executed.ipynb`.
- 19 pre-existing source/split files verified unchanged against the existing
  protected hash manifest. Git diff shows no tracked existing-file modifications.
  New methods live only in this package; generated runs live in new directories.
- No new dependencies, large model weights, full benchmark or full GPU training
  was installed/downloaded/run on the local CPU machine.

Still required at school: GPU smoke for all selected methods, full E1 clean and
robustness reproduction, real DINO extraction/fusion, full SigLIP2/CSP/Troika
training and official evaluation. CSP/Troika are explicitly documented DPD/B16
adaptations, not claims of reproducing published L14 results. R5 has six sourced
descriptions only, with explicit name-only fallback. R1 intentionally does not
use E1 for contaminated proxy-validation tuning. New trained methods use6119
images and derive their OOD subsets from that partition. Original test-file
Seen/Unseen columns are preserved separately.
