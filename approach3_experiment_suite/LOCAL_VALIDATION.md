# Fresh-suite validation - 25 November 2026

- 7 fresh-suite regression tests passed, plus 7 shared-protocol tests: 14 total.
- All eight methods ran development and sealed proxy evaluation with tiny/random
  CPU backbones and actual training-image preprocessing. R3/R4/F3/F4 performed
  real updates and saved/reloaded their own new checkpoints. These mechanics
  fixtures do not establish accuracy or real-backbone GPU memory fit.
- Official CSV reads and shared pair/name getters were blocked during the
  eight-method smoke, verifying explicit training-only inputs.
- Expansion to 333 candidates was tested for R3/R4/F3/F4; all learned parameters
  remained exactly unchanged. Pair order, held-out loss masks, seal tampering,
  forbidden old-checkpoint inputs and topk(1) tie handling were checked.
- All seven notebook code cells executed top-to-bottom in default audit-only
  mode: no pretrained-model calls, training or official test access.
- REAL cached public CLIP B/16 R1 smoke completed on RTX 4050, PyTorch
  2.9.1+cu128/OpenCLIP3.3.0: development, seal checks, model reload, proxy eval.
  No official tests were read. Results:
  `results/approach3_suite_fresh/fresh_smoke_20261125T081248Z_fb2a1855/`
  `evaluations/final_smoke_20261125T081925Z_01a378d9/`
- Diff checks passed. Full official evaluation and formal training were NOT run.
  Real SigLIP2/CSP/Troika/DINO GPU smokes remain required at school. No formal
  eight-method performance or RTX5090 memory claim is made.

Historical E1 suite results stay in their old folders and are not fresh-protocol
results. No existing user weights/results were changed. This migration changes
only approach3_experiment_suite and uses master's shared common unchanged.
