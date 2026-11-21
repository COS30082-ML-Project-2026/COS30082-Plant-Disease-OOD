# Local checks — 21 November 2026

- 15 unit tests passed using the local CPU environment.
- Existing E1 checkpoint loader/strict loading, stored prompts, candidate order,
  frozen parameters and 333-column inference interface tested with a tiny mock
  model. This is NOT a real E1 checkpoint reproduction.
- Actual training and validation metadata match the recorded E1 Linux split
  hashes when CSV line endings are canonicalized to LF.
- Prototype normalization/counts, missing crops, noncontiguous crop IDs, all333
  candidate mappings, score calibration, fixed0.8/0.2 fusion and crop-only metric
  semantics tested. End-to-end CPU mechanics smoke used synthetic backbones,
  actual split metadata and fresh temporary output files.
- Failure-gate test confirms that a full E1 reproduction mismatch writes its
  report and aborts before DINO loading/feature extraction.
- Six-step notebook executed in default audit-only mode: six code cells, no
  execution errors. Model runs intentionally skipped. Executed copy:
  `results/dino_e1_fusion/notebook_audit_3ca225b5/audit_executed.ipynb`.
- E1 asset identity/size/SHA256 checked through authenticated GitHub metadata;
  original E1 evaluation JSON copied from the previously downloaded result ZIP.
  No model weights were downloaded and no dependencies were changed locally.
- 19 pre-existing source/split files checked against the protected hash manifest;
  unchanged. No existing checkpoint, result or original split was written.

Pending on the school CUDA GPU: real E1 loading and full clean/robustness
reproduction; real frozen DINO inference; all7,223-image prototype extraction;
full E1 vs DINO crop-only vs fused evaluation. No performance improvement or
successful real E1 reproduction is claimed by these local mechanics checks.
