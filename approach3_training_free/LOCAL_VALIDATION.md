# Local verification — 25 November 2026

- 13 new tests passed. Coverage: fixed recipes cannot be retuned through config;
  333-candidate/image-order/source-checkpoint/scope cache guards; missing prototype
  labels; label-free diagnostics; original input scores unchanged; independent
  topk(1) direct-component tie handling; exact two-view raw-logit TTA; formal E1
  reproduction failure stops subsequent variants; strict prior-D3 cache import.
- All five stages completed an end-to-end smoke with explicitly MOCK CLIP/DINO
  backbones and real metadata. No performance evidence is inferred from this.
- The full-size cache-only T1 test uses SYNTHETIC scores with actual2000/1448
  shared labels and a mocked reproduction gate. It verifies no weight/model
  loading, shared metric equality and source-cache immutability. It is NOT a
  successful real E1 reproduction or an official model result.
- A prior-D3 importer fixture tests full shape/provenance/row alignment with a
  mocked clean reproduction check, then proves reordered images are rejected.
  No real full E1 prediction cache was available locally for import verification.
- Six-step notebook executed in audit-only mode: six code cells, zero errors.
  Executed copy: `results/training_free/notebook_audit_8a52acad/executed.ipynb`.
- E1/E3/E4 release metadata and SHA256 identities were checked via GitHub.
  The original metric JSONs were copied from the already-downloaded formal
  result archive. No model weights or datasets were downloaded in this task.
- All19 pre-existing protected source/split files match their prior hashes.
  Git diff shows no existing tracked-file changes; only this new package exists.
  Original common.evaluate and E1–E5 code, checkpoints and results are preserved.

Pending: real full E1/E3/E4 loading/reproduction, GPU DINO feature extraction,
all formal cached-score comparisons, ensemble/TTA/prompt/prototype performance.
No new improvement is claimed. Existing R1/R5 zero-shot caches are deliberately
NOT accepted as E1 caches. Use actual E1 provenance or extract once on GPU.

The0.5 bias, equal ensemble weights,0.8/0.2 fusion, identity/horizontal-flip views
and six existing sourced descriptions were frozen before any new test-scoring
run. No new validation/test hyperparameter search is implemented. This remains
exploratory after previous official test results were viewed; it does not create
a newly blind test or remove E1's exposure to the later proxy-held-out pairs.
