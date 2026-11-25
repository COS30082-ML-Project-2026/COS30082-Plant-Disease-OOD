# Evaluation consistency audit — 25 November 2026

## Original behavior (preserved)

`common.evaluate.top_k_accuracy` independently calls `scores.topk(k, dim=1)`
with k=1 and k=5. It checks target membership and rounds percentages to two
decimals. Its harmonic_mean consumes those rounded Seen/Unseen Top1 values.
Shared collect_scores converts prediction scores to float32 CPU tensors.

Shared crop/disease accuracy and confusion matrices instead use scores.argmax.
Under exact ties, this can choose a different candidate than topk(1). This
internal legacy distinction is deliberately preserved, not "corrected" in common.

## Fix

- Extended pair Top1 predictions now call topk(1) directly; they no longer take
  the first index from topk(5).
- Extended aggregate Top1/Top5 delegate to the shared top_k_accuracy function;
  the aggregate harmonic mean delegates to the shared harmonic_mean function.
- Extended per-pair accuracy uses the corrected Top1 prediction indices.
- Extended concept accuracy uses argmax, matching shared concept_accuracy.
- The 333 candidate rows, pair_row mapping, splits and original evaluator are
  unchanged. No original E1–E5 checkpoint or result was rewritten.

## Verification

51 tests passed: 23 OOD tests (including six new parity tests), 13 suite tests,
15 frozen-fusion tests. New tests include all333 scores equal, two equal maxima,
Top5-boundary ties, quantized logits, concept accuracy and full split sizes.

Read-only cache audit:
`results/evaluation_consistency/parity_20261125T033625Z_381098c8/summary.json`.
Each case calls the actual common.evaluate.evaluate with cached row IDs in place
of images, routes its output into a NEW reference directory, and scores the
same saved-and-reloaded tensor/label payload with the extended evaluator.

| Cache | Seen / Unseen rows | All five requested metrics | Top1 index disagreement |
|---|---:|---|---:|
| Real R1 smoke scores | 2 / 2 | Identical | 0 |
| Real R5 smoke scores | 2 / 2 | Identical | 0 |
| Synthetic tied-score full-size fixture | 2000 / 1448 | Identical | 0 |

Crop/disease accuracy also matches in every case. The five requested metrics are
Seen Top1, Seen Top5, Unseen Top1, Unseen Top5 and harmonic mean. These use all333
candidate columns; none of the audit cases restricts predictions to10 classes.
The full-size synthetic fixture reproduces 473 seen / 340 unseen old-vs-shared
Top1 index mismatches, reduced to zero by the fix. Its accuracies are NOT model
performance results.

The available real caches are smoke-sized, not full formal model predictions.
Their old files did not embed labels; labels were explicitly reconstructed from
the unchanged shared split head(2), matching the recorded smoke runner's order.
New labelled cache snapshots record this provenance. Source score-file hashes
were checked before/after reading. All19 protected original source/split files
also matched their existing hashes.

## Remaining differences / limits before new experiments

1. No differences remain in the five requested final metrics for the checked
   cache payloads. No new model inference, training or test-based tuning ran.
2. Some existing validation/checkpoint/coefficient-selection routines use argmax,
   whereas shared val_accuracy uses topk(1). Exact validation ties may therefore
   select differently. This evaluation-only fix does not rewrite those routines
   or any historical selections. Final-metric parity does not imply identical
   hyperparameter-selection tie behavior.
3. Shared concept/confusion argmax and pair-metric topk(1) may still disagree on
   ties by original design. The extended evaluator now preserves both rules.
4. PyTorch does not promise the same tied index across devices/versions. Keep
   float32 CPU evaluation and record/pin the software environment. This audit
   compared identical tensors within one local environment, not bitwise results
   between local CPU and a different Colab/school runtime.
5. Empty extra OOD categories remain N/A. Per-class/macro metrics are additions
   without a shared counterpart. Model-specific preprocessing/data budgets and
   robustness protocols are separately documented; score parity does not make
   those experimental settings identical.
6. Full real-model cache parity still requires full saved prediction scores and
   labels. Only synthetic scores were available locally at2000/1448 rows. No
   claim is made that a new full E1/model evaluation was performed.

## Repeat from identical cached predictions

```bash
python -m unittest discover -s approach3_ood_backbones/tests -v
python -m approach3_ood_backbones.verify_evaluation --cache results/approach3_suite/suite_smoke_20261123T155038Z_cbb3262f/R1/scores.pt results/approach3_suite/suite_smoke_20261123T155038Z_cbb3262f/R5/scores.pt --reconstruct-smoke-labels --include-synthetic-full-split
```

Those local generated caches are not committed. For a full saved score bundle,
use a trusted local torch.save payload containing `pairs`, `seen_scores`,
`seen_labels`, `unseen_scores`, `unseen_labels`, plus optional `scope` and
`label_provenance`. Rows must follow the unchanged shared CSV ordering. Then:

```bash
python -m approach3_ood_backbones.verify_evaluation --cache /path/to/full_cached_scores.pt
```

Every audit creates a new directory, records equality/deltas and source hashes,
and refuses candidate remapping. No input score files are overwritten.
