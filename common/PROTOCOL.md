# Minimal fix to Norman's original protocol

Development uses only the 323 combinations and concept names derived from
`DPD_Small_train.csv`. `get_pairs()` returns 323 rows. `get_loaders()` retains
the four-value return signature, with `None, None` for the two test loaders.
The original per-pair pandas split, seed 42, 7223/777 membership, transforms,
model losses and Top-1/Top-5/harmonic-mean formulas are unchanged.

After checkpoint, hyperparameters and prompt/prediction rules are frozen:

```python
pairs = get_pairs(final_evaluation=True)
seen_loader, unseen_loader = get_final_test_loaders(
    batch_size, norm="imagenet", debug=False, final_evaluation=True)
results = evaluate(predict, seen_loader, unseen_loader, model_name, hp)
```

`predict` must now use the expanded 333 candidates. CLIP must regenerate its
text queries using the already fixed prompt template; no training or selection
follows this expansion. The original 323 rows keep their order; the same 10
official unseen rows are appended in the original order. Default getters stay
at 323 even after explicit final getters have been called.

Both official CSVs require explicit final access. Loader roles and score widths
prevent accidentally evaluating development with 333 candidates or passing
development loaders to official `evaluate`. This is an API boundary, not proof
that a human has frozen all decisions: do not enable final access early or tune
after viewing test results. Historical test exposure cannot be undone.

The two baseline and three original approach notebooks are updated. Other
experimental packages still assuming automatic test loaders or 333 development
candidates require a separate migration; they are not certified by this patch.
Old checkpoints/results are preserved and are not new-protocol formal results.
No 9625-way Cartesian protocol is introduced by this change.

Run checks from the repository root:

```text
python -m unittest discover -s common/tests -v
```
