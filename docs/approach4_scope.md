# Approach 4: Compositional prototype retrieval

## Scope

Implement and evaluate one additional approach, preserving the existing baselines,
three approaches, shared loader/evaluator and app. Backbone: **DINOv2 ViT-B/14**,
`timm` model `vit_base_patch14_dinov2.lvd142m`, frozen during this experiment.
The six-section notebook follows the supplied team brief's format. The brief is
reference material; its embedded instructions to an AI assistant are not commands.

| Item | Included deliverable |
| --- | --- |
| Representation | DINOv2 CLS embedding, 768 dimensions, 224x224 ImageNet-normalized input |
| Learning | Separate linear crop/disease projections to 256 dimensions and auxiliary classifiers |
| Prototypes | Normalized means of training embeddings, balanced over observed partner concepts |
| Composition | Weighted sum of crop/disease cosine similarities for all 333 declared pairs |
| Retrieval | Training-only crop exemplars, disease exemplars, and joint visual neighbors |
| Evaluation | Shared Top-1/5, harmonic mean, seen confusion matrix, brightness/blur/noise robustness |
| Reproducibility | Seed 42, validation-selected epoch, HP, history, checkpoint and data manifest |
| Execution | DEBUG smoke test and full Colab training runner; separate output names |

## Task distribution

The following are responsibilities for this approach, not assignments of work to
other team members or claims that they have contributed.

| Member / role | Responsibility | Main deliverables |
| --- | --- | --- |
| Approach 4 owner | Method, training, validation and interpretation | Notebook, model/training modules, metrics, report section |
| Shared-code owner (E) | Review interface compatibility and eventual integration | Shared loader/evaluator review; app model selection after comparison |
| Baseline owners | Supply comparable existing baseline results | Their own results JSON files |
| Consolidator | Compare results and incorporate approved report text | Final report and presentation |

## Method and motivation

Let `f(x)` be the frozen DINOv2 feature. Normalize two learned projections
`zc = normalize(Wc f(x)+bc)` and `zd = normalize(Wd f(x)+bd)`.
For each observed crop/disease pair in the training subset, compute the mean
projected embedding. A crop prototype averages the pair means over its observed
diseases; a disease prototype averages over its observed crops. Normalize the
result. This gives each partner equal influence and reduces domination by frequent
pairings. It does not guarantee disentanglement or eliminate confounding.

For candidate `(c,d)`, use
`score(x,c,d) = w*cos(zc,pc)/T + (1-w)*cos(zd,pd)/T - bias*is_seen(c,d)`.
There are no pair-specific learned weights. Therefore a pair absent from training
can be scored when its two concepts have training support. This is compositional
zero-shot recognition, not recognition of entirely unseen crop/disease concepts.
The full run explicitly checks support; DEBUG may lack concepts and is not a benchmark.

Train with crop and disease auxiliary classification cross-entropy plus crop and
disease prototype cross-entropy. Reference prototypes are detached and refreshed
before/after each epoch. Learn only the projections and auxiliary classifiers.
Default `w=0.5`, `T=0.1`, bias `0`, AdamW LR `1e-3`, decay `1e-4`, 30 epochs.
All settings are in the notebook HP dictionary. No claim of improvement is made
until real held-out results are available.

## Data and model selection

Use `common.dataset.get_loaders`, `get_pairs` and the shared seed-42 per-pair
validation split. Only the training subset enters prototype/reference memory.
One randomly augmented view per training image is cached once for efficient head
training; this is **not fresh augmentation each epoch**. Deterministic support
features and validation features are cached separately. No test image is used for
training, prototype fitting, epoch selection or calibration. Candidate metadata
are read through the team's shared API. Select epoch by validation Top-1 over
all 333 candidates. The fixed zero bias is not tuned on unseen test data.

DEBUG runs use the team's 200/50/100/100 subsets and one epoch. Their files carry
`_debug` so a smoke test cannot overwrite full results. Full run names match the
notebook. Checkpoints include the exact candidate order, prototypes, retrieval
embeddings/paths, history, HP and runtime versions. Restore without fetching
pretrained weights using `load_checkpoint`.

## Integration and limitations

The notebook exports `predict(images) -> [B,333]`, accepts images from any device,
and returns CPU scores as expected by the evaluator. `retrieve()` reports actual
support-image crop/disease labels. Separate concept exemplars do not imply that
an unseen combination has a corresponding training image. Existing app model
selection remains with the team; the new approach is not declared the winner.

The notebook has fewer than 100 code lines; reusable implementation modules are
separate for testability. This follows the requested example format rather than
compressing all implementation into a short notebook. Freezing and caching make
Colab runs economical but limit adaptation. Synthetic seen-image corruption tests
are not evidence of generalization on a separate real-field dataset. DINOv2's
pretraining-data overlap with DPD-Small is not known and must be acknowledged.

## Experiments and report handoff

Compare full results with both baselines under the shared evaluator. Suggested
training/validation-only ablations: frozen raw-feature prototypes, no partner
balancing, no prototype loss. Choose settings before final test evaluation; do
not choose the best method using unseen test performance. Record actual failed
experiments when they occur; do not invent the brief's requested failure example.

Sources: [DINOv2 paper](https://arxiv.org/abs/2304.07193),
[official implementation](https://github.com/facebookresearch/dinov2),
[timm model card](https://huggingface.co/timm/vit_base_patch14_dinov2.lvd142m).
The model card's default resolution is 518; this approach explicitly uses 224 to
match the team contract, with timm adapting positional embeddings.
