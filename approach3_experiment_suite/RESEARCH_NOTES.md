# Method identity and research provenance

Only the user-selected recommendation list is included. F1, FF-CLIP, D4/D5,
R2 and R8 are not new suite experiments. Older packages are preserved.

## R4 — CSP (DPD/OpenCLIP adaptation)

Nayak, Yu and Bach, *Learning to Compose Soft Prompts for Compositional Zero-Shot
Learning*, ICLR 2023. Paper: https://arxiv.org/abs/2204.03574 ; official source:
https://github.com/BatsResearch/csp . The original repository uses BSD-3-Clause.

This implementation learns one embedding per disease and crop, initialized by
averaging the pretrained subword embeddings of the verified label. It composes
the two learned vocabulary tokens into a fixed natural-language prefix, passes
them through the frozen CLIP text transformer and scores frozen image features.
The base towers and context prefix are frozen; only vocabulary embeddings learn.
Disease is the attribute and crop is the object. All 333 valid pairs, including
unseen compositions, can be composed at inference.

Differences: OpenCLIP3.3/PyTorch2.9 instead of the original CLIP package; B/16
instead of the published recommended L/14; 77-token context, DPD mappings,
shared compositional split and suite AdamW/validation checkpoint selection.
These outputs must be reported as DPD adaptations, not reproduced paper scores.

## R3 — Troika (DPD/OpenCLIP adaptation)

Huang et al., *Troika: Multi-Path Cross-Modal Traction for Compositional Zero-Shot
Learning*, CVPR 2024. Paper: https://arxiv.org/abs/2303.15230 ; official source:
https://github.com/bighuang624/Troika . Architecture inspected at
`code/model/troika.py` on the official master branch.

The suite implements all distinguishing mechanisms rather than calling a
three-score sum “Troika”: pair/disease/crop text paths; shared learned component
vocabulary plus branch-specific learned contexts; two bottleneck residual visual
adapters per frozen transformer block; branch visual disentanglers; patch-token
cross-attention and a learned residual traction scale; pair and primitive losses;
composition logits plus the product of primitive probabilities at inference.

Differences: B/16/77-token OpenCLIP implementation, PyTorch batch-first MHA and
GELU cross-attention MLP rather than a verbatim source port, one default CMT layer,
DPD labels/splits, shared training/evaluation, AdamW and explicit checkpointing.
Do not describe it as the authors' original code or an exact L/14 reproduction.
Open-world feasibility filters are not used: the prediction space is the fixed
333 DPD candidate pairs. It is more than a simple three-branch baseline, but its
performance remains unverified until the school GPU experiments run.

## R1 — Seen/novel bias calibration

The suite investigates subtracting a scalar from seen-pair standardized logits,
selecting it on proxy harmonic mean (seen validation + compositional holdout).
It does not claim a new research algorithm. The default base is pristine CLIP
B/16. E1 has already seen the proxy-held-out pairs and is intentionally NOT used
for selecting this bias. The development mask contains reduced-training pairs;
the frozen scalar is applied to the original323 seen pairs at official evaluation.
This mask transfer is an experimental assumption that must be disclosed.

## R5 — Symptom-aware text

`symptoms.json` contains six explicit disease-ID/label checks and source URLs.
University extension descriptions support generic powdery mildew, downy mildew,
rust and sooty-mould visible signs. This is partial coverage, not a verified
175-disease symptom catalogue. Unlisted diseases retain label-only text. Coverage,
final prompts and sources are saved with each run. No descriptions are derived
from validation/test images and no prompt is selected using test outcomes.

## D3 / F2 / F3 / F4

D3 reuses the prior exact-checkpoint E1 reproduction gate and frozen DINO crop
prototypes, with the fixed0.8/0.2 fusion. F2 reuses frozen SigLIP2 and validation
coefficient selection. F3 optimizes only configurable final SigLIP2 visual blocks,
with frozen text/logit scale and optional frozen-teacher preservation. F4 adds
verified crop/disease supervision and validation-selected factorized inference.
Training-time supervision and inference-time score fusion are distinct.

All methods are exploratory: official test results were already inspected in
earlier work. A new training-only validation split reduces tuning leakage but
does not make the official test newly blind. Do not adjust the suite from test
outcomes or claim that any backbone/method guarantees OOD improvement.
