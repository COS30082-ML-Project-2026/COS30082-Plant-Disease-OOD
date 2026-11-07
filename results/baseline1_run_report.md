# Baseline 1: Independent CNNs — Colab run

Completed on 31 October 2026 using a Google Colab Tesla T4 GPU. Both ResNet-50 classifiers were trained for 20 epochs with debug mode disabled. The checkpoint selected by validation accuracy was epoch 19 (82.37%); epoch 20 validation accuracy was 81.98%.

## Test results

| Metric | Percent |
| --- | ---: |
| Seen top-1 | 62.10 |
| Seen top-5 | 84.15 |
| Unseen top-1 | 8.49 |
| Unseen top-5 | 38.26 |
| Harmonic mean of seen/unseen top-1 | 14.94 |
| Crop accuracy on seen test | 85.75 |
| Disease accuracy on seen test | 64.95 |

The independent classifiers perform substantially better on seen crop–disease pairs than on unseen combinations. The harmonic mean reflects this generalization gap. Test results were computed using the best validation checkpoint, with scores over all 333 candidate pairs.

## Robustness on seen test

| Perturbation | Top-1 (%) | Drop from clean (percentage points) |
| --- | ---: | ---: |
| Brightness | 61.25 | 0.85 |
| Blur | 38.50 | 23.60 |
| Noise | 36.35 | 25.75 |

Perturbations and evaluation follow `common/evaluate.py` unchanged. Crop and disease accuracy are derived from the highest-scoring pair, as defined by the shared evaluator.

## Training and data

- Source commit: `29f704167aefd6771857383be4a3e966ff60b2f4`.
- Dataset: the existing `MyDrive/DPD_Small.7z`, mounted from Google Drive and extracted to Colab's local disk.
- Split sizes: 7,223 train, 777 validation, 2,000 seen test, 1,448 unseen test.
- Seed: 42; shared per-pair 10% validation split; ImageNet normalization and the shared image augmentations.
- Backbone: two independently trained ImageNet-pretrained ResNet-50 models.
- AdamW: crop learning rate `1e-4`, disease learning rate `3e-4`, weight decay `1e-4`, label smoothing `0.1`, batch size 64, cosine learning-rate schedules over 20 epochs.
- Pair score: `log P(crop) + log P(disease)`.
- Two data-loader workers; epoch checkpoints and training history saved to Drive.

## Artifacts and reproduction

- [Colab notebook](https://colab.research.google.com/drive/1UqLFdwarx1IJ-dsgIN3Gqb24Wl5HAu59).
- Repository notebook: [`../baselines/baseline1_independent_colab.ipynb`](../baselines/baseline1_independent_colab.ipynb). It mounts Drive, extracts the existing archive, and runs the full baseline. Run all cells with a GPU runtime and the archive at the path above.
- Metrics: [`baseline1_independent.json`](baseline1_independent.json).
- Confusion counts: [`baseline1_independent_confusion.csv`](baseline1_independent_confusion.csv), 323 true seen-pair rows by 333 predicted-pair columns; counts sum to 2,000.
- Confusion figure: [`baseline1_independent_confusion.png`](baseline1_independent_confusion.png).
- Training history: [`baseline1_training_history.json`](baseline1_training_history.json), one entry per epoch. Loss fields are the last training batch's losses, not epoch averages.
- [Drive output folder](https://drive.google.com/drive/folders/1J1U0gusB8sjJFh2UsjEKpYqBh_xspR5K).
- [Best model weights](https://drive.google.com/file/d/1_n_T_z5xGNrwbEeiws9Usd6diJ0S8PCp/view): approximately 191 MB, with `model_name`, `state_dict` (`crop` and `disease`), and `hp` fields. The best checkpoint is also written after every completed epoch.
- [Latest training checkpoint](https://drive.google.com/file/d/1yn4jl3TLfJfeToWz0pbk5-OGWFQAnBT8/view) includes optimizer and scheduler state for continuing an interrupted run.

Drive files retain their existing access permissions. Dataset images and large model checkpoints are stored in Drive; the repository's existing ignore rules exclude them.

## Verification

Confirmed all 20 history entries, the confusion matrix dimensions and total, agreement between its diagonal and seen top-1 accuracy, the reported harmonic mean, and Python syntax for every code cell in the Colab notebook.
