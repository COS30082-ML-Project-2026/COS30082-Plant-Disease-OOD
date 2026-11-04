# Member D CLIP improvements

Open `approach3_clip_improvements.ipynb`. Original `approach3_clip.ipynb`, common code and existing artifacts are reference files and remain untouched. Four new helper/test files implement strict loading, data/prompt contracts, controlled training and experiment orchestration.

## Sources and setup

Use the same Python 3.11 / pinned requirements as the original school run. The Anaconda environment used for local DEBUG may have different package versions; its actual versions are recorded and its numbers are not formal report evidence.

The notebook automatically tries these Partial-6 locations, in order:

1. `checkpoints/approach3_clip/approach3_clip_6.pt`
2. Your Downloads folder's `approach3_clip_6.pt`
3. Your OneDrive Model/checkpoints/approach3_clip folder.

For a different machine/location, set `CLIP_PARTIAL6` before starting Jupyter, or set `PARTIAL6_PATH` in Setup. Optional `CLIP_BASELINE_RESULTS` points to the original comparison JSON. To re-evaluate the old Full model, copy its `approach3_clip_full.pt` next to Partial-6. Original Zero-shot is reconstructed from the same pretrained OpenAI model without retraining. A missing/unreadable Full checkpoint is explicitly recorded, not replaced with guessed metrics. Historical Full metrics can be retained separately from new evaluation.

Example from the project root in PowerShell:

```powershell
$env:CLIP_PARTIAL6 = (Resolve-Path checkpoints/approach3_clip/approach3_clip_6.pt).Path
python -m jupyter lab
```

All actual image/CSV paths come from common/config.py. Downloaded OpenAI weights are needed on first setup; no OpenAI API key is required. An unreadable OneDrive placeholder must be downloaded locally first.

## Fair protocol

The original 7,223/777 shared train/validation split is retained, plus 2,000 seen and 1,448 unseen test images. No additional pairs are held out. Split/image hashes, shared mapping and sample counts are checked. All candidates see the same train batches and seed; validation Top-1 is over all 333 candidates. Learning rate, AdamW, weight decay, epochs, batch size, augmentation and float32 match the original formal Partial-6 experiment. Text/logit_scale freeze and extra preservation/ensemble/interpolation are explicit experimental factors. Evaluation workers are set to zero for consistent perturbation RNG, without changing split membership or image transformations.

E3/E4/E5 each start from original pretrained CLIP, not an already fine-tuned model. E3 selects positive lambda on validation. E5 transfers that chosen lambda and E2's chosen ensemble unchanged, trains independently, then selects its own alpha using validation. E1 selects a separate alpha for the existing Partial-6. Both floating weights and stored logit_scale use raw-state interpolation; integer/bool buffers come from original CLIP.

Every selection is finished and written to selection_frozen.json before any final testing. Prior baseline test results already motivated this follow-up, so report it as exploratory work, with one seed, seen-only validation and an increased hyperparameter-search budget. E5 is a hypothesis: a combination may perform worse. Do not tune against new test scores.

## DEBUG and full run

Keep DEBUG=True and restart/run all. It exercises all five stages, with 2 images per shared DEBUG split, batch 1, 1 epoch, three alpha values and one positive lambda. It tests real loading, gradients, freeze state, checkpoint save/reload, final shared evaluation, robustness and confusion output. These tiny results are not valid accuracy comparisons. CUDA memory compatibility must still be checked on the actual RTX 4050/5090; the teacher stays on CPU and text chunks/gradient checkpointing bound device memory. No experiment is silently skipped to make a run fit.

After DEBUG passes on the school GPU, set DEBUG=False, restart and run all. Full mode uses 10 epochs, batch 8 and float32, with the full declared grids. If it OOMs, the error reports batch/precision; fix the environment or declare a new comparison protocol instead of silently changing just one model. Do not repeatedly rerun final test evaluation to choose a configuration.

Outputs are unique per execution. DEBUG verifies each checkpoint through a real disk save/reload, removes that newly created temporary file, and retains selected states in CPU memory until evaluation. Its JSON labels checkpoint paths as ephemeral. This avoids filling the laptop disk with disposable model copies; closing the run releases the states. Full mode retains all checkpoints and checks for at least 4 GB free disk space before starting. DEBUG requires at least 768 MB free and additional host RAM for selected model states.

```text
results/clip_improvements/<run_id>/
    setup.json
    E*_selection.json
    selection_frozen.json
    baseline_zero/ ... baseline_partial6/ ... E1_wise/ ... E5_combined/
    comparison.json
    comparison.csv
checkpoints/clip_improvements/<run_id>/
    E1_wise.pt ... E5_combined.pt
    E5_combined_raw.pt
```

Allocate several GB for the six full-mode FP32 checkpoint files, plus the source weights. Existing output files are never overwritten; a failed run remains inspectable and a new execution gets a fresh ID. Copy new full-run results/checkpoints to your own durable storage after school training; Git is not a checkpoint backup.

Targeted tests: `python -m unittest discover -s tests -p test_clip_improvements.py -v`. School notebook execution: `python -m jupyter nbconvert --execute --to notebook --ExecutePreprocessor.timeout=-1 --output-dir checkpoints approaches/approach3_clip_improvements.ipynb`. Keep the source notebook output-free for Git; retain executed copies under ignored checkpoints.
