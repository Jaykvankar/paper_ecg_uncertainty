# Research decisions

## Locked calibration question (2026-10-24)
- The question is the change in classwise ECE when the same frozen classifier reads a traced paper page instead of the clean waveform. Diagnosis is an outcome, not an input. See `PROTOCOL.md` and `configs/paired.yaml`.
- Two arms share seed 42 and the compact residual network: standardized leads, and raw millivolts. Both request CUDA and stop if no NVIDIA GPU is visible. `--smoke` remains the CPU check.
- Google Colab and an RTX 40-series laptop use torch 2.5.1 with CUDA 12.4. Their checkpoints stay in different directories. Fold 10 is scored once per arm.
- The page tracer is ECG-Image-Kit's column max-finder at commit `27b90f56`. A printed page of training record `00001_lr` must correlate with the clean leads before any fold-10 paired score.
- Full training has not been run. No fold-10 result is claimed.

## Colab preparation and Stage 2 implementation (2026-10-12)
- Existing Stage 0/1 source files and saved results remain unchanged.
- Bundle existing data/code/results with per-file hashes and a checked ZIP.
  The Colab notebook preserves the original user GPU check output.
- Save the transfer bundle and completed epoch checkpoints to Google Drive;
  keep runtime dependencies and extracted data on Colab's local disk.
- Exclude 411 records with missing diagnostic superclasses from Stage 2 supervised
  training/evaluation; retain them in the Stage 1 manifest.
- Per-record, per-lead standardization; compact six-block residual classifier;
  unweighted BCE, AdamW, cosine schedule, 30 epochs, seed 42. No augmentations yet.
- Select on fold 9 macro AUROC only. Gate clean-test evaluation at validation
  AUROC 0.88 for debugging. Record low test performance for investigation without
  tuning against fold 10. This threshold is a heuristic, not a result.
- Explicit metric definitions: mean classwise 15-bin positive-probability ECE,
  record/class-average Brier, tie-aware AUROC.
- Full Stage 2 training and all Stage 3 work remain pending remote execution/review.
