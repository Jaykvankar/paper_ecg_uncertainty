# Research decisions

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
