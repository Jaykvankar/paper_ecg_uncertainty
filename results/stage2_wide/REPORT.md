# Stage 2 clean-signal baseline

Status: baseline_measured. Checkpoint selected on fold 9 only.

Fold 10 records: 2158. Macro AUROC: 0.9112820130782093. ECE: 0.034407707723469586. Brier: 0.08997331935245904.

Per-class AUROC:

- NORM: 0.9419179082104824
- MI: 0.919246947082768
- STTC: 0.9308329337055636
- CD: 0.9240130429719343
- HYP: 0.8403992334202982

## Honest limitations

- One seed and compact architecture; no digitized-image evaluation yet.
- Unlabeled diagnostic records excluded: {'train': 334, 'validation': 37, 'test': 40}.
- Per-record per-lead standardization removes absolute voltage scale.
- Labels follow the official SCP mapping, including zero-likelihood statements.
- Calibration metrics are for uncalibrated probabilities; no confidence intervals yet.
- This clean baseline alone does not test the digitization-degradation hypothesis.
- A low test score requires documented investigation, not tuning against fold 10.
