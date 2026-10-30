# Stage 2 clean-signal baseline

Status: baseline_measured. Checkpoint selected on fold 9 only.

Fold 10 records: 2158. Macro AUROC: 0.9257316210501665. ECE: 0.03339367429872114. Brier: 0.0841046707611495.

Per-class AUROC:

- NORM: 0.9456292878339568
- MI: 0.9213591135232926
- STTC: 0.9361302978037864
- CD: 0.9225294534373666
- HYP: 0.9030099526524302

## Honest limitations

- One seed and compact architecture; no digitized-image evaluation yet.
- Unlabeled diagnostic records excluded: {'train': 334, 'validation': 37, 'test': 40}.
- Per-record per-lead standardization removes absolute voltage scale.
- Labels follow the official SCP mapping, including zero-likelihood statements.
- Calibration metrics are for uncalibrated probabilities; no confidence intervals yet.
- This clean baseline alone does not test the digitization-degradation hypothesis.
- A low test score requires documented investigation, not tuning against fold 10.
