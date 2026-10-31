# Stage 2 clean-signal baseline

Status: baseline_measured. Checkpoint selected on fold 9 only.

Fold 10 records: 2158. Macro AUROC: 0.925860635653734. ECE: 0.030868303878720294. Brier: 0.08394726159856472.

Per-class AUROC:

- NORM: 0.9452530229365173
- MI: 0.9218385345997286
- STTC: 0.9345075550167258
- CD: 0.9217348899499243
- HYP: 0.9059691757657745

## Honest limitations

- One seed and compact architecture; no digitized-image evaluation yet.
- Unlabeled diagnostic records excluded: {'train': 334, 'validation': 37, 'test': 40}.
- Per-record per-lead standardization removes absolute voltage scale.
- Labels follow the official SCP mapping, including zero-likelihood statements.
- Calibration metrics are for uncalibrated probabilities; no confidence intervals yet.
- This clean baseline alone does not test the digitization-degradation hypothesis.
- A low test score requires documented investigation, not tuning against fold 10.
