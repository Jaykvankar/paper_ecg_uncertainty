# Stage 2 clean-signal baseline

Status: baseline_measured. Checkpoint selected on fold 9 only.

Fold 10 records: 2158. Macro AUROC: 0.9076432889752459. ECE: 0.03979251337008378. Brier: 0.0913006453230234.

Per-class AUROC:

- NORM: 0.9401095773754438
- MI: 0.912244459520579
- STTC: 0.9330853100740201
- CD: 0.9191558945693102
- HYP: 0.8336212033368764

## Honest limitations

- One seed and compact architecture; no digitized-image evaluation yet.
- Unlabeled diagnostic records excluded: {'train': 334, 'validation': 37, 'test': 40}.
- Per-record per-lead standardization removes absolute voltage scale.
- Labels follow the official SCP mapping, including zero-likelihood statements.
- Calibration metrics are for uncalibrated probabilities; no confidence intervals yet.
- This clean baseline alone does not test the digitization-degradation hypothesis.
- A low test score requires documented investigation, not tuning against fold 10.
