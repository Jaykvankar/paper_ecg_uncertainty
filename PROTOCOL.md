# Locked question

Does tracing a synthetic PTB-XL paper page change classwise calibration, and does that change differ by diagnosis?

The classifier is the existing compact residual network. It is not a new model. The page tracer is the ECG-Image-Kit column max-finder. It is not a new digitizer. Diagnosis is an outcome. It is never an input to rendering or tracing.

## Estimands

- Primary outcome: classwise ECE on the reconstructed waveform minus classwise ECE on the clean waveform.
- Primary page: creased.
- Uncertainty score, not a second primary: change in mean binary entropy, in nats.
- Companion: change in classwise AUROC.
- Reconstruction error: PhysioNet Challenge 2024 lead-wise SNR. Alignment may shift by at most 0.5 seconds and 1 mV. A perfect copy has undefined SNR.
- Uncertainty of the estimate: patient-cluster bootstrap, 1000 resamples, seed 42, percentiles 2.5 and 97.5.

## Two arms

Both arms use seed 42 and the same architecture.

- `standardized` reads `configs/stage2.yaml` and writes `results/stage2`. Each lead is z-scored.
- `raw_mv` reads `configs/stage2_raw.yaml` and writes `results/stage2_raw`. Voltages stay in millivolts.

The machine-readable lock is `configs/paired.yaml`.

## Page settings

ECG-Image-Kit commit `27b90f56896c9fc78b05a83ca14844ea2637aa0b`. One column, so each lead is the full 10 seconds. Resolution 200. No calibration pulse, header, lead names, or handwriting.

- Printed: red grid only.
- Creased: wrinkles plus 3 vertical and 3 horizontal creases at 45 degrees. This is the primary page.
- Noisy: the creased page plus rotation of 5 degrees, noise 40, crop 0, and color temperature 6500.

Lead boxes from the renderer locate each lead. The waveform values come from the max-finder on the ink. Stored plotted pixels are not the recovered signal.

## Gate

Before any fold-10 paired score, training record `records100/00000/00001_lr` (official fold 3) must be printed and traced. Every recovered lead must correlate at least 0.80 with the clean lead. Fold 10 is not a tuning split. One score per arm.

## Where it runs

Google Colab and a laptop with an RTX 40-series GPU use the same code and CUDA 12.4. Research configs request `device: cuda` and stop if no NVIDIA GPU is visible. `--smoke` is the CPU check. Checkpoints from the two machines stay in different directories. Do not score fold 10 twice and keep the better number.

## Claim

This measures synthetic PTB-XL paper pages only. It is not a clinical claim and it does not cover photographs of real paper.
