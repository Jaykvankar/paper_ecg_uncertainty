# Diagnosis-Aware Uncertainty for Paper-ECG Pipelines

This research project is intended to study uncertainty in ECG analysis pipelines,
with a focus on digitizing ECGs from paper images and understanding how diagnostic
context affects the resulting measurements and predictions.

## Current status

The repository currently implements the data-preparation foundation using the
PTB-XL 1.0.3 dataset. It downloads and validates the 100 Hz digital ECG records,
builds diagnostic multi-labels, and checks that the official train, validation,
and test splits do not share patients.

Stage 2 training and clean-test evaluation are implemented and locally smoke-tested.
Full baseline training has not yet been run. The saved Stage 0/1 results validate
the input data and labels only; they do not test the research hypothesis or support
clinical conclusions. Paper-image generation, digitization, and Stage 3/4 evaluation
are not implemented yet.

## Data pipeline

- Uses PTB-XL 1.0.3, including the 100 Hz recordings (`records100`).
- Validates each waveform against the dataset SHA-256 manifest, expected lead
  order, 100 Hz sampling rate, 1,000 samples, mV units, and finite values.
- Maps diagnostic SCP statements to the five superclasses `NORM`, `MI`, `STTC`,
  `CD`, and `HYP`. Labels are multi-label; records without a diagnostic label
  are retained and flagged rather than treated as normal.
- Uses folds 1–8 for training, fold 9 for validation, and fold 10 for testing.
  Patient separation is checked across all three splits.

The PTB-XL data is not included in the repository. The configured download
retrieves the dataset files needed for this stage; use and distribution remain
subject to the dataset's license and terms.

## Setup and usage

The commands below are for Windows PowerShell. From the repository root:

For a fresh installation, `python` must be standard Windows Python 3.10+, not the
MSYS2 interpreter that originally appeared on this laptop's PATH. This workspace
already has a working `.venv`; use its explicit Python path below.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.environment
.\.venv\Scripts\python.exe -m src.data --download
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

To validate an already-downloaded dataset without downloading again, run
`.\.venv\Scripts\python.exe -m src.data` without `--download`.

## Repository contents

- `src/`: environment recording and PTB-XL download, validation, and label
  generation.
- `configs/pilot.yaml`: dataset location, labels, split definitions, and seed.
- `tests/`: regression checks for label aggregation and patient leakage.
- `results/`: generated environment metadata, split class counts, per-record
  labels, and pipeline configuration/source-hash metadata.

The generated results are reproducibility artifacts for data preparation, not
model-performance or clinical evaluation results.

## Continue in Colab from VS Code

The notebook `notebooks/colab_pilot.ipynb` preserves the original GPU check and adds
transfer, Drive backup, isolated Python setup, training/resume, evaluation, and export
cells. Run them one at a time. **The local terminal is not the Colab kernel.**

1. In this local workspace, generate the transfer bundle and update the notebook:

   ```powershell
   .\.venv\Scripts\python.exe scripts/prepare_colab.py
   .\.venv\Scripts\python.exe scripts/build_colab_notebook.py
   ```

2. In VS Code Explorer, right-click `.artifacts/colab_stage01.zip` and choose
   **Upload to Colab**. The connected extension supports file uploads to `/content`.
3. Open the notebook on your Colab GPU kernel, run the Drive mount cell, and authorize
   access. Continue through the setup cells. The notebook checks the exact bundle
   SHA256 and stores a copy in `MyDrive/paper_ecg_uncertainty/`.
4. The restore checks file hashes, preserves existing edits, and reuses Stage 1 data
   without downloading it again. On a later session it can use the Drive copy.
5. The project scripts use `/content/ecg-venv/bin/python` (Python 3.11.13,
   torch 2.5.1 with CUDA 12.4). The notebook host kernel remains unchanged.
6. Run the Stage 2 training cell. Its output directory is on Drive, so the last
   completed epoch's model, optimizer, scheduler, scaler and RNG states persist.
   Reconnect, repeat setup, and run the training cell again to resume automatically.
7. Run clean-test evaluation only after validation passes the review gate. Share
   `clean_baseline.json` and `REPORT.md` before proceeding to Stage 3.

Remote execution and Drive authorization must be performed in the connected notebook;
the local assistant's shell cannot operate that GPU runtime. No full GPU result is
claimed by the local tests. Notebook uploads do not automatically sync later local
edits. Rebuild/upload a new bundle and choose a new run name when changing code/config.
For a changed bundle, restore into a new Colab folder to preserve older work.

Official extension guide: https://github.com/googlecolab/colab-vscode/wiki/User-Guide

## Stage 2 protocol

- Compact 1D residual network, BCE-with-logits, AdamW and cosine learning-rate schedule.
  Seed 42, 30 epochs, batch size 64; all settings in `configs/stage2.yaml`.
- Each recording is standardized separately per lead across time with epsilon 1e-6.
  This uses no pooled validation/test statistics, but removes absolute voltage scale.
- Exclude missing diagnostic targets from supervised learning/evaluation: 334 train,
  37 validation, 40 test. Retained sizes: 17,084 / 2,146 / 2,158.
- Checkpoint selection uses validation macro AUROC only. The test split is loaded
  only by `src.eval`. No temperature scaling or other calibration is fitted yet.
- ECE is the mean of five classwise positive-probability calibration errors, using
  15 equal-width bins. Brier averages squared error over every record/class pair.
  AUROC uses average ranks for ties; undefined class AUROC yields null, not zero.
- Validation macro AUROC below 0.88 stops the evaluation workflow for investigation.
  This is an explicit debugging heuristic, not a guaranteed benchmark performance.
  A low clean-test AUROC is flagged for review; do not tune using test performance.
- CUDA uses automatic mixed precision; CPU uses float32. One GPU is used even if
  more are visible. Deterministic algorithms and RNG states are recorded, but exact
  reproducibility across different hardware/software is not guaranteed.
- Waveforms are read into RAM once per process (about 1 GB for train/validation).
  Resuming reloads local waveforms, not the Stage 1 download/validation pipeline.

Local CPU smoke test (never a research baseline):

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m src.train --smoke --output-dir .artifacts/smoke_stage2
```

Run the smoke command only in a new directory; use `--resume` for an existing
checkpoint. Smoke mode uses 64 train and 64 validation records and one epoch.
The evaluator refuses smoke checkpoints.

Full training commands (Linux/Colab; the notebook supplies persistent Drive paths):

```bash
/content/ecg-venv/bin/python -u -m src.train --output-dir results/stage2
# After an interruption, use the same config, code and directory:
/content/ecg-venv/bin/python -u -m src.train --output-dir results/stage2 --resume
/content/ecg-venv/bin/python -u -m src.eval --run-dir results/stage2
```

Run these from the restored project root. Resume checks hashes of metadata, labels,
and baseline source files. Use a new run directory for changed experiments.
`best.pt` and `last.pt` are trusted project checkpoints; do not load third-party
pickle checkpoints with this workflow. Checkpoints and transfer bundles are ignored
by Git. Metrics and predictions are saved in the run directory alongside a report.

## What remains

Run and review the full Stage 2 clean baseline on Colab. Stage 3 then requires image
generation and testing digitizer candidates before selecting one; it cannot be
reported complete from environment checks or a smoke test.
