# Paper-ECG calibration under tracing error

The locked question is whether tracing a synthetic PTB-XL paper page changes classwise calibration, and whether that change differs by diagnosis. The classifier and the page tracer are existing tools. Diagnosis is an outcome, not an input. The full lock is `PROTOCOL.md`.

## Current status

Both arms are trained, and clean fold-10 scores are in `results/stage2` and `results/stage2_raw`. A creased-page rerun of fold 10 matched the locked classwise calibration shifts in `results/paired_standardized.json` and `results/paired_raw_mv.json`. Those locked files were not rewritten. The same rerun stores a patient bootstrap of the raw-minus-standardized difference. The five classwise shifts stay the primary result. A conduction-disturbance corrector target is secondary.

Research training uses one NVIDIA GPU through CUDA 12.4. The same code runs in Google Colab and on a laptop with an RTX 40-series GPU. A run stops if no NVIDIA GPU is visible. `--smoke` is the only CPU training check. Checkpoints from the two machines stay in different directories. Score fold 10 once per arm.

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

- `src/`: data checks, the CUDA training run, the page renderer, the trace check, and the paired scorer.
- `configs/pilot.yaml`: dataset location, labels, split definitions, and seed.
- `configs/paired.yaml`: the locked calibration question. `PROTOCOL.md` is the same lock in words.
- `tests/`: label, leakage, training, and trace-check tests.
- `results/`: generated environment metadata, split class counts, per-record
  labels, and pipeline configuration/source-hash metadata.
- The smoke-test summary and report are kept under `results/smoke_test_50/`;
  regenerated per-record image traces are intentionally ignored because they
  are disposable cache files.

The generated results are reproducibility artifacts for data preparation, not
model-performance or clinical evaluation results.

## Continue in Colab from VS Code

The notebook `notebooks/colab_pilot.ipynb` preserves the original GPU check and adds
transfer, Drive backup, isolated Python setup, both training arms, the trace check,
paired scoring, and export cells. Run them one at a time. **The local terminal is not the Colab kernel.**

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
7. Run the raw-millivolt arm in its own Drive folder. It uses the same seed and
   network and keeps absolute voltage.
8. Run the printed-page check on training record `00001_lr`. Paired fold-10 scoring
   stays blocked until every recovered lead correlates with the clean lead.
9. Run clean-test evaluation only after validation passes the review gate. Share
   `clean_baseline.json` and `REPORT.md`. Then run one paired score per arm.

The laptop uses the same commands from the project root, with local output folders:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt -r requirements-paper.txt
python -u -m src.train --output-dir results/stage2
python -u -m src.train --config configs/stage2_raw.yaml --output-dir results/stage2_raw
python -m src.digitize --check
python -u -m src.eval --run-dir results/stage2
```

If both machines train, keep the checkpoints apart and score fold 10 once per arm.

Remote execution and Drive authorization must be performed in the connected notebook;
the local assistant's shell cannot operate that GPU runtime. No full GPU result is
claimed by the local tests. Notebook uploads do not automatically sync later local
edits. Rebuild/upload a new bundle and choose a new run name when changing code/config.
For a changed bundle, restore into a new Colab folder to preserve older work.

Official extension guide: https://github.com/googlecolab/colab-vscode/wiki/User-Guide

## Stage 2 protocol

- Compact 1D residual network, BCE-with-logits, AdamW and cosine learning-rate schedule.
  Seed 42, 30 epochs, batch size 64; all settings in `configs/stage2.yaml`.
- Each recording in the standardized arm is standardized separately per lead across
  time with epsilon 1e-6. The raw-millivolt arm, `configs/stage2_raw.yaml`, keeps
  absolute voltage. Neither arm uses pooled validation or test statistics.
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
- CUDA automatic mixed precision is used when `device: cuda` and a GPU is visible.
  Research configs stop if no NVIDIA GPU is visible. `--smoke` forces CPU and is
  not a baseline. One GPU is used even if more are visible. Deterministic algorithms
  and RNG states are recorded. Exact bitwise agreement between a Colab T4 and a
  40-series GPU is not expected; the run records the GPU name.
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

Copy the D0 color-scan variant id from `dataset_description.xlsx` into `configs/paired.yaml` before scoring real scans. Until that id and the images are on disk, the training-fold gate stays blocked and no fold-10 scan file is written. Printed and noisy pages are not a new primary.
