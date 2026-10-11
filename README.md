# Diagnosis-Aware Uncertainty for Paper-ECG Pipelines

This research project is intended to study uncertainty in ECG analysis pipelines,
with a focus on digitizing ECGs from paper images and understanding how diagnostic
context affects the resulting measurements and predictions.

## Current status

The repository currently implements the data-preparation foundation using the
PTB-XL 1.0.3 dataset. It downloads and validates the 100 Hz digital ECG records,
builds diagnostic multi-labels, and checks that the official train, validation,
and test splits do not share patients.

Paper-image digitization, uncertainty estimation, model training, calibration,
and evaluation have not yet been implemented. The current results validate the
input data and labels only; they do not test the research hypothesis or support
clinical conclusions.

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
