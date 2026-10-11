# Stage 0 and Stage 1 checkpoint

Scope: environment repair, data loading, diagnostic labels and official splits.
Stopped before Stage 2; no classifier or performance results exist yet.

## Stage 0

Command: `.\.venv\Scripts\python.exe -m src.environment`

```text
Python: 3.11.13
wfdb: 4.2.0
torch: 2.5.1+cpu
import wfdb, torch: PASS
CUDA available: False
GPU count: 0
Device: cpu
Seed: 42
```

The PATH interpreter was MSYS2 Python with no pip or scientific dependencies.
Project-local CPython fixed the import failure. Installation commands are in
README.md; resolved package versions are in results/environment.json.
Stage 0 commit: `773248b`.

## Stage 1

Command: `.\.venv\Scripts\python.exe -m src.data --download`

```text
PTB-XL 1.0.3: 21799 records loaded at 100 Hz
Waveform validation: PASS (12 leads x 1000 samples, mV, finite, SHA256 verified)
Patient leakage assertion: PASS (all three split pairs disjoint)
     split  records  patients  NORM   MI  STTC   CD  HYP  unlabeled
     train    17418     15023  7596 4379  4186 3907 2119        334
validation     2183      1942   955  540   528  495  268         37
      test     2198      1904   963  550   521  496  262         40
```

Folds 1-8/9/10 define train/validation/test. Patient separation was asserted on
all records. All 43,598 waveform/header files were checked against the official
SHA256 manifest, and every signal was loaded with WFDB. No waveform failed.

The main PhysioNet server was slow (a diagnostic curl request timed out).
Downloads completed through the official public S3 mirror, using 32 workers.
No credentials were required. See configs/pilot.yaml and results/stage1.json
for the exact configuration and source hashes. results/record_labels.csv records
every ECG ID, patient ID, fold, split and label vector.

Verification command: `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`.
Both tests passed: diagnostic multi-label aggregation (including zero likelihood
and exclusion of nondiagnostic statements), and rejection of introduced patient
leakage.

## Honest limitations

- These stages do not test the uncertainty/degradation hypothesis; no AUROC,
  calibration, digitization, or clinical conclusions can be drawn.
- 411 records have no diagnostic superclass (334/37/40 across the three splits).
  They are retained and flagged, not relabeled as normal. A future training stage
  must explicitly decide how to handle these records.
- Diagnostic labels follow the official SCP aggregation example, including
  codes with likelihood zero. Their clinical correctness was not independently
  adjudicated; label uncertainty remains.
- The installed PyTorch build is CPU-only. GPU training and mixed precision have
  not been tested.
- File integrity and waveform structure checks do not establish clinical signal
  quality. No synthetic or real ECG images have been generated or evaluated.
