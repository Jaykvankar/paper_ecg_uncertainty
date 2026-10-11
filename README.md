# Diagnosis-Aware Uncertainty for Paper-ECG Pipelines

Scope: stages 0 and 1 only. No model training or hypothesis testing yet.

## Stage 0 (Windows PowerShell)

The original PATH Python was MSYS2/MinGW Python 3.11.7 without pip or
scientific packages. Use the project environment explicitly. The repair installs
CPython locally without changing PATH or the Windows registry.

```powershell
New-Item -ItemType Directory -Force .tools | Out-Null
curl.exe -fL --retry 2 https://github.com/astral-sh/uv/releases/download/0.9.2/uv-x86_64-pc-windows-msvc.zip -o .tools/uv.zip
Expand-Archive -LiteralPath .tools/uv.zip -DestinationPath .tools/uv -Force
.\.tools\uv\uv.exe python install 3.11.13 --install-dir .tools/python --no-bin --no-registry --cache-dir .tools/cache
.\.tools\uv\uv.exe venv .venv --python .tools/python/cpython-3.11.13-windows-x86_64-none/python.exe --cache-dir .tools/cache
.\.tools\uv\uv.exe pip install --python .venv/Scripts/python.exe -r requirements.txt --cache-dir .tools/cache
.\.venv\Scripts\python.exe -m src.environment
```

Direct dependencies are pinned in requirements.txt; the resolved environment is
recorded in results/environment.json. CPU operation is supported. No training or
mixed precision execution is performed in these stages.

Sources: [uv installation](https://docs.astral.sh/uv/getting-started/installation/),
[uv Python management](https://docs.astral.sh/uv/guides/install-python/),
[WFDB README](https://github.com/MIT-LCP/wfdb-python),
[PTB-XL 1.0.3](https://physionet.org/content/ptb-xl/1.0.3/).

## Stage 1

```powershell
.\.venv\Scripts\python.exe -m src.data --download
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

For an offline rerun after downloading, omit `--download`. All records are read
one at a time per worker, with 32 workers, to bound memory use. Downloads resume by
checking existing files against the official SHA256 manifest. Only records100
and metadata are downloaded (roughly 0.55 GB); records500 is not needed.
Each waveform is checked for 100 Hz, 1000 samples, 12 leads in the expected order,
mV units and finite values. `load_signal` returns a float32 `(12, 1000)` array.

Labels follow the official `example_physionet.py`: keep SCP statements with
`diagnostic == 1`, map their `diagnostic_class`, and take the set of classes.
SCP likelihood values are not thresholded. Classes have a fixed order:
NORM, MI, STTC, CD, HYP. Multi-label counts therefore need not sum to record counts.
Records without an assigned diagnostic superclass are retained in the manifest
and explicitly flagged (`has_diagnostic_label`); they are counted as `unlabeled`,
not assumed normal. Any later training must explicitly choose their handling.

Folds 1-8 are train, 9 validation/calibration, and 10 test. Patient separation is
asserted on all records before any filtering. Outputs include class_counts.csv,
record_labels.csv, stage1.json and stage1_checkpoint.txt in results/.

The installed WFDB `rdsamp` source and official PTB-XL example were read before
implementing the loader. Source checksums and the exact config are logged.
No Stage 2-4 work is performed.
