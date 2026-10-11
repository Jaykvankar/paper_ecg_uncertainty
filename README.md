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
