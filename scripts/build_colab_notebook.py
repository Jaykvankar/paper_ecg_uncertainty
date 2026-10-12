"""Update the Colab notebook after building .artifacts/colab_stage01.zip."""
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source, name):
    result = {'cell_type': kind, 'id': name, 'metadata': {},
              'source': textwrap.dedent(source).strip().splitlines(keepends=True)}
    # splitlines keeps all but the final newline; both forms are valid notebook source.
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result


def main():
    path = ROOT / 'notebooks/colab_pilot.ipynb'
    notebook = json.loads(path.read_text(encoding='utf-8'))
    existing = next((c for c in notebook['cells'] if c['cell_type'] == 'code'
                     and 'get_device_name(0)' in ''.join(c['source'])), None)
    checksum = (ROOT / '.artifacts/colab_stage01.zip.sha256').read_text().split()[0]
    cells = [cell('markdown', '''
        # Colab setup and Stage 2 clean baseline

        Run cells individually in order on the connected Colab GPU kernel.
        Stages 0/1 already completed locally. Stage 2 code is implemented, but no full
        training results exist until you run it. Stage 3 must wait for baseline review.

        **One-time manual step:** in VS Code Explorer, right-click
        `.artifacts/colab_stage01.zip` and choose **Upload to Colab**.
        If ignored files are hidden, use Explorer's Open File or show ignored files.
        The ZIP should arrive at `/content/colab_stage01.zip`.
        It contains data and results, not the Windows environment.

        Next authorize Google Drive when the mount cell prompts. The archive and
        every completed training epoch will be stored in your Drive. Colab runtime
        storage and the installed environment remain temporary. Keep this notebook
        saved locally with Ctrl+S. Run All is unnecessary when resuming.
    ''', 'intro')]
    if existing:
        cells.append(existing)
    cells += [cell('markdown', '''
        ## 1. Mount persistent storage
        Authorize your Google account. If VS Code cannot display the authorization,
        use its command palette: **Colab: Mount Google Drive to Server...** and run
        the generated mount cell before continuing. No local files are uploaded by mounting.
    ''', 'mount-notes'), cell('code', '''
        from pathlib import Path
        from google.colab import drive
        drive.mount('/content/drive')
        persistent = Path('/content/drive/MyDrive/paper_ecg_uncertainty')
        persistent.mkdir(parents=True, exist_ok=True)
        print('Persistent folder:', persistent)
    ''', 'mount-drive'), cell('markdown', '''
        ## 2. Verify and preserve the transfer archive
        First run: upload the local ZIP using the VS Code extension.
        Later sessions: this cell can use the saved Drive copy without uploading again.
        The checksum belongs to the bundle generated with this notebook.
    ''', 'bundle-notes'), cell('code', f'''
        import hashlib
        import shutil
        EXPECTED_SHA256 = '{checksum}'
        uploaded = Path('/content/colab_stage01.zip')
        saved_bundle = persistent / ('colab_stage01_' + EXPECTED_SHA256[:12] + '.zip')
        bundle = uploaded if uploaded.exists() else saved_bundle
        if not bundle.exists():
            raise FileNotFoundError('Upload .artifacts/colab_stage01.zip to Colab first.')
        def sha256_file(path):
            with path.open('rb') as stream:
                return hashlib.file_digest(stream, 'sha256').hexdigest()
        if sha256_file(bundle) != EXPECTED_SHA256:
            raise ValueError('Wrong or incomplete ZIP. Upload the bundle matching this notebook.')
        if bundle != saved_bundle:
            shutil.copy2(bundle, saved_bundle)
        if sha256_file(saved_bundle) != EXPECTED_SHA256:
            raise ValueError('Drive copy failed verification')
        print('Verified backup on Drive:', saved_bundle)
    ''', 'preserve-bundle'), cell('markdown', '''
        ## 3. Restore completed Stage 1 files
        This verifies transfer hashes; it does not download ECGs or rerun Stage 1.
        Existing files are reused if identical. Different files are never silently overwritten.
    ''', 'restore-notes'), cell('code', '''
        import subprocess
        import sys
        from zipfile import ZipFile
        project = Path('/content/paper_ecg_uncertainty')
        helper = Path('/content/restore_ecg_bundle.py')
        with ZipFile(bundle) as archive:
            helper.write_bytes(archive.read('scripts/restore_colab.py'))
        subprocess.run([sys.executable, str(helper), str(bundle), str(project)], check=True)
        import json
        stage1 = json.loads((project / 'results/stage1.json').read_text())
        print('Existing Stage 1 records:', stage1['waveforms_validated'])
        print('Existing leakage assertion:', stage1['patient_leakage_assertion'])
    ''', 'restore-files'), cell('markdown', '''
        ## 4. Recreate the pinned GPU environment
        Runs on Colab Linux. The notebook kernel keeps its default packages; all project
        scripts use the separate Python below. Skip installation only if that environment
        still exists and was already installed in this session.
    ''', 'environment-notes'), cell('code', '''
        import os
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'uv==0.9.2'], check=True)
        uv = shutil.which('uv')
        if uv is None:
            raise RuntimeError('uv executable was not found after installation')
        python = Path('/content/ecg-venv/bin/python')
        uv_env = dict(os.environ)
        uv_env.pop('UV_SYSTEM', None)
        if not python.exists():
            subprocess.run([uv, 'venv', '--python', '3.11.13', '/content/ecg-venv'], check=True, env=uv_env)
        subprocess.run([uv, 'pip', 'install', '--python', str(python),
                        'torch==2.5.1', '--index-url', 'https://download.pytorch.org/whl/cu124'], check=True, env=uv_env)
        subprocess.run([uv, 'pip', 'install', '--python', str(python),
                        '-r', str(project / 'requirements.txt')], check=True, env=uv_env)
        subprocess.run([str(python), '-c',
                        "import torch,wfdb; print('torch:',torch.__version__); print('wfdb:',wfdb.__version__); "
                        "assert torch.cuda.is_available(), 'GPU unavailable'; print('GPU:',torch.cuda.get_device_name(0)); "
                        "print('GPU calculation:', (torch.ones(2,device='cuda')*2).cpu().tolist())"], check=True)
    ''', 'install-environment'), cell('markdown', '''
        ## 5. Check the implementation
        These short tests cover labels, patient leakage, metrics, normalization,
        model gradients, transfer integrity, and interrupted-training recovery.
        They do not train the research baseline or repeat full Stage 1 validation.
    ''', 'test-notes'), cell('code', '''
        subprocess.run([str(python), '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                       cwd=project, check=True)
    ''', 'run-tests'), cell('markdown', '''
        ## 6. Train Stage 2, or resume it
        Uses folds 1-8 and selects the best checkpoint on fold 9. Fold 10 is not
        evaluated here. Missing diagnostic targets are excluded: 17,084 training
        and 2,146 validation records remain. Seed 42; 30 epochs; batch size 64.

        Checkpoints, optimizer/scaler/RNG states, configs, IDs and epoch logs are saved
        directly to Drive. A disconnect loses at most work since the last completed
        epoch. After reconnecting run setup cells, then this cell again with the same
        RUN_NAME. For changed code/config use a new RUN_NAME; resume rejects mismatches.
        If a run failed before its first checkpoint, also use a new RUN_NAME.
        Training duration and AUROC are not known until measured.
    ''', 'train-notes'), cell('code', '''
        RUN_NAME = 'stage2_seed42_v1'
        run_dir = persistent / RUN_NAME
        command = [str(python), '-u', '-m', 'src.train', '--output-dir', str(run_dir)]
        if (run_dir / 'last.pt').exists():
            command.append('--resume')
        subprocess.run(command, cwd=project, check=True)
        summary = json.loads((run_dir / 'training_summary.json').read_text())
        print(json.dumps(summary, indent=2))
        if summary['status'] != 'ready_for_clean_test':
            raise RuntimeError('Stop and review validation performance before test evaluation or Stage 3.')
    ''', 'train-stage2'), cell('markdown', '''
        ## 7. Evaluate the selected checkpoint on clean fold 10
        Run only after training completes and validation has passed the review gate.
        The 0.88 review threshold is a debugging heuristic, not a literature guarantee.
        Do not tune hyperparameters against fold 10. No calibration is fitted here.
    ''', 'eval-notes'), cell('code', '''
        subprocess.run([str(python), '-u', '-m', 'src.eval', '--run-dir', str(run_dir)],
                       cwd=project, check=True)
        baseline = json.loads((run_dir / 'clean_baseline.json').read_text())
        for key in ['status', 'records', 'macro_auroc', 'per_class_auroc', 'ece', 'brier']:
            print(key, ':', baseline[key])
        print('Saved report:', run_dir / 'REPORT.md')
        print('STOP: share these measured results for review before Stage 3.')
    ''', 'evaluate-stage2'), cell('markdown', '''
        ## 8. Export small result files for GitHub
        Everything in the run directory is already on Drive. This archive contains
        metrics/configs/logs/reports and predictions, excluding model checkpoints.
        Retrieve it from the extension's Colab Contents view or Google Drive.
        Keep best.pt and last.pt on Drive for subsequent stages.
    ''', 'export-notes'), cell('code', '''
        from zipfile import ZIP_DEFLATED
        exported = persistent / (RUN_NAME + '_results.zip')
        with ZipFile(exported, 'w', ZIP_DEFLATED) as archive:
            for file in sorted(run_dir.rglob('*')):
                if file.is_file() and file.suffix in {'.json', '.csv', '.md', '.npz'}:
                    archive.write(file, file.relative_to(run_dir))
        print('Results export:', exported)
        print('Checkpoints remain in:', run_dir)
    ''', 'export-results')]
    notebook['cells'] = cells
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    for entry in cells:
        if entry['cell_type'] == 'code':
            compile(''.join(entry['source']), entry['id'], 'exec')
    print(f'Updated {path}: {len(cells)} cells; preserved original GPU check output.')


if __name__ == '__main__':
    main()
