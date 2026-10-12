"""Bundle the completed local stages for transfer to Colab (standard library only)."""
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]


def main():
    result = json.loads((ROOT / 'results/stage1.json').read_text())
    data = ROOT / result['config']['data_dir']
    expected = result['waveforms_validated']
    if result['patient_leakage_assertion'] != 'PASS':
        raise ValueError('Stage 1 did not pass patient separation')
    for extension in ('dat', 'hea'):
        count = sum(1 for _ in data.glob(f'records100/**/*_lr.{extension}'))
        if count != expected:
            raise ValueError(f'Expected {expected} {extension} files, found {count}')
    if not (ROOT / 'results/environment.json').is_file():
        raise FileNotFoundError('Missing Stage 0 environment record')
    output = ROOT / '.artifacts'
    output.mkdir(exist_ok=True)
    bundle = output / 'colab_stage01.zip'
    temporary = bundle.with_suffix('.zip.part')
    hashes = {}
    # Explicit allowlist avoids bundling environments, Git credentials or tools.
    include = ['src', 'configs', 'tests', 'scripts', 'results',
               'README.md', 'requirements.txt', result['config']['data_dir']]
    if (ROOT / 'CHANGELOG.md').exists():
        include.append('CHANGELOG.md')
    with ZipFile(temporary, 'w', ZIP_DEFLATED, compresslevel=1) as archive:
        for name in include:
            path = ROOT / name
            paths = [path] if path.is_file() else sorted(path.rglob('*'))
            for file in paths:
                if not file.is_file() or '__pycache__' in file.parts:
                    continue
                relative = file.relative_to(ROOT).as_posix()
                content = file.read_bytes()
                hashes[relative] = hashlib.sha256(content).hexdigest()
                archive.writestr(relative, content)
        manifest = {'format': 1, 'stage1_records': expected, 'sha256': hashes}
        archive.writestr('transfer_manifest.json', json.dumps(manifest, indent=2))
    with ZipFile(temporary) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f'Archive integrity failure: {bad}')
    temporary.replace(bundle)
    with bundle.open('rb') as stream:
        checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    bundle.with_suffix('.zip.sha256').write_text(checksum + '  ' + bundle.name + '\n')
    print(f'Created: {bundle}')
    print(f'Size: {bundle.stat().st_size / 1024**2:.1f} MiB')
    print(f'Files: {len(hashes)}; Stage 1 records: {expected}')
    print(f'SHA256: {checksum}')
    print('This packages existing results; it does not rerun Stage 1.')


if __name__ == '__main__':
    main()
