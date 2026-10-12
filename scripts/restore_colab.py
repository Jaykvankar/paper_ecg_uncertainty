"""Restore an explicitly trusted local transfer archive without overwriting edits."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


def restore(archive_path, destination):
    destination = Path(destination).resolve()
    with ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read('transfer_manifest.json'))
        if manifest['format'] != 1:
            raise ValueError('Unsupported transfer format')
        for name, expected in manifest['sha256'].items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or '..' in relative.parts or '\\' in name:
                raise ValueError(f'Unsafe archive path: {name}')
            target = destination / name
            if not target.resolve().is_relative_to(destination):
                raise ValueError(f'Path escapes destination: {name}')
            if target.exists():
                if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                    raise FileExistsError(f'Existing file differs; preserve your edits and restore to a new folder: {target}')
                continue
            content = archive.read(name)
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError(f'Invalid transfer hash: {name}')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    print(f"Restored {len(manifest['sha256'])} verified files; Stage 1 records: {manifest['stage1_records']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    restore(args.archive, args.destination)
