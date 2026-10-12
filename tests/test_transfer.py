import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from scripts.restore_colab import restore


class TransferChecks(unittest.TestCase):
    def archive(self, path, name='src/example.py', content=b'example', checksum=None):
        with ZipFile(path, 'w') as z:
            z.writestr(name, content)
            z.writestr('transfer_manifest.json', json.dumps({
                'format': 1, 'stage1_records': 1,
                'sha256': {name: checksum or hashlib.sha256(content).hexdigest()},
            }))

    def test_restore_resume_and_preserve_edits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive, destination = root / 'bundle.zip', root / 'restored'
            self.archive(archive)
            restore(archive, destination)
            restore(archive, destination)
            (destination / 'src/example.py').write_text('user edit')
            with self.assertRaises(FileExistsError):
                restore(archive, destination)
            self.assertEqual((destination / 'src/example.py').read_text(), 'user edit')

    def test_reject_corruption_and_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / 'bundle.zip'
            self.archive(archive, checksum='0' * 64)
            with self.assertRaises(ValueError):
                restore(archive, root / 'bad')
            self.archive(archive, name='../escape')
            with self.assertRaises(ValueError):
                restore(archive, root / 'bad')


if __name__ == '__main__':
    unittest.main()
