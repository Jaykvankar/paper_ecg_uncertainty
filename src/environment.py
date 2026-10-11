"""Stage 0: verify imports, seed RNGs, and record the environment."""
import json
import platform
import random
from importlib.metadata import distributions
from pathlib import Path

import numpy as np
import torch
import wfdb
import yaml


def main():
    config = yaml.safe_load(Path('configs/pilot.yaml').read_text())
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    result = {
        'python': platform.python_version(), 'platform': platform.platform(),
        'wfdb': wfdb.__version__, 'torch': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'gpu_count': torch.cuda.device_count(),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'config': config,
        'packages': dict(sorted((d.metadata['Name'], d.version) for d in distributions())),
    }
    out = Path(config['results_dir'])
    out.mkdir(parents=True, exist_ok=True)
    (out / 'environment.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = [f"Python: {result['python']}", f"wfdb: {result['wfdb']}",
             f"torch: {result['torch']}", 'import wfdb, torch: PASS',
             f"CUDA available: {result['cuda_available']}",
             f"GPU count: {result['gpu_count']}", f"Device: {result['device']}",
             f'Seed: {seed}']
    output = '\n'.join(lines) + '\n'
    (out / 'stage0_checkpoint.txt').write_text(output)
    print(output, end='')


if __name__ == '__main__':
    main()
