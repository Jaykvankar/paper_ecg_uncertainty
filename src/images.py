"""Render one PTB-XL record with the locked ECG-Image-Kit commit.

Paths are resolved before changing into the kit directory. Handwriting is never imported.
"""
import json
import os
from pathlib import Path
import random
import sys

import numpy as np

KIT_COMMIT = '27b90f56896c9fc78b05a83ca14844ea2637aa0b'
GENERATOR = Path('.tools/ecg-image-kit/codes/ecg-image-generator')
LEVELS = ('printed', 'creased', 'noisy')
CREASE = {
    'wrinkles': True, 'creases': True, 'crease_angle': 45,
    'num_creases_vertically': 3, 'num_creases_horizontally': 3,
}
NOISE = {'rotate': 5, 'noise': 40, 'crop': 0, 'temperature': 6500}


def render_record(record, output_dir, level='printed', seed=42, resolution=200):
    if level not in LEVELS:
        raise ValueError(f'Unknown page level {level}')
    header = Path(record).resolve()
    header = header if header.suffix == '.hea' else header.with_suffix('.hea')
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not header.exists():
        raise FileNotFoundError(header)
    generator = GENERATOR.resolve()
    if not (generator / 'extract_leads.py').exists():
        raise FileNotFoundError(
            'Clone https://github.com/alphanumericslab/ecg-image-kit into .tools/ecg-image-kit '
            f'and check out {KIT_COMMIT}.')
    previous = os.getcwd()
    os.chdir(generator)
    if str(generator) not in sys.path:
        sys.path.insert(0, str(generator))
    os.environ['MPLBACKEND'] = 'Agg'
    try:
        random.seed(seed)
        np.random.seed(seed)
        from scipy.stats import bernoulli
        from helper_functions import read_config_file
        from extract_leads import get_paper_ecg
        written = get_paper_ecg(
            input_file=str(header.with_suffix('.dat')), header_file=str(header), output_directory=str(output_dir),
            seed=seed, add_dc_pulse=bernoulli(0), add_bw=bernoulli(0), show_grid=bernoulli(1),
            add_print=bernoulli(0), configs=read_config_file(str(generator / 'config.yaml')),
            start_index=-1, store_configs=2, store_text_bbox=False, resolution=resolution,
            add_lead_names=False, pad_inches=0, standard_colours=5, full_mode='II',
            bbox=True, columns=1)
        png = Path(written[0]).resolve()
        layout_path = png.with_suffix('.json')
        if level in ('creased', 'noisy'):
            from CreasesWrinkles.creases import get_creased
            random.seed(seed)
            get_creased(
                str(png), str(output_dir), ifWrinkles=CREASE['wrinkles'], ifCreases=CREASE['creases'],
                crease_angle=CREASE['crease_angle'],
                num_creases_vertically=CREASE['num_creases_vertically'],
                num_creases_horizontally=CREASE['num_creases_horizontally'], bbox=False)
        if level == 'noisy':
            from ImageAugmentation.augment import get_augment
            layout = json.loads(layout_path.read_text())
            random.seed(seed)
            get_augment(
                str(png), str(output_dir), rotate=NOISE['rotate'], noise=NOISE['noise'],
                crop=NOISE['crop'], temperature=NOISE['temperature'], bbox=True, json_dict=layout)
            layout_path.write_text(json.dumps(layout, indent=4))
    finally:
        os.chdir(previous)
    return png, layout_path
