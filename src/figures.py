"""Forest plot and draft text read from the result JSON files."""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
FIGURE = Path('paper/figures/delta_ece.pdf')
DRAFT = Path('paper/draft.md')


def _load(path):
    file = Path(path)
    if not file.exists():
        return None
    return json.loads(file.read_text())


def series():
    rows = []
    for arm, label in (('standardized', 'Standardized, synthetic'), ('raw_mv', 'Raw millivolts, synthetic')):
        payload = _load(f'results/paired_{arm}.json')
        if payload:
            rows.append((label, payload.get('patient_bootstrap'), payload.get('delta_classwise_ece')))
    oracle = _load('results/mechanism/raw_mv.json')
    if oracle:
        rows.append(('Raw oracle scale match', oracle.get('patient_bootstrap'), oracle.get('delta_classwise_ece')))
    corrected = _load('results/corrector/fold10_raw_mv.json')
    if corrected and corrected.get('patient_bootstrap'):
        rows.append(('Raw trace-only corrector', corrected.get('patient_bootstrap'), corrected.get('delta_classwise_ece')))
    for arm, label in (('standardized', 'Standardized, color scan'), ('raw_mv', 'Raw millivolts, color scan')):
        payload = _load(f'results/real_scan/fold10_{arm}.json')
        if payload:
            rows.append((label, payload.get('patient_bootstrap'), payload.get('delta_classwise_ece')))
    scan = _load('results/real_scan/fold10_raw_mv.json')
    if scan and scan.get('corrected_bootstrap'):
        rows.append(('Raw corrector, color scan', scan.get('corrected_bootstrap'), scan.get('corrected_delta_ece')))
    logit = _load('results/corrector/fold10_raw_mv.json')
    if logit and logit.get('logit_only_delta_ece'):
        rows.append(('Raw logit-only baseline', None, logit.get('logit_only_delta_ece')))
    contrast = _load('results/paired_pages/creased/test/arm_contrast.json')
    if contrast:
        rows.append(('Raw minus standardized', contrast, contrast.get('point')))
    wide = _load('results/sensitivity_wide/summary.json')
    if wide and wide.get('contrast'):
        rows.append(('Width 64, raw minus standardized', wide['contrast'], wide['contrast'].get('point')))
    return rows


def write_figure(rows, path=FIGURE, snr=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if snr:
        figure, (axis, snr_axis) = plt.subplots(1, 2, figsize=(12, 6))
    else:
        figure, axis = plt.subplots(figsize=(9, 6))
        snr_axis = None
    positions = np.arange(len(CLASSES))
    span = 0.8 / max(len(rows), 1)
    for index, (label, interval, point) in enumerate(rows):
        offset = (index - (len(rows) - 1) / 2) * span
        if interval and 'point' in interval and 'low' in interval:
            center = [interval['point'][name] for name in CLASSES]
            low = [interval['point'][name] - interval['low'][name] for name in CLASSES]
            high = [interval['high'][name] - interval['point'][name] for name in CLASSES]
            axis.errorbar(center, positions + offset, xerr=[low, high], fmt='o', label=label, markersize=4)
        elif point:
            axis.plot([point[name] for name in CLASSES], positions + offset, 'o', label=label, markersize=4)
    axis.axvline(0, color='black', linewidth=0.8)
    axis.set_yticks(positions)
    axis.set_yticklabels(CLASSES)
    axis.set_xlabel('Change in classwise ECE (traced page minus clean waveform)')
    axis.legend(fontsize=7, loc='best')
    if snr_axis is not None:
        by_class = snr.get('by_class', snr)
        with_label = [by_class[name]['with_label_mean_snr_db'] for name in CLASSES]
        without_label = [by_class[name]['without_label_mean_snr_db'] for name in CLASSES]
        snr_axis.bar(positions - 0.15, with_label, width=0.3, label='With superclass')
        snr_axis.bar(positions + 0.15, without_label, width=0.3, label='Without superclass')
        snr_axis.set_xticks(positions)
        snr_axis.set_xticklabels(CLASSES)
        snr_axis.set_ylabel('Mean lead SNR (dB)')
        snr_axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def _format_snr(payload):
    if payload is None:
        return 'Not on disk yet.\n'
    by_class = payload.get('by_class', {})
    lines = []
    for name in CLASSES:
        row = by_class.get(name)
        if not row:
            continue
        lines.append(
            f"- {name}: with label {row['with_label_mean_snr_db']:.2f} dB "
            f"({row['with_label_records']} records); without label "
            f"{row['without_label_mean_snr_db']:.2f} dB ({row['without_label_records']} records)")
    return '\n'.join(lines) + '\n'


def _format_corrector(payload):
    if payload is None:
        return 'Not on disk yet.\n'
    logit = payload.get('logit_only_delta_ece') or {}
    lines = [
        f"Status against the uncorrected traced page: {payload.get('status')}.",
        f"CD target role: {payload.get('target_role')}.",
        f"Trace features versus the logit-only map: {payload.get('trace_versus_logit_only')}.",
        f"Uncorrected macro AUROC {payload.get('uncorrected_macro_auroc'):.4f}; "
        f"trace-corrected macro AUROC {payload.get('corrected_macro_auroc'):.4f}.",
        'Logit-only classwise change in ECE:',
    ]
    for name in CLASSES:
        if name in logit:
            lines.append(f'- {name}: {logit[name]:.4f}')
    lines.append('Trace-only corrected change in ECE:')
    return '\n'.join(lines) + '\n' + _format_arm(payload)


def _format_sensitivity(payload):
    if payload is None:
        return 'Not on disk yet.\n'
    lines = [
        f"Agrees with the locked CD contrast: {payload.get('agrees')}.",
        'Raw minus standardized:',
        _format_arm(payload.get('contrast')).rstrip(),
        'Standardized classwise change:',
        _format_arm(payload.get('standardized')).rstrip(),
        'Raw-millivolt classwise change:',
        _format_arm(payload.get('raw_mv')).rstrip(),
    ]
    return '\n'.join(lines) + '\n'


def _format_tracer(payload):
    if payload is None:
        return 'Not on disk yet.\n'
    validation = payload.get('validation') or {}
    lines = [
        f"Chosen method: {payload.get('method')}.",
        f"Fold 10 retraced: {payload.get('fold10_retraced')}.",
        f"Fold-9 dark-run SNR: {validation.get('dark_run_mean_snr_db')} dB.",
        f"Fold-9 image_to_sequence SNR: {validation.get('image_to_sequence_mean_snr_db')} dB.",
        f"Test-page SNR kept: {payload.get('mean_snr_db')} dB.",
    ]
    return '\n'.join(lines) + '\n'


def _format_arm(payload):
    if payload is None:
        return 'Not on disk yet.\n'
    lines = []
    interval = payload.get('patient_bootstrap') or {}
    if not interval and 'point' in payload and 'low' in payload:
        interval = payload
    point = interval.get('point') or payload.get('delta_classwise_ece') or {}
    low = interval.get('low') or {}
    high = interval.get('high') or {}
    for name in CLASSES:
        if name not in point:
            continue
        if name in low and name in high:
            lines.append(f'- {name}: {point[name]:.4f} [{low[name]:.4f}, {high[name]:.4f}]')
        else:
            lines.append(f'- {name}: {point[name]:.4f}')
    if 'mean_snr_db' in payload:
        lines.append(f'- Mean SNR: {payload["mean_snr_db"]:.2f} dB')
    if 'records' in payload:
        lines.append(f'- Records: {payload["records"]}')
    return '\n'.join(lines) + '\n'


def write_draft(path=DRAFT):
    standardized = _load('results/paired_standardized.json')
    raw = _load('results/paired_raw_mv.json')
    text = [
        '# Tracing error and classwise calibration',
        '',
        'The primary page is the synthetic creased page. The locked fold-10 scores are not replaced by later reruns.',
        '',
        '## Synthetic creased page, standardized',
        '',
        _format_arm(standardized),
        '## Synthetic creased page, raw millivolts',
        '',
        _format_arm(raw),
        '## Arm contrast, raw minus standardized',
        '',
        'Positive means the raw-millivolt calibration shift is larger than the standardized shift.',
        '',
        _format_arm(_load('results/paired_pages/creased/test/arm_contrast.json')),
        '## Lead SNR by superclass',
        '',
        _format_snr(_load('results/mechanism/snr_by_class.json')),
        '## Oracle scale match, raw millivolts',
        '',
        'This check uses the clean waveform. It is not a feature of the corrector.',
        '',
        _format_arm(_load('results/mechanism/raw_mv.json')),
        '## Trace-only corrector, synthetic fold 10',
        '',
        'The five classwise shifts above stay primary. The conduction-disturbance target is secondary. '
        'The trace map earns its place only when its CD change is closer to zero than the fold-9 logit-only map '
        'and macro AUROC stays within 0.01 of the uncorrected traced page.',
        '',
        _format_corrector(_load('results/corrector/fold10_raw_mv.json')),
        '## Width-64 sensitivity, raw minus standardized',
        '',
        'Side column. The locked width-32 scores remain the primary. Agreement requires the CD interval to exclude zero and keep the locked sign.',
        '',
        _format_sensitivity(_load('results/sensitivity_wide/summary.json')),
        '## Tracer comparison, fold 9',
        '',
        'The dark-run reader stays. Fold 10 was not traced again.',
        '',
        _format_tracer(_load('results/tracer_sensitivity/summary.json')),
        '## Real color scans',
        '',
        'Color scans of printed PTB-XL pages, fold-10 overlap only. Emory, Akershus, phone photographs, and mold are outside this claim.',
        '',
        '### Standardized',
        '',
        _format_arm(_load('results/real_scan/fold10_standardized.json')),
        '### Raw millivolts',
        '',
        _format_arm(_load('results/real_scan/fold10_raw_mv.json')),
        '',
        'Published comparisons, kept separate from these files: Strodthoff et al. 2021 report super-diagnosis macro AUC 0.928 to 0.934 on clean waveforms. ECG-Image-Kit reports mean SNR about 11.9 dB on a different synthetic set. The PhysioNet 2024 leaderboard reports SNR 12.15 and macro F 0.82 on harder hidden images.',
        '',
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(text), encoding='utf-8')
    return path


def write_reliability(path=Path('paper/figures/reliability.pdf')):
    """Clean versus scale-matched trace, using the bins saved by the oracle check."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, len(CLASSES), figsize=(12, 5), sharex=True, sharey=True)
    for row, (arm, title) in enumerate((('standardized', 'Standardized'), ('raw_mv', 'Raw millivolts'))):
        payload = _load(f'results/mechanism/{arm}.json')
        for column, name in enumerate(CLASSES):
            axis = axes[row, column]
            axis.plot([0, 1], [0, 1], color='black', linewidth=0.6)
            if payload and payload.get('reliability'):
                for key, label in (('clean', 'Clean'), ('oracle', 'Scale-matched')):
                    pairs = [
                        (item['confidence'], item['accuracy'])
                        for item in payload['reliability'][key][name]
                        if item['confidence'] is not None and item['accuracy'] is not None
                    ]
                    if pairs:
                        confidence, accuracy = zip(*pairs)
                        axis.plot(confidence, accuracy, marker='o', label=label, markersize=3)
            if row == 0 and column == len(CLASSES) - 1:
                axis.legend(fontsize=6)
            axis.set_title(f'{title}\n{name}', fontsize=8)
            if row == 1:
                axis.set_xlabel('Confidence')
            if column == 0:
                axis.set_ylabel('Observed frequency')
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


SUMMARY = Path('paper/summary.pdf')
SUMMARY_SECTIONS = (
    ('What the computer is',
     'A small residual network reads a 10-second, 12-lead heart tracing and gives a probability for five diagnoses: normal, infarction, ST/T change, conduction disturbance, and hypertrophy. Two copies were trained the same way. One rescales each lead. One keeps the original voltages.'),
    ('What already existed',
     'PTB-XL, published clean-signal accuracy near 0.93, the ECG-Image-Kit paper renderer and ink tracer, and the PhysioNet signal-quality measure.'),
    ('What we did',
     'We printed each held-out recording as a wrinkled page, traced the ink, and compared the computer\u2019s probabilities with the real diagnoses. The test set is 2,158 recordings. We then checked whether the extra error was just a faint trace, just the wrong voltage scale, or an artifact of the small network.'),
    ('What we found',
     'After tracing, probability error for conduction disturbance rises by 0.142 when the computer kept the original voltages (interval 0.123 to 0.153) and falls by 0.019 when leads were rescaled (interval \u22120.028 to \u22120.002). The gap is 0.161 (interval 0.136 to 0.171). Those recordings are not a worse picture (8.83 dB versus 9.13 dB). Matching the voltage scale leaves the error at 0.125. A wider network gives a gap of 0.154 (interval 0.136 to 0.169). Clean-signal accuracy stays 0.908 and 0.926. On 2,146 earlier pages, the tracer used here averaged 9.14 dB and the other existing tracer averaged 7.20 dB, so the test pages stay at 9.06 dB.'),
    ('What can restore the probabilities',
     'An adjustment fit on an earlier batch of pages, using only the computer\u2019s own probability, brings the later batch\u2019s probability error back to about zero. Ink measurements add about 0.001.'),
    ('What is still open',
     'Real color scans of printed sheets, once a scan variant id is available. Phone photos and original thermal paper are outside this study. The pages used here are synthetic creases, average quality 9.06 dB.'),
)


def write_summary_pdf(path=SUMMARY):
    """One plain-language page. It does not replace the manuscript."""
    import textwrap
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(8.27, 11.69))
    figure.patch.set_facecolor('white')
    y = 0.96
    figure.text(0.08, y, 'Paper heart tracings and diagnostic confidence',
                fontsize=16, fontweight='bold', va='top', ha='left')
    y -= 0.035
    figure.text(0.08, y, 'One-page summary of the locked result',
                fontsize=11, color='#444444', va='top', ha='left')
    y -= 0.04
    for title, body in SUMMARY_SECTIONS:
        figure.text(0.08, y, title, fontsize=12, fontweight='bold', va='top', ha='left')
        y -= 0.028
        for line in textwrap.wrap(body, width=92):
            figure.text(0.08, y, line, fontsize=10.5, va='top', ha='left')
            y -= 0.024
        y -= 0.016
    if y < 0.04:
        raise RuntimeError('The summary does not fit on one page')
    figure.savefig(path)
    plt.close(figure)
    return path


def main():
    rows = series()
    figure = write_figure(rows, snr=_load('results/mechanism/snr_by_class.json'))
    reliability = write_reliability()
    draft = write_draft()
    summary = write_summary_pdf()
    print(f'Wrote {figure}', flush=True)
    print(f'Wrote {reliability}', flush=True)
    print(f'Wrote {draft}', flush=True)
    print(f'Wrote {summary}', flush=True)


if __name__ == '__main__':
    main()
