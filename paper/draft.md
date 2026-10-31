# Tracing error and classwise calibration

The primary page is the synthetic creased page. The locked fold-10 scores are not replaced by later reruns.

## Synthetic creased page, standardized

- NORM: 0.0565 [0.0324, 0.0760]
- MI: 0.0753 [0.0479, 0.0909]
- STTC: 0.0048 [-0.0100, 0.0189]
- CD: -0.0193 [-0.0279, -0.0022]
- HYP: 0.0339 [0.0244, 0.0389]
- Mean SNR: 9.06 dB
- Records: 2158

## Synthetic creased page, raw millivolts

- NORM: 0.0661 [0.0399, 0.0864]
- MI: 0.0770 [0.0495, 0.0932]
- STTC: 0.0414 [0.0189, 0.0537]
- CD: 0.1420 [0.1229, 0.1529]
- HYP: 0.0682 [0.0535, 0.0741]
- Mean SNR: 9.06 dB
- Records: 2158

## Arm contrast, raw minus standardized

Positive means the raw-millivolt calibration shift is larger than the standardized shift.

- NORM: 0.0096 [-0.0053, 0.0223]
- MI: 0.0016 [-0.0122, 0.0162]
- STTC: 0.0366 [0.0095, 0.0529]
- CD: 0.1613 [0.1361, 0.1705]
- HYP: 0.0342 [0.0227, 0.0402]

## Lead SNR by superclass

- NORM: with label 9.64 dB (963 records); without label 8.59 dB (1195 records)
- MI: with label 9.10 dB (550 records); without label 9.05 dB (1608 records)
- STTC: with label 7.60 dB (521 records); without label 9.53 dB (1637 records)
- CD: with label 8.83 dB (496 records); without label 9.13 dB (1662 records)
- HYP: with label 5.02 dB (262 records); without label 9.62 dB (1896 records)

## Oracle scale match, raw millivolts

This check uses the clean waveform. It is not a feature of the corrector.

- NORM: 0.0465 [0.0244, 0.0678]
- MI: 0.0716 [0.0452, 0.0881]
- STTC: 0.0401 [0.0183, 0.0509]
- CD: 0.1254 [0.1060, 0.1363]
- HYP: 0.0681 [0.0536, 0.0741]
- Mean SNR: 9.06 dB
- Records: 2158

## Trace-only corrector, synthetic fold 10

The five classwise shifts above stay primary. The conduction-disturbance target is secondary. The trace map earns its place only when its CD change is closer to zero than the fold-9 logit-only map and macro AUROC stays within 0.01 of the uncorrected traced page.

Status against the uncorrected traced page: applied.
CD target role: secondary.
Trace features versus the logit-only map: earned.
Uncorrected macro AUROC 0.8733; trace-corrected macro AUROC 0.8802.
Logit-only classwise change in ECE:
- NORM: -0.0091
- MI: 0.0049
- STTC: 0.0048
- CD: -0.0059
- HYP: 0.0029
Trace-only corrected change in ECE:
- NORM: -0.0051 [-0.0185, 0.0164]
- MI: 0.0046 [-0.0137, 0.0195]
- STTC: 0.0070 [-0.0108, 0.0223]
- CD: -0.0048 [-0.0178, 0.0119]
- HYP: -0.0021 [-0.0123, 0.0152]
- Records: 2158

## Width-64 sensitivity, raw minus standardized

Side column. The locked width-32 scores remain the primary. Agreement requires the CD interval to exclude zero and keep the locked sign.

Agrees with the locked CD contrast: True.
Raw minus standardized:
- NORM: 0.0755 [0.0604, 0.0914]
- MI: 0.0629 [0.0475, 0.0720]
- STTC: 0.0124 [-0.0098, 0.0349]
- CD: 0.1539 [0.1355, 0.1687]
- HYP: 0.0125 [0.0016, 0.0272]
Standardized classwise change:
- NORM: 0.0458 [0.0200, 0.0649]
- MI: 0.0160 [-0.0089, 0.0397]
- STTC: 0.0172 [0.0004, 0.0245]
- CD: 0.0098 [-0.0064, 0.0243]
- HYP: 0.0137 [-0.0034, 0.0253]
- Mean SNR: 9.06 dB
- Records: 2158
Raw-millivolt classwise change:
- NORM: 0.1212 [0.0940, 0.1432]
- MI: 0.0789 [0.0515, 0.0980]
- STTC: 0.0295 [0.0081, 0.0422]
- CD: 0.1636 [0.1450, 0.1748]
- HYP: 0.0262 [0.0177, 0.0329]
- Mean SNR: 9.06 dB
- Records: 2158

## Tracer comparison, fold 9

The dark-run reader stays. Fold 10 was not traced again.

Chosen method: dark_run.
Fold 10 retraced: False.
Fold-9 dark-run SNR: 9.140996524104244 dB.
Fold-9 image_to_sequence SNR: 7.203484147710708 dB.
Test-page SNR kept: 9.06 dB.

## Real color scans

Color scans of printed PTB-XL pages, fold-10 overlap only. Emory, Akershus, phone photographs, and mold are outside this claim.

### Standardized

Not on disk yet.

### Raw millivolts

Not on disk yet.


Published comparisons, kept separate from these files: Strodthoff et al. 2021 report super-diagnosis macro AUC 0.928 to 0.934 on clean waveforms. ECG-Image-Kit reports mean SNR about 11.9 dB on a different synthetic set. The PhysioNet 2024 leaderboard reports SNR 12.15 and macro F 0.82 on harder hidden images.
