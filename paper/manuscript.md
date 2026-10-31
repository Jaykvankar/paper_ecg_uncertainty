# Tracing a creased ECG page changes calibration by diagnosis

Target journal: Physiological Measurement.

## Abstract

Tracing a synthetic creased 12-lead page changes classwise calibration, and the change depends on whether the frozen classifier saw z-scored leads or raw millivolts. On 2,158 labeled PTB-XL fold-10 records, the raw-minus-standardized change in expected calibration error is 0.161 for conduction disturbance, with patient-bootstrap interval 0.136 to 0.171. Records with that label do not have a worse trace than records without it. Matching each traced lead to the clean voltage scale leaves most of the raw-millivolt shift in place. A logistic map fit on fold 9 from the predicted probability alone removes the fold-10 shift. A second network with twice the width reproduces the conduction-disturbance contrast. The pages are synthetic creases. Color scans of printed PTB-XL sheets were not scored, because the scan variant id is unset.

## Introduction

PTB-XL provides 21,799 clinical 12-lead electrocardiograms and five diagnostic superclasses: normal, myocardial infarction, ST/T change, conduction disturbance, and hypertrophy. Strodthoff et al. (2021) report a clean-waveform macro AUROC of about 0.928 to 0.934 for those superclasses. That number describes discrimination on the digital signal. This paper asks a different question: after the same recording is printed as a creased page and traced, how does classwise calibration move, and does the movement differ by diagnosis and by the voltage representation used in training?

The page renderer and the column max-finder are the ECG-Image-Kit tools at commit 27b90f56. The classifier is a compact one-dimensional residual network. It is the measuring instrument. The outcome is the change in classwise expected calibration error, traced page minus clean waveform.

## Methods

PTB-XL 1.0.3 was used at 100 Hz, twelve leads, and 1,000 samples. Diagnostic superclasses follow the official SCP mapping. Records without a diagnostic label were excluded from supervised learning and evaluation: 334 in folds 1–8, 37 in fold 9, and 40 in fold 10. The remaining sizes are 17,084, 2,146, and 2,158. Folds 1–8 train, fold 9 selects the checkpoint, and fold 10 is scored once.

The network, `ECGResNet`, has a convolutional stem, six residual blocks, global average pooling, and a five-logit head. The primary models use width 32, so the block widths are 32, 64, and 128. Training uses seed 42, binary cross-entropy, AdamW, batch size 64, and 30 epochs. The checkpoint is the epoch with the highest fold-9 macro AUROC. One copy z-scores each lead. The other keeps millivolts. No temperature scaling is fit.

Each test record is rendered once as a creased page: one column, 200 dpi, wrinkles, and three vertical and three horizontal creases at 45 degrees. There is no calibration pulse, header, lead name, or handwriting. Lead boxes locate the trace. Waveform values come from the dark-run max-finder. Stored plotted pixels are not the recovered signal. Reconstruction error is the PhysioNet 2024 lead-wise signal-to-noise ratio, allowing shifts of at most 0.5 seconds and 1 mV.

The primary outcome is classwise expected calibration error on the traced page minus the same error on the clean waveform. Expected calibration error uses 15 equal-width bins on the positive-class probability, and the reported value for a record set is the mean across the five classes only when a single summary is needed. Uncertainty is a patient-cluster bootstrap with 1,000 resamples, seed 42, and percentiles 2.5 and 97.5. The arm contrast is the raw-millivolt change minus the standardized change on the same records and the same resampled patients.

Three checks sit beside the primary table. An oracle affine match maps each traced lead onto that record's clean mean and standard deviation and then rescores both frozen models. It uses the clean waveform and is not a model input. A logistic map for each superclass is fit on synthetic fold 9 for the raw-millivolt arm, using the frozen logit, the traced lead's standard deviation, and the fraction of columns with no ink. A second map uses only the frozen logit. Both are applied once to fold 10. The conduction-disturbance target of those maps was suggested by the locked fold-10 table, so it is secondary. Trace features count as useful only when they move that secondary shift closer to zero than the logit-only map and macro AUROC stays within 0.01 of the uncorrected traced page.

A width-64 copy of the same network, with block widths 64, 128, and 256, was specified before it saw fold 10. Seed, loss, splits, and the fold-9 selection rule are unchanged. Fold 10 is scored only if fold-9 macro AUROC is at least 0.88. The sensitivity agrees with the locked result when the raw-minus-standardized conduction-disturbance interval excludes zero and has the same sign as the locked contrast. Its clean AUROC is a side column. The locked width-32 files are the primary result.

Color scans from ECG-Image-Database directory D0 would be scored with the two width-32 models only after a training-fold scan correlates at least 0.80 on every lead. That gate requires a variant id copied from `dataset_description.xlsx`. Emory records, Akershus thermal paper, phone photographs, and mold are outside the claim.

## Results

Clean fold-10 macro AUROC is 0.908 for the z-scored width-32 model and 0.926 for the raw-millivolt model.

On the creased page, the z-scored model changes classwise expected calibration error by 0.056 for normal (0.032 to 0.076), 0.075 for infarction (0.048 to 0.091), 0.005 for ST/T change (−0.010 to 0.019), −0.019 for conduction disturbance (−0.028 to −0.002), and 0.034 for hypertrophy (0.024 to 0.039). The raw-millivolt model changes it by 0.066 (0.040 to 0.086), 0.077 (0.050 to 0.093), 0.041 (0.019 to 0.054), 0.142 (0.123 to 0.153), and 0.068 (0.053 to 0.074). Mean lead signal-to-noise ratio is 9.06 dB in both arms. These are the locked scores. A later rerun matched them, and the locked files were not rewritten.

The raw-minus-standardized contrast is 0.010 for normal (−0.005 to 0.022), 0.002 for infarction (−0.012 to 0.016), 0.037 for ST/T change (0.010 to 0.053), 0.161 for conduction disturbance (0.136 to 0.171), and 0.034 for hypertrophy (0.023 to 0.040). The conduction-disturbance, ST/T, and hypertrophy intervals lie above zero.

Mean lead signal-to-noise ratio with a conduction-disturbance label is 8.83 dB across 496 records, and 9.13 dB across the 1,662 records without that label. Hypertrophy is the superclass with the degraded trace: 5.02 dB on 262 records, against 9.62 dB on 1,896 records without it. ST/T change is 7.60 dB with the label and 9.53 dB without it. Infarction is 9.10 dB and 9.05 dB.

After the oracle scale match, raw-millivolt conduction-disturbance calibration error remains 0.125 (0.106 to 0.136). The fold-9 logit-only map, applied once, moves the five raw-millivolt shifts to −0.009, 0.005, 0.005, −0.006, and 0.003. The trace-only map moves conduction disturbance to −0.005 (−0.018 to 0.012). Macro AUROC on the traced page moves from 0.873 to 0.880. The trace map meets the secondary rule by about 0.001.

The width-64 models pass the fold-9 gate. Clean fold-10 macro AUROC is 0.911 (z-scored) and 0.926 (raw millivolts). On the same creased traces, the raw-minus-standardized conduction-disturbance contrast is 0.154 (0.136 to 0.169), so the pre-specified agreement rule is met. The wider z-scored model’s own conduction-disturbance change is 0.010 (−0.006 to 0.024). The wider raw-millivolt change is 0.164 (0.145 to 0.175). In this side column, the normal and infarction contrasts also lie above zero, and the ST/T interval includes zero.

On the 2,146 fold-9 creased pages, the dark-run reader used for the locked scores averages 9.14 dB. The other existing column max-finder, `image_to_sequence`, averages 7.20 dB on the same pages, with no failed traces. Fold 10 was not traced again. The test-page quality remains 9.06 dB.

The color-scan gate is blocked because the variant id is unset. No fold-10 scan file was written.

## Figures

Figure 1. Change in classwise expected calibration error after tracing the creased page, with patient-bootstrap intervals. The raw-minus-standardized contrast is included. File: `paper/figures/delta_ece.pdf`.

Figure 2. Reliability of the clean waveform and of the scale-matched trace, for both the z-scored and the raw-millivolt models. File: `paper/figures/reliability.pdf`.

## Discussion

The primary pattern is a diagnosis-specific calibration shift that is larger when the frozen model was trained on raw millivolts. For conduction disturbance, that extra shift is about 0.16 on the expected-calibration-error scale, and the patient interval excludes zero. The traces of records that carry the label are within 0.3 dB of the traces that do not, so the shift is not a worse reconstruction of those records. Putting the traced lead onto the clean mean and standard deviation removes only a small part of the raw-millivolt shift.

The same pages do not produce one signed change in every preprocessing arm. In the locked z-scored model the conduction-disturbance change is negative and its interval excludes zero. In the wider z-scored model the interval includes zero. The contrast between raw millivolts and z-scoring remains positive in both networks. The paper therefore treats the contrast, not the negative z-scored point, as the finding that the second network confirms.

A probability recalibration fit on fold 9 removes the fold-10 shift. Lead standard deviation and ink gaps move the secondary conduction-disturbance number by about one thousandth. The useful correction is a monotone map of the frozen probability. It does not require the trace features, and it was not fit on fold 10.

Clean macro AUROC near 0.91 to 0.93 places these networks beside the published PTB-XL benchmarks. Widening the primary network from 32 to 64 does not raise the raw-millivolt clean AUROC. Discrimination on the digital signal is not the result this experiment was built to move.

## Limitations

The pages are synthetic creases rendered by ECG-Image-Kit. Mean signal-to-noise ratio is 9.06 dB. A second existing tracer was less clean on fold 9, so it was not applied to fold 10. The primary network is the compact residual model. The width-64 run is one pre-specified sensitivity, one seed, scored once. Color scans of printed PTB-XL sheets remain unscored until a variant id is set and a training-fold scan passes a correlation of 0.80. The claim is about calibration after tracing. It is not a clinical decision threshold, and it does not extend to phone photographs or original thermal paper.

## Data and code

Waveforms are PTB-XL 1.0.3. Pages are generated in this repository. The locked fold-10 summaries are `results/paired_standardized.json` and `results/paired_raw_mv.json`. The width-64 summary is `results/sensitivity_wide/summary.json`. The tracer comparison is `results/tracer_sensitivity/summary.json`.

## References

Wagner P, Strodthoff N, Bousseljot RD, Kreiseler D, Lunze FI, Samek W, Schaeffter T. PTB-XL, a large publicly available electrocardiography dataset. Scientific Data. 2020;7:154.

Strodthoff N, Wagner P, Schaeffter T, Samek W. Deep learning for ECG analysis: benchmarks and insights from PTB-XL. IEEE Journal of Biomedical and Health Informatics. 2021;25:1519–1528.

Shivashankara KK, Deepanshi, Shervedani AM, Reyna MA, Clifford GD, Sameni R. ECG-Image-Kit: a synthetic image generation toolbox to facilitate deep learning-based electrocardiogram digitization. Physiological Measurement. 2024. doi:10.1088/1361-6579/ad4954.

George B. Moody PhysioNet Challenge 2024. Digitization and classification of ECG images. The lead signal-to-noise ratio used here is the challenge definition, including shifts of at most 0.5 seconds and 1 mV. https://physionetchallenges.org/2024/
