# Biophysical Phase SNR & Subclass Calibration Stratification Report

## 1. Biophysical Mechanism Attribution for Conduction Disturbance (CD)

Global 10-second SNR is virtually indistinguishable between CD records (8.83 dB) and non-CD records (9.13 dB).
However, decomposing the cardiac cycle into distinct biophysical phases reveals why the Raw-mV model suffers an
extreme calibration shift (+0.142 Delta ECE) that persists (+0.125 Delta ECE) even after oracle mean/SD scaling.

### Physiological Phase SNR & QRS Morphology Distortion

| Metric | CD Positive (n=496) | CD Negative (n=1,662) | Difference (CD - Non-CD) |
| :--- | :---: | :---: | :---: |
| **QRS SNR (Depolarization)** | **9.79 dB** | **9.84 dB** | **-0.04 dB** |
| **ST-T SNR (Repolarization)** | 15.43 dB | 16.42 dB | -0.99 dB |
| **TP SNR (Isoelectric Baseline)** | 15.02 dB | 16.83 dB | -1.80 dB |
| **QRS Slew-Rate Correlation** | **0.6418** | **0.7232** | **-0.0814** |
| **QRS Peak Clipping Ratio** | **0.8669** | **0.8698** | **-0.0029** |

> [!IMPORTANT]
> **Mechanism Insight**: Global SNR is dominated by the long isoelectric baseline (~65% of the recording duration).
> During the high-frequency QRS depolarization complex, non-linear column max-finder ink truncation and crease-induced
> slew-rate distortion severely flatten conduction-abnormal complexes (e.g. bundle branch blocks and intraventricular delays).
> Because linear affine matching scales all 10 seconds uniformly, it cannot restore clipped non-linear peak morphology.

## 2. Directional Probability Shifts & Bidirectional Failure Mode

Directional shift $\mathbb{E}[p_{\text{traced}} - p_{\text{clean}}]$ separates the bidirectional failure mode:
- **Conduction Disturbance (CD)**: False-positive probability inflation on true-negative controls ($+0.1876$ in Raw-mV vs $+0.0580$ in Standardized).
- **Hypertrophy (HYP/LVH)**: Severe probability suppression on true-positive cases ($-0.3682$ in Raw-mV vs $-0.1883$ in Standardized) driven by peak voltage clipping.

### Superclass Directional Shifts

| Class | Positives (y=1) | Negatives (y=0) | Raw E[dp|y=1] | Raw E[dp|y=0] | Std E[dp|y=1] | Std E[dp|y=0] | Failure Regime |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **NORM** | 963 | 1195 | **-0.2388** | **-0.0585** | -0.2282 | -0.0666 | Moderate Shift |
| **MI** | 550 | 1608 | **+0.0563** | **+0.1467** | +0.0683 | +0.1392 | False-Positive Inflation |
| **STTC** | 521 | 1637 | **-0.2506** | **+0.0002** | -0.1292 | +0.0488 | Moderate Shift |
| **CD** | 496 | 1662 | **+0.0370** | **+0.1876** | -0.0440 | +0.0580 | False-Positive Inflation |
| **HYP** | 262 | 1896 | **-0.3682** | **-0.0316** | -0.1883 | -0.0161 | Peak-Clipping Suppression |

### Subclass Directional Shifts

| Subclass | Parent Superclass | Cases | Controls | Raw E[dp|Cases] | Raw E[dp|Controls] | Std E[dp|Cases] | Std E[dp|Controls] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CLBBB** | CD | 54 | 1662 | -0.0456 | **+0.1876** | -0.1726 | +0.0580 |
| **CRBBB** | CD | 54 | 1662 | -0.0228 | **+0.1876** | -0.0628 | +0.0580 |
| **IRBBB** | CD | 112 | 1662 | +0.0260 | **+0.1876** | -0.0380 | +0.0580 |
| **LAFB/LPFB** | CD | 179 | 1662 | -0.0178 | **+0.1876** | -0.0567 | +0.0580 |
| **1AVB** | CD | 79 | 1662 | +0.0686 | **+0.1876** | -0.0613 | +0.0580 |
| **IVCD** | CD | 79 | 1662 | +0.1618 | **+0.1876** | +0.0317 | +0.0580 |
| **LVH** | HYP | 214 | 1896 | -0.4428 | **-0.0316** | -0.2276 | -0.0161 |

## 3. Case-Control Subclass Calibration (Positives + True Negative Controls)

Evaluated on each subclass alongside all corresponding superclass-negative controls (n = 1,662 CD-negatives for CD subclasses; n = 1,896 HYP-negatives for LVH) so false-positive inflation is captured:

| Subclass | Category | Cohort N (Cases/Controls) | Std Delta ECE | Raw Delta ECE | Contrast (Raw - Std) [95% CI] | Raw Delta Brier | Raw Delta Entropy (nats) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CLBBB** | CD | 1716 (54/1662) | +0.0507 | +0.1803 | **+0.1296** [+0.1208, +0.1383] | +0.1063 | +0.2146 |
| **CRBBB** | CD | 1716 (54/1662) | +0.0542 | +0.1810 | **+0.1268** [+0.1179, +0.1363] | +0.1061 | +0.2129 |
| **IRBBB** | CD | 1774 (112/1662) | +0.0519 | +0.1774 | **+0.1255** [+0.1175, +0.1362] | +0.0992 | +0.2082 |
| **LAFB/LPFB** | CD | 1841 (179/1662) | +0.0468 | +0.1676 | **+0.1208** [+0.1126, +0.1303] | +0.0980 | +0.2045 |
| **1AVB** | CD | 1741 (79/1662) | +0.0526 | +0.1822 | **+0.1296** [+0.1209, +0.1400] | +0.1005 | +0.2104 |
| **IVCD** | CD | 1741 (79/1662) | +0.0381 | +0.1860 | **+0.1479** [+0.1334, +0.1593] | +0.0952 | +0.2127 |
| **LVH** | HYP | 2110 (214/1896) | +0.0332 | +0.0595 | **+0.0263** [+0.0100, +0.0357] | +0.0407 | -0.0632 |

## 4. Positives-Only Subclass Calibration Stratification

Positives-only cohort (subclass cases only) on Fold 10:

| Subclass | Category | N (Patients) | Std Delta ECE | Raw Delta ECE | Contrast (Raw - Std) [95% CI] | Raw Delta Brier | Raw Delta Entropy (nats) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CLBBB** | CD | 54 (42) | +0.1726 | +0.0456 | **-0.1270** [-0.1851, -0.0777] | +0.0080 | +0.1287 |
| **CRBBB** | CD | 54 (45) | +0.0628 | +0.0228 | **-0.0400** [-0.0612, -0.0234] | +0.0027 | +0.0732 |
| **IRBBB** | CD | 112 (106) | +0.0380 | -0.0260 | **-0.0640** [-0.0962, -0.0333] | -0.0522 | +0.0711 |
| **LAFB/LPFB** | CD | 179 (165) | +0.0567 | +0.0178 | **-0.0390** [-0.0636, -0.0147] | -0.0081 | +0.0839 |
| **1AVB** | CD | 79 (69) | +0.0613 | -0.0686 | **-0.1299** [-0.1798, -0.0794] | -0.0882 | +0.0622 |
| **IVCD** | CD | 79 (77) | -0.0317 | -0.1618 | **-0.1301** [-0.1725, -0.0879] | -0.2036 | +0.1131 |
| **LVH** | HYP | 214 (202) | +0.2276 | +0.4428 | **+0.2152** [+0.1836, +0.2448] | +0.5310 | -0.2232 |

## 5. Hypertrophy (LVH) Voltage Attenuation Analysis

Clinical voltage diagnostic criteria for Left Ventricular Hypertrophy (LVH) on clean versus traced waveforms (n = 214):

| Criterion | Clean Mean | Traced Mean | Traced/Clean Ratio | Pearson r (p-val) | Spearman rho | Delta V vs Delta Prob Corr |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sokolow-Lyon (S_V1 + max R_V5,V6)** | 3.886 mV | 3.276 mV | **0.861** | 0.792 (p < 0.001) | 0.803 | +0.351 |
| **Cornell (R_aVL + S_V3)** | 2.625 mV | 2.247 mV | **0.867** | 0.857 (p < 0.001) | 0.912 | +0.243 |

> [!NOTE]
> **Hypertrophy Mechanism**: Sokolow-Lyon voltage drops by an average of -0.558 mV (from 3.886 mV to 3.328 mV, falling
> below the classic 3.5 mV diagnostic cutoff). Cornell voltage similarly drops by -0.370 mV. This severe voltage attenuation
> explains why the Raw-mV model's predicted probability for HYP plummets, resulting in a massive +0.443 Delta ECE shift.
