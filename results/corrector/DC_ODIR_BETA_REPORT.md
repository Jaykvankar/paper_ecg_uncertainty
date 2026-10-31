# Degradation-Conditioned Multi-Label ODIR-Beta Recalibration (DC-ODIR-Beta)

> [!IMPORTANT]
> **Calibration Protocol Integrity Guardrails:**
> - All feature normalizers, PCA bases, and DC-ODIR-Beta parameters fitted strictly on Fold 1-8 / Fold 9 ($n = 2,146$).
> - Clean reference waveforms ($X_{\text{clean}}$) and Fold-10 test labels ($y_{\text{test}}$) were strictly forbidden and never accessed during inference.
> - Optimal hyperparameters selected via 5-fold patient-grouped cross-validation within Fold 9.

## 1. Optimal Hyperparameters & Convergence (Fold 9)
- **lambda_odir**: `0.01`
- **lambda_phi**: `0.01`
- **lambda_beta**: `0.01`

## 2. Fold 10 Recalibration Performance (Raw-mV Arm)

| Class | Uncalibrated $\Delta$ECE | logit\_only $\Delta$ECE | raw\_mv Baseline $\Delta$ECE | DC-ODIR-Beta $\Delta$ECE [95% CI] |
| :--- | :---: | :---: | :---: | :---: |
| **NORM** | `+0.0661` | `-0.0662` | `-0.0615` | **`+0.0006`** [`-0.0135`, `+0.0207`] |
| **MI** | `+0.0770` | `-0.0500` | `-0.0489` | **`-0.0140`** [`-0.0211`, `+0.0084`] |
| **STTC** | `+0.0414` | `-0.0166` | `-0.0161` | **`+0.0069`** [`-0.0084`, `+0.0223`] |
| **CD** | `+0.1420` | `-0.0286` | `-0.0253` | **`-0.0076`** [`-0.0186`, `+0.0110`] |
| **HYP** | `+0.0682` | `-0.0741` | `-0.0579` | **`-0.0066`** [`-0.0161`, `+0.0081`] |

### Summary Metrics (Fold 10 Raw-mV Arm)
- **Macro-AUROC**: Uncalibrated `0.8733` | raw\_mv `0.8802` | **DC-ODIR-Beta `0.8891`**
- **Brier Score**: Uncalibrated `0.1314` | raw\_mv `0.1080` | **DC-ODIR-Beta `0.1034`**
- **Mean Absolute $\Delta$ECE**: Uncalibrated `0.0789` | raw\_mv `0.0419` | **DC-ODIR-Beta `0.0071`**

## 3. Fold 10 Recalibration Performance (Standardized Arm)

| Class | Uncalibrated $\Delta$ECE | DC-ODIR-Beta $\Delta$ECE [95% CI] |
| :--- | :---: | :---: |
| **NORM** | `+0.0565` | **`+0.0097`** [`-0.0066`, `+0.0292`] |
| **MI** | `+0.0753` | **`-0.0028`** [`-0.0150`, `+0.0149`] |
| **STTC** | `+0.0048` | **`+0.0565`** [`+0.0388`, `+0.0618`] |
| **CD** | `-0.0193` | **`+0.0298`** [`+0.0219`, `+0.0369`] |
| **HYP** | `+0.0339` | **`-0.0106`** [`-0.0269`, `+0.0063`] |

## 4. Learned Mechanistic Attribution Parameters

### Biophysical Intercept Shifts $r_k$ (per class across 8 telemetry features)
Features: `[Einthoven, Goldberger, Precordial PCA, Crease Spikes, Peak Clip, OOB HF, TP/QRS Slew, Voltage Proxy]`

| Class | $r_1$ | $r_2$ | $r_3$ | $r_4$ | $r_5$ | $r_6$ | $r_7$ | $r_8$ | Interpretation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **NORM** | `+0.035` | `+0.110` | `+0.091` | `-0.000` | `+0.084` | `+0.115` | `+0.095` | `+0.006` | Positive: rewards clean unperturbed baseline traces |
| **MI** | `+0.028` | `-0.115` | `-0.033` | `-0.049` | `-0.039` | `-0.167` | `-0.062` | `-0.022` | Negative: counteracts spurious ST/QRS noise artifacts |
| **STTC** | `+0.043` | `+0.052` | `+0.060` | `+0.002` | `+0.014` | `+0.099` | `+0.003` | `-0.002` | Subtle adjustments to baseline ST-T distortion |
| **CD** | `-0.016` | `-0.102` | `-0.095` | `-0.073` | `-0.137` | `-0.142` | `-0.040` | `-0.025` | Consistently negative: counteracts upward false-positive probability inflation from creases |
| **HYP** | `+0.067` | `+0.048` | `+0.028` | `+0.037` | `+0.062` | `+0.085` | `+0.059` | `+0.012` | Consistently positive: counteracts downward probability suppression from peak clipping |

### Cross-Class Interaction Matrix $W_{k,l}$ (Off-Diagonal Co-Occurrence)

| Class | **NORM** | **MI** | **STTC** | **CD** | **HYP** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **NORM** | `0.000` | `+0.076` | `-0.036` | `+0.171` | `+0.022` |
| **MI** | `-0.013` | `0.000` | `-0.030` | `-0.098` | `-0.058` |
| **STTC** | `-0.070` | `-0.049` | `0.000` | `+0.027` | `-0.043` |
| **CD** | `+0.054` | `+0.045` | `-0.044` | `0.000` | `-0.025` |
| **HYP** | `-0.057` | `-0.026` | `+0.044` | `+0.024` | `0.000` |

### Beta Calibration Shape Weights

| Class | $a_k^+$ | $a_k^-$ | Base Intercept $b_{k,0}$ |
| :--- | :---: | :---: | :---: |
| **NORM** | `0.9238` | `0.9484` | `+0.9181` |
| **MI** | `0.9893` | `0.9967` | `-1.0019` |
| **STTC** | `1.0978` | `0.9862` | `+0.0244` |
| **CD** | `0.9251` | `1.0043` | `-1.0140` |
| **HYP** | `0.9578` | `0.9875` | `+0.5693` |

> [!NOTE]
> When $a_k^+ \approx a_k^-$ and $\phi \approx 0$, the DC-ODIR-Beta calibrator smoothly reduces to standard Platt/Beta calibration. On degraded paper traces, it uses biophysical telemetry to selectively rectify the bidirectional CD inflation and HYP suppression.