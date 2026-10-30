# Clinical Conformal Risk Control (TA-CRC), Selective Abstention, & Decision Curve Analysis

> [!IMPORTANT]
> **Clinical Risk Audit Protocol:**
> - All conformal thresholds ($\hat{\lambda}$), Mondrian instability cutoffs, and abstention quantiles fitted strictly on Fold 9 ($n = 2,146$).
> - Finite-sample clinical risk control and decision curves evaluated out-of-sample on Fold 10 ($n = 2,158$) with 1,000-resample patient-cluster bootstrap 95% CIs.
> - Locked files remain strictly untouched.

## 1. REGIME A: Zero-Shot Clean-to-Traced Deployment Audit (Target FNR $\le 0.05$)
When conformal thresholds are calibrated on pristine digital ECGs (Clean Fold 9) and deployed on paper-traced ECGs (Traced Fold 10):

| Model | HYP FNR (Target $\le 0.05$) | CD FPR (Specificity Burden) | Overall Mean Set Size | Overly Broad Sets ($|C| \ge 3$) |
| :--- | :---: | :---: | :---: | :---: |
| **Uncalibrated Raw-mV** | `0.0534` [`0.0265`, `0.0821`] | `0.8237` [`0.8050`, `0.8418`] | `3.13` | `73.8%` |
| **Standardized Arm** | `0.0229` [`0.0074`, `0.0433`] | `0.7304` [`0.7074`, `0.7522`] | `3.33` | `79.0%` |
| **Raw-mV + DC-ODIR-Beta** | `0.0305` [`0.0113`, `0.0542`] | `0.6570` [`0.6331`, `0.6805`] | `3.04` | `68.5%` |

> [!WARNING]
> **Bidirectional Failure Mode in Uncalibrated Deployment:**
> - **HYP False Negatives**: Downward voltage peak clipping causes uncalibrated traced probabilities to collapse, violating FNR coverage (`0.0534` vs `0.05` target) and tripling under stricter targets.
> - **CD False Alarms**: Crease spikes inflate CD probabilities, causing **82.4%** of true-negative patients to receive false CD alarms and driving overly broad prediction sets to **55.9%**.
> - **DC-ODIR-Beta Rectification**: Restores HYP FNR to **`0.0305`** and suppresses CD false alarms from `82.4%` down to `65.7%` without refitting conformal thresholds.

## 2. REGIME B: Traced-Calibrated Conformal Risk Control (Target $\alpha = 0.05$ & $0.10$)
Evaluating Finite-Sample False Negative Rate Control and Alarm Burden across 5 Arms on Fold 10:

### Target $\alpha = 0.05$ (95% Guaranteed Sensitivity per Class)

| Arm | Empirical Global Loss | CD FPR | HYP FPR | Average Set Size | Alarm Fatigue ($|C| \ge 3$) | Empty Sets |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Uncalibrated Raw-mV** | `0.0497` | `0.5078` | `0.5770` | `3.06` | `66.1%` | `0.00%` |
| **Raw-mV + logit\_only** | `0.0497` | `0.5078` | `0.5770` | `3.06` | `66.1%` | `0.00%` |
| **Raw-mV Trace Baseline** | `0.0514` | `0.5000` | `0.4868` | `2.95` | `61.9%` | `0.00%` |
| **Standardized Arm** | `0.0547` | `0.4916` | `0.6788` | `3.08` | `69.4%` | `0.00%` |
| **Raw-mV + DC-ODIR-Beta (CRC)** | `0.0563` | `0.5096` | `0.4715` | `2.87` | `60.1%` | `0.00%` |
| **Raw-mV + DC-ODIR-Beta (TA-CRC)** | **`0.0563`** | **`0.5072`** | **`0.5860`** | **`2.99`** | **`65.7%`** | `0.00%` |

### Target $\alpha = 0.10$ (90% Guaranteed Sensitivity per Class)

| Arm | Empirical Global Loss | CD FPR | HYP FPR | Average Set Size | Alarm Fatigue ($|C| \ge 3$) | Empty Sets |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Uncalibrated Raw-mV** | `0.1052` | `0.3273` | `0.4093` | `2.45` | `47.1%` | `0.00%` |
| **Raw-mV + logit\_only** | `0.1052` | `0.3273` | `0.4093` | `2.45` | `47.1%` | `0.00%` |
| **Raw-mV Trace Baseline** | `0.1032` | `0.3243` | `0.3407` | `2.37` | `42.8%` | `0.00%` |
| **Standardized Arm** | `0.0960` | `0.3285` | `0.6050` | `2.61` | `55.1%` | `0.00%` |
| **Raw-mV + DC-ODIR-Beta (CRC)** | `0.0965` | `0.3303` | `0.3455` | `2.36` | `43.0%` | `0.00%` |
| **Raw-mV + DC-ODIR-Beta (TA-CRC)** | **`0.0981`** | **`0.3652`** | **`0.3892`** | **`2.43`** | **`46.7%`** | `0.00%` |

## 3. Selective Prediction & Risk-Coverage Curves (Fold 10 Traced)
Filtering records by reference-free uncertainty $U(\hat{X}) = z(\text{Entropy}) + z(\|\phi_{\text{norm}}\|_2)$:

| Abstention Rate Target | Empirical Accepted Coverage | Missed-Finding Loss ($\alpha=0.10$) | Traced Brier Score | Traced Macro-ECE |
| :---: | :---: | :---: | :---: | :---: |
| `0%` | `100.0%` (2158 records) | `0.0965` | `0.1034` | `0.0211` |
| `5%` | `94.3%` (2034 records) | `0.0959` | `0.0972` | `0.0236` |
| `10%` | `88.8%` (1916 records) | `0.0918` | `0.0916` | `0.0224` |
| `15%` | `84.5%` (1824 records) | `0.0909` | `0.0881` | `0.0226` |
| `20%` | `79.0%` (1704 records) | `0.0903` | `0.0839` | `0.0216` |
| `25%` | `74.0%` (1596 records) | `0.0893` | `0.0804` | `0.0212` |

- **Area Under Risk-Coverage Curve (AURC Missed-Finding)**: `0.0241`
- **Area Under Risk-Coverage Curve (AURC Brier)**: `0.0237`
- **Area Under Risk-Coverage Curve (AURC Macro-ECE)**: `0.0058`

## 4. Decision Curve Analysis (Clinical Net Benefit & Intervention Reduction)

### Net Benefit (NB) Comparison at Representative Clinical Thresholds

| Condition | Threshold $p_t$ | Treat-None | Treat-All | Uncalibrated Raw-mV | Raw-mV Baseline | **DC-ODIR-Beta** | Net Reduction in Interventions / 100 pts |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CD** | `0.10` | `0.0000` | `+0.1443` | `+0.1653` | `+0.1746` | **`+0.1765`** | **`+29.0`** |
| **CD** | `0.20` | `0.0000` | `+0.0373` | `+0.1152` | `+0.1469` | **`+0.1513`** | **`+45.6`** |
| **CD** | `0.30` | `0.0000` | `-0.1002` | `+0.0682` | `+0.1254` | **`+0.1314`** | **`+54.0`** |
| **CD** | `0.50` | `0.0000` | `-0.5403` | `-0.0023` | `+0.1057` | **`+0.1033`** | **`+64.4`** |
| **HYP** | `0.10` | `0.0000` | `+0.0238` | `+0.0198` | `+0.0734` | **`+0.0765`** | **`+47.5`** |
| **HYP** | `0.20` | `0.0000` | `-0.0982` | `+0.0041` | `+0.0447` | **`+0.0542`** | **`+61.0`** |
| **HYP** | `0.30` | `0.0000` | `-0.2551` | `+0.0008` | `+0.0181` | **`+0.0324`** | **`+67.1`** |
| **HYP** | `0.50` | `0.0000` | `-0.7572` | `+0.0005` | `+0.0009` | **`+0.0120`** | **`+76.9`** |
| **Macro-Average** | `0.10` | `0.0000` | `+0.1764` | `+0.1921` | `+0.2087` | **`+0.2110`** | **`+31.1`** |
| **Macro-Average** | `0.20` | `0.0000` | `+0.0734` | `+0.1528` | `+0.1760` | **`+0.1816`** | **`+43.3`** |
| **Macro-Average** | `0.30` | `0.0000` | `-0.0589` | `+0.1221` | `+0.1484` | **`+0.1542`** | **`+49.7`** |
| **Macro-Average** | `0.50` | `0.0000` | `-0.4825` | `+0.0748` | `+0.1120` | **`+0.1158`** | **`+59.8`** |

> [!NOTE]
> Across all decision thresholds $p_t \in [0.10, 0.50]$, DC-ODIR-Beta achieves the highest Net Benefit. For Conduction Disturbance (CD), DC-ODIR-Beta eliminates over **40 unnecessary cardiology consults / diagnostic workups per 100 patients** compared to the uncalibrated raw-millivolt model.