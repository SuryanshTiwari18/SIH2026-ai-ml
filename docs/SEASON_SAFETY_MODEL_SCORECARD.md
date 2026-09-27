# SkyGuard AI: Production Model Scorecard (Configuration 3)
**Architecture**: `TIER1B_PLUS_T3_STATIC_BASELINE` (Configuration 3)  
**Climatology Source**: `data/real_climatology/station_month_climatology_heldout_fit.json` (15-Year Non-Circular Fit)  
**Threshold Source**: `models/tier3_thresholds_v4_recalibrated.json` (Buddy Threshold = `2.5619`)  
**Evaluation Splits**: `data/tier4_results/test.parquet` ($N = 15,552$) and `data/tier4_results/spatial_holdout.parquet` ($N = 34,560$)  
**Date**: September 28, 2026  
**Status**: COMPLETE, AUDITED & STANDALONE

---

## 1. Executive Summary Table

This table summarizes overall anomaly detection and diagnostic performance across both detection tracks on the temporal test set and unseen spatial holdout stations.

| Evaluation Split | Detection Mechanism | Accuracy | Precision | Recall | F1-Score | Specificity | Macro-F1 (7-Class) | Weighted-F1 (7-Class) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Test Split** ($N=15,552$) | **Track 1 (Hard Rule)** | 84.81% | 25.83% | 17.40% | 20.79% | 93.54% | — | — |
| **Test Split** ($N=15,552$) | **Track 2 (Learned Fusion)** | **90.11%** | **54.84%** | **77.55%** | **64.25%** | **91.74%** | **46.73%** | **58.17%** |
| **Spatial Holdout** ($N=34,560$) | **Track 1 (Hard Rule)** | 84.87% | 4.14% | 16.30% | 6.61% | 87.20% | — | — |
| **Spatial Holdout** ($N=34,560$) | **Track 2 (Learned Fusion)** | **76.25%** | **8.39%** | **62.82%** | **14.80%** | **76.71%** | **25.65%** | **21.38%** |

*Definitions*:
* **Binary Task**: Ground truth `is_anomaly` vs. Track 1 `hard_rule_flagged` and Track 2 `fusion_flagged`.
* **Multi-Class Task**: Ground truth `anomaly_type` (7 fault categories) vs. Track 2 `fusion_predicted_type`.
* **Macro-F1**: Unweighted arithmetic mean of F1 scores across the 7 fault classes.
* **Weighted-F1**: Support-weighted mean of F1 scores across the 7 fault classes.

---

## 2. Multi-Class Fault-Type Diagnostic Scorecard (Track 2)

Track 2's LightGBM meta-classifier classifies flagged telemetry into specific physical failure modes (`fusion_predicted_type`).

### 2.1 Test Split Multi-Class Metrics ($N = 15,552$)

| Fault Category | Support ($N$) | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Precision | Recall | F1-Score | Diagnostic Characteristic |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`communication_dropout`** | 134 | 134 | 0 | 0 | **100.00%** | **100.00%** | **100.00%** | Flawless detection (NaN rule) |
| **`data_corruption`** | 20 | 20 | 1,046 | 0 | **1.88%** | **100.00%** | **3.68%** | Catch-all for Tier 1 overrides |
| **`spike_or_drop`** | 17 | 11 | 3 | 6 | **78.57%** | **64.71%** | **70.97%** | High precision transient filter |
| **`calibration_drift`** | 1,124 | 742 | 306 | 382 | **70.80%** | **66.01%** | **68.32%** | Strong, balanced drift capture |
| **`power_fluctuation_glitch`** | 57 | 25 | 47 | 32 | **34.72%** | **43.86%** | **38.76%** | Moderate electrical jitter recall |
| **`frozen_sensor`** | 284 | 46 | 49 | 238 | **48.42%** | **16.20%** | **24.27%** | Low point-in-time variance recall |
| **`cross_sensor_inconsistency`** | 146 | 25 | 66 | 121 | **27.47%** | **17.12%** | **21.10%** | Often grouped into drift / glitch |
| **Macro Average (7 Classes)** | **1,782** | — | — | — | **51.69%** | **58.27%** | **46.73%** | Unweighted class performance |
| **Weighted Average (7 Classes)**| **1,782** | — | — | — | **67.61%** | **59.93%** | **58.17%** | Volume-weighted performance |

---

### 2.2 Spatial Holdout Multi-Class Metrics ($N = 34,560$)

| Fault Category | Support ($N$) | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Precision | Recall | F1-Score | Diagnostic Characteristic |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`communication_dropout`** | 71 | 71 | 0 | 0 | **100.00%** | **100.00%** | **100.00%** | Zero spatial transfer loss |
| **`data_corruption`** | 9 | 9 | 4,385 | 0 | **0.20%** | **100.00%** | **0.41%** | Tier 1B hill-station false alarm sink |
| **`power_fluctuation_glitch`** | 30 | 11 | 47 | 19 | **18.97%** | **36.67%** | **25.00%** | Generalizes moderately to holdout |
| **`frozen_sensor`** | 125 | 23 | 83 | 102 | **21.70%** | **18.40%** | **19.91%** | Similar to test split behavior |
| **`calibration_drift`** | 812 | 360 | 3,204 | 452 | **10.10%** | **44.33%** | **16.45%** | Precision impacted by holdout FPR |
| **`spike_or_drop`** | 13 | 1 | 2 | 12 | **33.33%** | **7.69%** | **12.50%** | Small sample size, filtered to T1 |
| **`cross_sensor_inconsistency`** | 75 | 10 | 292 | 65 | **3.31%** | **13.33%** | **5.31%** | Sensitive to regional microclimate |
| **Macro Average (7 Classes)** | **1,135** | — | — | — | **26.80%** | **45.78%** | **25.65%** | Unweighted class performance |
| **Weighted Average (7 Classes)**| **1,135** | — | — | — | **17.20%** | **41.76%** | **21.38%** | Volume-weighted performance |

---

## 3. Confusion Matrices

### 3.1 Binary Detection 2x2 Confusion Matrices

#### Test Split ($N = 15,552$)
* **Track 1 (Hard Rule)**:
  | Ground Truth \ Predicted | Flagged Normal (Pred Neg) | Flagged Anomaly (Pred Pos) | Total |
  | :--- | :---: | :---: | :---: |
  | **Actual Normal** | 12,880 (TN) | 890 (FP) | **13,770** |
  | **Actual Anomaly** | 1,472 (FN) | 310 (TP) | **1,782** |
  | **Total** | **14,352** | **1,200** | **15,552** |

* **Track 2 (Learned Fusion)**:
  | Ground Truth \ Predicted | Flagged Normal (Pred Neg) | Flagged Anomaly (Pred Pos) | Total |
  | :--- | :---: | :---: | :---: |
  | **Actual Normal** | 12,632 (TN) | 1,138 (FP) | **13,770** |
  | **Actual Anomaly** | 400 (FN) | 1,382 (TP) | **1,782** |
  | **Total** | **13,032** | **2,520** | **15,552** |

#### Spatial Holdout Split ($N = 34,560$)
* **Track 1 (Hard Rule)**:
  | Ground Truth \ Predicted | Flagged Normal (Pred Neg) | Flagged Anomaly (Pred Pos) | Total |
  | :--- | :---: | :---: | :---: |
  | **Actual Normal** | 29,145 (TN) | 4,280 (FP) | **33,425** |
  | **Actual Anomaly** | 950 (FN) | 185 (TP) | **1,135** |
  | **Total** | **30,095** | **4,465** | **34,560** |

* **Track 2 (Learned Fusion)**:
  | Ground Truth \ Predicted | Flagged Normal (Pred Neg) | Flagged Anomaly (Pred Pos) | Total |
  | :--- | :---: | :---: | :---: |
  | **Actual Normal** | 25,640 (TN) | 7,785 (FP) | **33,425** |
  | **Actual Anomaly** | 422 (FN) | 713 (TP) | **1,135** |
  | **Total** | **26,062** | **8,498** | **34,560** |

---

### 3.2 Full Multi-Class Confusion Matrices (Track 2)

#### Test Split Full Confusion Matrix ($N = 15,552$)
*Rows indicate ground truth classes; columns indicate model predictions.*

| True \ Pred | normal | data_corruption | comm_dropout | spike_or_drop | power_glitch | frozen_sensor | calib_drift | cross_sensor | insuf_context | Total |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **normal** | **11,764** | 890 | 0 | 0 | 20 | 24 | 204 | 0 | 868 | **13,770** |
| **data_corruption** | 0 | **20** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **20** |
| **communication_dropout**| 0 | 0 | **134** | 0 | 0 | 0 | 0 | 0 | 0 | **134** |
| **spike_or_drop** | 0 | 4 | 0 | **11** | 0 | 0 | 0 | 0 | 2 | **17** |
| **power_fluctuation_glitch**| 20 | 6 | 0 | 3 | **25** | 0 | 0 | 1 | 2 | **57** |
| **frozen_sensor** | 168 | 0 | 0 | 0 | 0 | **46** | 31 | 39 | 0 | **284** |
| **calibration_drift** | 205 | 146 | 0 | 0 | 0 | 5 | **742** | 26 | 0 | **1,124** |
| **cross_sensor_inconsistency**| 3 | 0 | 0 | 0 | 27 | 20 | 71 | **25** | 0 | **146** |
| **Total Predicted** | **12,160** | **1,066** | **134** | **14** | **72** | **95** | **1,048** | **91** | **872** | **15,552** |

---

#### Spatial Holdout Split Full Confusion Matrix ($N = 34,560$)
*Rows indicate ground truth classes; columns indicate model predictions.*

| True \ Pred | normal | data_corruption | comm_dropout | spike_or_drop | power_glitch | frozen_sensor | calib_drift | cross_sensor | insuf_context | Total |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **normal** | **25,186** | 4,280 | 0 | 0 | 29 | 66 | 3,161 | 249 | 454 | **33,425** |
| **data_corruption** | 0 | **9** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **9** |
| **communication_dropout**| 0 | 0 | **71** | 0 | 0 | 0 | 0 | 0 | 0 | **71** |
| **spike_or_drop** | 3 | 6 | 0 | **1** | 3 | 0 | 0 | 0 | 0 | **13** |
| **power_fluctuation_glitch**| 13 | 3 | 0 | 2 | **11** | 0 | 0 | 1 | 0 | **30** |
| **frozen_sensor** | 75 | 22 | 0 | 0 | 0 | **23** | 5 | 0 | 0 | **125** |
| **calibration_drift** | 322 | 74 | 0 | 0 | 0 | 14 | **360** | 42 | 0 | **812** |
| **cross_sensor_inconsistency**| 9 | 0 | 0 | 0 | 15 | 3 | 38 | **10** | 0 | **75** |
| **Total Predicted** | **25,608** | **4,394** | **71** | **3** | **58** | **106** | **3,564** | **302** | **454** | **34,560** |

---

## 4. Key Performance Insights & Operational Interpretation

1. **Where the Model Excels (Critical Telemetry & Hard Hardware Failures)**:
   - Communication dropouts (`communication_dropout`) achieve a perfect **100.00% Precision and 100.00% Recall** across all splits with zero false alarms, ensuring AWS transmission interruptions are identified with complete operational certainty.
   - Sensor calibration drift (`calibration_drift`), the dominant degradation failure mode in field AWS deployments ($N=1,124$ on test), is detected with **81.76% binary recall** and classified with a strong **70.80% precision and 68.32% F1-score** on unseen test telemetry.

2. **Understanding Precision vs. Recall Divergence**:
   - For `data_corruption`, recall is **100.00%**, but precision drops to **1.88%** (test) and **0.20%** (holdout). This occurs because Tier 1 deterministic hard overrides route all out-of-bounds readings—including normal hill-station weather violating unadjusted plains climatology bounds—into the `data_corruption` label.
   - For `calibration_drift` on spatial holdout, recall remains high at **60.34%** (binary) and **44.33%** (exact multi-class), but precision decreases to **10.10%** because regional microclimate pressure variations are occasionally misclassified by the ML fusion layer as slow pressure drifts.

3. **Operational Takeaway for Real Deployments**:
   - In field operations, SkyGuard AI's Track 2 should be utilized as the primary diagnostic alert system, providing high overall detection accuracy (**90.11% on test**) and strong anomaly recall (**77.55%**).
   - Because Tier 1 overrides currently funnel regional climatology exceedances into `data_corruption`, deploying the **Soft Advisory Warning (Option B)** recommendation will directly eliminate the 890 test and 4,280 holdout false flags, boosting downstream precision across both tracks.
