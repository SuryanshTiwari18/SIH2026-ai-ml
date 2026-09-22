# SkyGuard AI: Tier 2 Temporal ML Evaluation Report

**Problem Statement 26073 | Smart India Hackathon 2026 | Disaster Management Theme**  
**Component**: `src/tier2_temporal_ml.py` (Tier 2 Temporal Machine Learning Layer)  
**Models Compared**: Isolation Forest vs. GRU-Autoencoder vs. Ensemble  
**Evaluation Date**: 2026-09-23  

---

## Executive Summary

Tier 2 of SkyGuard AI detects moderate-difficulty temporal anomalies (`spike_or_drop`, `frozen_sensor`, `power_fluctuation_glitch`) using the 12 engineered temporal and rolling volatility features from `src/features.py`. Both models are completely **unsupervised**, trained strictly on normal telemetry (`is_anomaly == False`) from the training set.

Decision thresholds were tuned **strictly on the validation set (`val.parquet`)** and frozen into `models/tier2_thresholds.json` before evaluating the test and spatial holdout sets. This prevents data leakage and mimics operational deployment.

> [!IMPORTANT]
> **Sample Size Notice**: In accordance with dataset balancing, `test/spike_or_drop` contains 6 episodes (13 rows) and `val/power_fluctuation_glitch` contains 7 episodes (61 rows). These specific sub-metrics are reported with statistical transparency acknowledging small-sample limits.

---

## 1. Model Architectures & Training Protocols

| Attribute | Model A: Isolation Forest | Model B: GRU-Autoencoder |
| :--- | :--- | :--- |
| **Algorithm** | `sklearn.ensemble.IsolationForest` | PyTorch Deep Recurrent Autoencoder |
| **Input Representation** | 12 instantaneous scaled temporal/volatility features | 72-step (12h) sliding window $\times$ 12 features |
| **Latent Bottleneck** | Ensemble of 150 Isolation Trees | 16-dimensional continuous latent space |
| **Training Objective** | Unsupervised path-length isolation | Unsupervised sequence reconstruction MSE |
| **Training Data** | 69,925 normal training rows | 32,663 normal contiguous 12h windows |
| **Window Boundaries** | Point-wise (independent rows) | Strictly intra-segment (never cross gaps or stations) |
| **Val Early Stopping** | N/A | Evaluated on 4,386 val-normal windows (best val loss: 0.3867) |

---

## 2. Threshold Selection (Validation Set Only)

Thresholds were selected strictly on `val.parquet` by optimizing the F1-score for the 3 target Tier-2 types combined:

- **Isolation Forest Threshold**: `0.5617` (Percentile: `98.9%`, Val F1: `0.2687`, Precision: `0.4557`, Recall: `0.1905`)
- **GRU-Autoencoder Threshold**: `1.0082` (Percentile: `95.0%`, Val F1: `0.2525`, Precision: `0.1923`, Recall: `0.3677`)
- **Ensemble Strategy**: Logical OR (`if_flagged | gru_flagged`) yielding Val F1: `0.2525` (Precision: `0.1923`, Recall: `0.3677`)

---

## 3. Head-to-Head Comparison Matrix (All Splits)

The table below directly answers 'which model wins' across the validation, test, and spatial holdout splits on the 3 Tier-2 target anomalies:

| Split | Metric | Isolation Forest | GRU-Autoencoder | Ensemble (`OR`) | Winning Model |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **VAL** | **Precision** | 0.4557 | 0.1923 | 0.1923 | — |
| **VAL** | **Recall** | 0.1905 | 0.3677 | 0.3677 | — |
| **VAL** | **F1-Score** | **0.2687** | **0.2525** | **0.2525** | **Isolation Forest** |
| **TEST** | **Precision** | 0.3309 | 0.1680 | 0.1680 | — |
| **TEST** | **Recall** | 0.1307 | 0.4148 | 0.4148 | — |
| **TEST** | **F1-Score** | **0.1874** | **0.2391** | **0.2391** | **Ensemble** |
| **SPATIAL_HOLDOUT** | **Precision** | 0.4062 | 0.1220 | 0.1220 | — |
| **SPATIAL_HOLDOUT** | **Recall** | 0.2321 | 0.4167 | 0.4167 | — |
| **SPATIAL_HOLDOUT** | **F1-Score** | **0.2955** | **0.1887** | **0.1887** | **Isolation Forest** |

---

## 4. Per-Type Recall Breakdown (Individual Target Faults)

Recall rates across each target anomaly type individually:

| Split | Fault Type | Sample Size | Isolation Forest | GRU-Autoencoder | Ensemble (`OR`) | Confidence Flag |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **val** | `spike_or_drop` | 15 rows | 100.00% | 100.00% | **100.00%** | Standard |
| **val** | `frozen_sensor` | 302 rows | 0.00% | 20.86% | **20.86%** | Standard |
| **val** | `power_fluctuation_glitch` | 61 rows | 93.44% | 100.00% | **100.00%** | ⚠️ Low sample (7 episodes) |
| **test** | `spike_or_drop` | 13 rows | 100.00% | 100.00% | **100.00%** | ⚠️ Low sample (6 episodes) |
| **test** | `frozen_sensor` | 284 rows | 0.00% | 27.46% | **27.46%** | Standard |
| **test** | `power_fluctuation_glitch` | 55 rows | 60.00% | 100.00% | **100.00%** | Standard |
| **spatial_holdout** | `spike_or_drop` | 13 rows | 100.00% | 100.00% | **100.00%** | Standard |
| **spatial_holdout** | `frozen_sensor` | 125 rows | 0.00% | 21.60% | **21.60%** | Standard |
| **spatial_holdout** | `power_fluctuation_glitch` | 30 rows | 86.67% | 100.00% | **100.00%** | Standard |

---

## 5. Cross-Tier Interception of Tier 3 Anomaly Types

Detection percentage on downstream Tier-3 anomaly types (`calibration_drift` and `cross_sensor_inconsistency`). These are recorded as informative auxiliary catches and excluded from Tier-2 false positive counts:

| Split | Downstream Fault Type | Total Rows | Caught by IF | Caught by GRU-AE | Caught by Ensemble |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **val** | `calibration_drift` | 1185 | 0.00% | 2.03% | **2.03%** |
| **val** | `cross_sensor_inconsistency` | 174 | 3.45% | 49.43% | **49.43%** |
| **test** | `calibration_drift` | 1124 | 0.00% | 2.22% | **2.22%** |
| **test** | `cross_sensor_inconsistency` | 146 | 4.11% | 36.30% | **36.30%** |
| **spatial_holdout** | `calibration_drift` | 812 | 0.00% | 5.05% | **5.05%** |
| **spatial_holdout** | `cross_sensor_inconsistency` | 75 | 4.00% | 40.00% | **40.00%** |

---

## 6. Genuine Extreme Weather Event Audit & Investigation

Reconstructed evaluation of the 5 severe weather phenomena from `generation_metadata.json` across their true simulation time windows:

| Event ID | Station | Event Type | Total Steps | IF False Positives | GRU False Positives | Ensemble False Positives | FPR |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | `AWS_IND_A01` | heatwave | 445 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | **0.0%** |
| **2** | `AWS_IND_H01` | convective_storm_squall | 77 | 25 (32.5%) | 38 (49.4%) | 38 (49.4%) | **49.4%** |
| **3** | `AWS_IND_H02` | temperature_inversion_fog | 138 | 0 (0.0%) | 6 (4.3%) | 6 (4.3%) | **4.3%** |
| **4** | `AWS_IND_P03` | heatwave | 584 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | **0.0%** |
| **5** | `AWS_IND_P04` | temperature_inversion_fog | 134 | 0 (0.0%) | 24 (17.9%) | 24 (17.9%) | **17.9%** |
| **TOTAL** | — | **5 Severe Events** | **1378** | **25 (1.81%)** | **68 (4.93%)** | **68 (4.93%)** | **4.93%** |

### Investigation of AWS_IND_H01 Convective Squall False Positives

During Event 2 on `AWS_IND_H01` (`convective_storm_squall`), GRU-AE flagged 38/77 rows (49.4%) and Isolation Forest flagged 25/77 rows (32.5%). Direct inspection of the physical features during this window reveals:
- Peak barometric rate-of-change: $\Delta P = -6.91\text{ hPa} / 10\text{-min}$ (total storm pressure drop: $12.21\text{ hPa}$)
- Peak temperature drop: $\Delta T = -5.64^\circ\text{C} / 10\text{-min}$ (total convective cooling: $9.37^\circ\text{C}$)
- Short-window rolling variance surge: $\sigma^2_P(1\text{h}) = 26.16, \sigma^2_T(1\text{h}) = 19.24$

From the perspective of a single isolated station observing only univariate/temporal features, **a violent convective squall exhibits the exact mathematical signature of an abrupt hardware spike or power glitch** ($|\Delta P| > 2.5\text{ hPa}$, $|\Delta T| > 3.0^\circ\text{C}$).

#### Mitigation Strategy Experiments:
1. **Window Oversampling / Upweighting**: Experimentally upweighting the 148 normal windows overlapping the H01 squall by 20x during GRU-AE training reduced the squall false alarms only marginally (from 49.4% to 46.8%). Because an extreme squall occurs once in 60 days, forcing a low-capacity bottleneck to compress both quiet diurnal waves and sudden 12 hPa drops degrades sensitivity to genuine hardware step jumps.
2. **Threshold Raising & Recall Tradeoff**: Sweeping the decision threshold on validation data reveals the following tradeoff curve:

| GRU Threshold | Val Percentile | Val Recall (Tier 2) | Val Precision | Val F1 | H01 Squall FPR (Event 2) | Total Extreme Weather FPR |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0.5980` | 90.0% | 49.47% | 12.93% | 0.2050 | 47 / 77 (61.0%) | 138 / 1378 (10.0%) |
| `1.0082` (Selected) | 95.0% | 36.77% | 19.23% | 0.2525 | 38 / 77 (49.4%) | 68 / 1378 (4.9%) |
| `1.3206` | 96.0% | 31.22% | 20.38% | 0.2466 | 32 / 77 (41.6%) | 55 / 1378 (4.0%) |
| `2.5243` | 97.0% | 20.11% | 17.51% | 0.1872 | 29 / 77 (37.7%) | 36 / 1378 (2.6%) |
| `7.2158` | 98.0% | 20.11% | 26.21% | 0.2275 | 25 / 77 (32.5%) | 28 / 1378 (2.0%) |
| `18.6869` | 98.5% | 17.72% | 30.88% | 0.2252 | 17 / 77 (22.1%) | 17 / 1378 (1.2%) |
| `38.2753` | 99.0% | 12.17% | 31.72% | 0.1759 | 5 / 77 (6.5%) | 5 / 1378 (0.4%) |
| `1267.2398` | 99.9% | 1.06% | 26.67% | 0.0204 | 0 / 77 (0.0%) | 0 / 1378 (0.0%) |

> [!IMPORTANT]
> **Key Takeaway & Handoff to Tier 4 Fusion**: Reducing H01 squall false alarms to zero via threshold manipulation collapses validation recall from 36.8% to 1.1%, completely blinding the model to hardware spikes. Single-station temporal models fundamentally cannot distinguish a localized hardware spike from a regional squall. This provides the direct empirical justification for **Tier 3 (Spatial Buddy Check)** and **Tier 4 (Multi-Tier Fusion)**: when a regional squall strikes, neighboring stations (H02, H03, H04) experience the same pressure drop simultaneously ($|\Delta P_\text{buddy}| \approx 0$), allowing the Tier 4 fusion engine to override and downweight Tier 2 false alarms.

---

## 7. Operational Viability & Model Comparison Recommendation

### A. Clarification on the 'Ensemble' Configuration
Empirical audit confirms that **`if_flagged` is a strict subset of `gru_flagged` across all splits** (there are 0 rows in train, val, test, or spatial holdout where Isolation Forest flags an anomaly that GRU-AE misses). Consequently, the logical OR ensemble (`if_flagged | gru_flagged`) is **mathematically identical to standalone GRU-Autoencoder**. There is no third 'hybrid' model—the real deployment decision is a direct two-way choice between:

1. **GRU-Autoencoder (High-Recall Deep Sequence Learner)**:
   - **Strengths**: Substantially higher recall across all splits (36.8%–41.7% vs. 13.1%–23.2% for IF). Specifically, it is the only Tier 2 model capable of catching `frozen_sensor` flatlines (20.9%–27.5% recall, whereas IF has 0.0% recall).
   - **Weaknesses**: Lower precision (12.2%–19.2%) and higher susceptibility to severe weather false alarms (4.9% across extreme events). Requires PyTorch runtime.

2. **Isolation Forest (Lightweight High-Precision Edge Guard)**:
   - **Strengths**: Much higher precision (33.1%–45.6%), near-instant scoring latency (<1 ms for 15k rows), zero false alarms on 4 out of 5 extreme weather events (heatwaves and inversions), and trivial CPU deployment footprint.
   - **Weaknesses**: Lower recall (13.1%–23.2%) and completely blind to zero-variance flatlines (`frozen_sensor` recall: 0.0%).

### B. Architectural Recommendation
- **Standard Cloud / Central Server Pipeline**: Deploy **GRU-Autoencoder** to capture maximum temporal anomalies, relying on **Tier 4 Multi-Tier Fusion** (incorporating Tier 3 spatial buddy delta features) to filter out the ~5% convective squall false alarms.
- **Low-Power Remote Edge AWS Stations**: Deploy **Isolation Forest** directly on station data loggers as an instantaneous, ultra-lightweight sanity filter for sudden spikes and voltage glitches.
