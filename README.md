# SkyGuard AI: Intelligent AWS Telemetry Quality Control & Anomaly Detection
**Smart India Hackathon 2026 — Problem Statement 26073 (Disaster Management Theme)**  
*Automated Quality Control & Diagnostic Anomaly Detection for Surface Telemetry across Indian Automated Weather Stations (IMD/MoES)*

---

## 1. Project Summary

**SkyGuard AI** is a real-time, physics-informed machine learning and deterministic quality control system engineered to detect, classify, explain, and mitigate sensor anomalies across India's Automated Weather Station (AWS) network. Developed for Smart India Hackathon 2026 (Problem Statement 26073), the platform resolves the critical operational challenge of automated meteorological quality control: achieving high sensitivity to subtle sensor faults (such as transducer calibration drift and frozen flatlines) while maintaining near-zero false-positive rates during genuine extreme atmospheric phenomena (including convective storm squalls, intense heatwaves, and radiation fog). The solution organizes detection into an integrated 5-tier pipeline:
- **Tier 1 (Deterministic Physical QC)**: Validates physical limits, sentinel values, and communication dropouts using hard thermodynamic boundaries at sub-microsecond latency.
- **Tier 2 (Temporal Machine Learning)**: Detects abrupt rate-of-change spikes, power glitches, and sequence flatlines via unsupervised Isolation Forests and Deep GRU-Autoencoders ($12\text{h}$ sliding windows).
- **Tier 3 (Multivariate & Spatial Consensus)**: Attributes psychrometric inconsistency ($T$ vs. $RH$) using hourly Mahalanobis distance with climate-zone fallbacks, and identifies regional agreement via 3-nearest-neighbor spatial buddy checks.
- **Tier 4 (Two-Track Decision Fusion)**: Routes incoming observations through a dual-track architecture—dispatching immediate operational alerts via a Gated Hard Rule while directing subtle, low-rate-of-change degradations into an asynchronous Technical Maintenance Queue via a LightGBM meta-classifier.
- **Tier 5 (TreeSHAP Explainability & Predictive Maintenance)**: Generates human-readable diagnostic rationales via TreeSHAP feature attribution, tracks continuous sensor health via an Exponential Moving Average (EMA) Sensor Health Index (SHI), and recommends 3-NN spatial median value reconstructions.

---

## 2. Key Results (Verified Operational Benchmarks)

All metrics are extracted directly from [`docs/FINAL_EVALUATION.md`](file:///d:/SIH/docs/FINAL_EVALUATION.md):

### 2.1 Two-Track Anomaly Recall Rollup

| Fault Category | Assigned Routing Track | Test Split Recall (Steps) | Spatial Holdout Recall (Steps) | Primary Detection Mechanism |
| :--- | :---: | :---: | :---: | :--- |
| **`data_corruption`** | Track 1 | **100.00%** (20 / 20) | **100.00%** (9 / 9) | Tier 1 exact sentinel match (`-999.0`) |
| **`communication_dropout`** | Track 1 | **100.00%** (134 / 134) | **100.00%** (71 / 71) | Tier 1 concurrent three-channel null check |
| **`spike_or_drop`** | Track 1 | **76.47%** (13 / 17) | **100.00%** (13 / 13) | Tier 2 rate-of-change + spatial isolation gate |
| **`power_fluctuation_glitch`** | Track 1 | **59.65%** (34 / 57) | **53.33%** (16 / 30) | Tier 2 1-hour rolling volatility surge |
| **`frozen_sensor`** | Track 2 | **41.55%** (118 / 284) | **40.00%** (50 / 125) | Tier 2 GRU reconstruction + Tier 4 LightGBM |
| **`calibration_drift`** | Track 2 | **80.96%** (910 / 1,124) | **51.97%** (422 / 812) | Tier 3 Mahalanobis distance ($D_M$) creep |
| **`cross_sensor_inconsistency`** | Track 2 | **97.95%** (143 / 146) | **88.00%** (66 / 75) | Tier 3 August-Roche-Magnus psychrometric breakdown |
| **Total Anomaly Set** | — | **76.99%** (1,372 / 1,782) | **57.00%** (647 / 1,135) | Unified Two-Track Architecture |

- **Normal False Positive Rate (FPR)**:
  - **Test Split** ($13,770\text{ normal steps}$): Track 1 Operational FPR = **0.21%** ($29 / 13,770$), Track 2 Maintenance FPR = **1.91%** ($263 / 13,770$), Combined = **2.12%**.
  - **Spatial Holdout Split** ($33,425\text{ normal steps}$): Track 1 Operational FPR = **0.01%** ($4 / 33,425$), Track 2 Maintenance FPR = **21.49%** ($7,183 / 33,425$ — concentrated on high-altitude microclimate station `AWS_IND_H03`).

### 2.2 Extreme Weather Hardening & Latency
- **5-Event Severe Weather Audit** ($1,378\text{ steps}$ across heatwaves, squalls, and fog): Track 1 False Positive Rate = **0.22%** ($3 / 1,378\text{ steps}$, $99.78\%$ specificity).
- **H01 Convective Storm Squall Suppression**: Standalone GRU-AE flagged $38 / 77\text{ steps}$ ($49.35\%$ FPR); Two-Track spatial consensus suppressed **35 of 38 false alarms (92.11%)**, leaving only 3 alarms ($3.90\%$ event FPR).
- **Predictive Lead Time**: Detects sensor calibration drift with an average **+5.5 hours of advance warning** prior to formal static alerting.
- **Inference Latency**: Mean = **8.99 ms**, Median (P50) = **0.09 ms**, P99 = **31.74 ms** (beating the $<500\text{ ms}$ operational SLA by **15.7×**). Sequential throughput: **111.3 rows/sec**.

---

## 3. Quick Start: Running the Production Pipeline

### 3.1 Environment Setup
Python 3.10+ is required. Install the necessary dependencies:
```bash
pip install torch lightgbm shap scikit-learn pandas numpy joblib
```
*(Or install via `pip install -r requirements.txt`)*

### 3.2 Ingesting Telemetry with `SkyGuardPipeline`
Use the pre-trained, persisted pipeline models directly without retraining:

```python
from src.skyguard_pipeline import SkyGuardPipeline

# 1. Initialize pipeline once at application startup (loads all 8 model artifacts into memory)
pipeline = SkyGuardPipeline()

# 2. Ingest an incoming telemetry record
telemetry_row = {
    "station_id": "AWS_IND_C02",
    "timestamp": "2026-07-22 01:20:00",
    "temperature_c": 26.10,
    "pressure_hpa": 1001.94,
    "humidity_pct": 95.99
}

# 3. Process the observation
result = pipeline.process_row(telemetry_row)

# 4. Inspect diagnostic decision payload
print(f"Is Anomaly:       {result['is_anomaly']}")
print(f"Predicted Fault:  {result['predicted_type']} (Confidence: {result['confidence']:.2f})")
print(f"Routing Track:    Track {result['which_track']} (1 = Real-time Alert, 2 = Maintenance Queue)")
print(f"TreeSHAP Reason:  {result['shap_rationale_text']}")
print(f"Sensor Health:    {result['sensor_health_index']:.1f} / 100.0")
```

### 3.3 Running the Live 9-Scenario Demonstration
To execute the end-to-end demo demonstrating baseline normal telemetry, all 7 sensor fault types, and genuine extreme convective weather squall suppression:
```bash
python examples/run_pipeline_demo.py
```

---

## 4. Full Pipeline Reproduction from Scratch

To regenerate all synthetic datasets, extract features, retrain all machine learning models, and reproduce every evaluation report from raw code:

```bash
# Step 1: Generate 138,240 rows of physics-informed AWS telemetry across 16 stations (60 days)
python generate.py

# Step 2: Validate thermodynamic, barometric, and climatological invariants
python validate.py

# Step 3: Run comprehensive Exploratory Data Analysis & baseline benchmarking
python eda_pipeline.py

# Step 4: Extract 19 engineered features and fit zero-leakage standard scaler
python -m src.features

# Step 5: Evaluate Tier 1 Physical Quality Control filter
python -m src.tier1_qc

# Step 6: Train and evaluate Tier 2 Isolation Forest & Deep GRU-Autoencoder
python -m src.tier2_temporal_ml

# Step 7: Fit and evaluate Tier 3 Dynamic Mahalanobis & Spatial Buddy models
python -m src.tier3_multivariate_spatial

# Step 8: Train Tier 4 LightGBM meta-classifier and evaluate Two-Track decision engine
python -m src.tier4_fusion

# Step 9: Compute Tier 5 TreeSHAP attributions and fit Sensor Health Index trajectories
python -m src.tier5_explainability

# Step 10: Run the integrated end-to-end pipeline demonstration
python examples/run_pipeline_demo.py
```

---

## 5. Repository Structure

```
SIH/
├── configs/          # Configuration files and hyperparameter definitions
├── data/
│   ├── raw/          # Master synthetic dataset and generation metadata audit trail
│   ├── splits/       # Canonical temporal (train/val/test) and spatial holdout splits
│   ├── features/     # Tier-1-cleansed 19-dimensional engineered feature tables
│   ├── tier1_results/# Output telemetry flagged by Tier 1 Physical QC
│   ├── tier2_results/# Isolation Forest and GRU-Autoencoder inference scores
│   ├── tier4_results/# Master row-reconciled fusion decisions and predictions
│   └── tier5_results/# TreeSHAP attribution matrices and Sensor Health Index logs
├── docs/             # Technical specifications, mathematical formulations, and evaluation reports
├── examples/         # Standalone, executable demonstration scripts
├── models/           # Persisted machine learning models, scalers, and threshold dictionaries
├── notebooks/        # Research exploration and visualization notebooks
├── scratch/          # Verification scripts and intermediate audit tooling
└── src/              # Production source code implementing Tiers 1 through 5
```

### Folder Descriptions:
- **`configs/`**: JSON configuration files specifying station coordinates and pipeline parameters.
- **`data/`**: Parquet telemetry stores capturing raw observations, canonical splits, engineered features, and intermediate tier predictions.
- **`docs/`**: Comprehensive markdown documentation covering data dictionaries, meteorological modeling, and tier evaluation reports.
- **`examples/`**: Minimal runnable scripts for deploying and testing the callable pipeline.
- **`models/`**: Serialized model artifacts (`.joblib`, `.pt`, `.json`) loaded during pipeline initialization.
- **`src/`**: Core modular Python package implementing physics, feature engineering, and the 5 detection tiers.

---

## 6. Documentation Index

For in-depth narrative, theoretical background, and per-tier evaluation reports, consult the dedicated documentation:

- **[WALKTHROUGH.md](file:///d:/SIH/WALKTHROUGH.md)**: **The complete project narrative** — problem framing, synthetic formulation, real bugs discovered and resolved, tier-by-tier evaluation, and disclosed operational limitations.
- **[docs/PIPELINE_USAGE.md](file:///d:/SIH/docs/PIPELINE_USAGE.md)**: Production API manual, input/output schemas, cold-start handling, and deployment configurations.
- **[docs/FINAL_EVALUATION.md](file:///d:/SIH/docs/FINAL_EVALUATION.md)**: Final unified rollup metrics, severe weather audit, latency profiling, and self-verification checklist.
- **[docs/DATA_DICTIONARY.md](file:///d:/SIH/docs/DATA_DICTIONARY.md)**: Telemetry schema contract and 7-fault anomaly definitions.
- **[docs/METEOROLOGICAL_MODELING.md](file:///d:/SIH/docs/METEOROLOGICAL_MODELING.md)**: Governing atmospheric equations (ISA barometric, Magnus dew point, $S_2$ tides, synoptic AR(1) drift).
- **[docs/EDA_INSIGHTS.md](file:///d:/SIH/docs/EDA_INSIGHTS.md)**: Exploratory data analysis, autocorrelation timescales, baseline benchmarks, and feature recipes.
- **[docs/FEATURE_DICTIONARY.md](file:///d:/SIH/docs/FEATURE_DICTIONARY.md)**: Specification of all 19 engineered features and lookback gap-edge handling.
- **[docs/TIER1_EVALUATION.md](file:///d:/SIH/docs/TIER1_EVALUATION.md)**: Tier 1 Physical Quality Control validation.
- **[docs/TIER2_EVALUATION.md](file:///d:/SIH/docs/TIER2_EVALUATION.md)**: Tier 2 Isolation Forest vs. GRU-Autoencoder benchmark and squall false-alarm investigation.
- **[docs/TIER3_EVALUATION.md](file:///d:/SIH/docs/TIER3_EVALUATION.md)**: Tier 3 Mahalanobis consistency, spatial buddy-check, and climate-zone fallback resolution.
- **[docs/TIER4_EVALUATION.md](file:///d:/SIH/docs/TIER4_EVALUATION.md)**: Tier 4 Two-Track decision engine and spatial consensus gate analysis.
- **[docs/TIER5_EVALUATION.md](file:///d:/SIH/docs/TIER5_EVALUATION.md)**: Tier 5 TreeSHAP explainability, Sensor Health Index, and predictive lead-time validation.
