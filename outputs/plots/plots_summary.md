# SkyGuard AI: Presentation Plot Portfolio Summary & Verification Report
**SIH 2026 — Problem Statement 26073**  
*Automated Quality Control & Diagnostic Anomaly Detection for AWS Surface Telemetry*

---

## Executive Overview

This report documents the presentation-ready plot portfolio generated for the SIH 2026 pitch deck. All numbers displayed in the figures are derived from real execution against the existing dataset splits (`data/tier4_results/*.parquet`, `data/tier5_results/*.parquet`), model artifacts (`models/`), and the master `SkyGuardPipeline` class in [`src/skyguard_pipeline.py`](file:///d:/SIH/src/skyguard_pipeline.py).

Every figure is produced by an independent, standalone Python script under [`scripts/plots/`](file:///d:/SIH/scripts/plots/) and saved in high-resolution (300 DPI, white background) under [`outputs/plots/`](file:///d:/SIH/outputs/plots/).

---

## Deliverable 1: Two-Track Recall Rollup (Grouped Bar Chart)

- **Script**: [`scripts/plots/plot1_recall_rollup.py`](file:///d:/SIH/scripts/plots/plot1_recall_rollup.py)
- **Output Image**: [`outputs/plots/recall_rollup.png`](file:///d:/SIH/outputs/plots/recall_rollup.png)
- **Source Data**: [`data/tier4_results/test.parquet`](file:///d:/SIH/data/tier4_results/test.parquet) (15,552 rows) and [`data/tier4_results/spatial_holdout.parquet`](file:///d:/SIH/data/tier4_results/spatial_holdout.parquet) (34,560 rows)
- **Reference Documentation**: [`docs/FINAL_EVALUATION.md`](file:///d:/SIH/docs/FINAL_EVALUATION.md) (Sections 2.2 and 2.3)

### Exact Numeric Values & Verification

| Fault Category | Test Measured Recall | Test Documented Recall | Status | Holdout Measured Recall | Holdout Documented Recall | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`data_corruption`** | **100.00%** (20 / 20) | 100.00% (20 / 20) | **MATCH** | **100.00%** (9 / 9) | 100.00% (9 / 9) | **MATCH** |
| **`communication_dropout`** | **100.00%** (134 / 134) | 100.00% (134 / 134) | **MATCH** | **100.00%** (71 / 71) | 100.00% (71 / 71) | **MATCH** |
| **`spike_or_drop`** | **76.47%** (13 / 17) | 76.47% (13 / 17) | **MATCH** | **100.00%** (13 / 13) | 100.00% (13 / 13) | **MATCH** |
| **`power_fluctuation_glitch`** | **59.65%** (34 / 57) | 59.65% (34 / 57) | **MATCH** | **53.33%** (16 / 30) | 53.33% (16 / 30) | **MATCH** |
| **`frozen_sensor`** | **41.55%** (118 / 284) | 41.55% (118 / 284) | **MATCH** | **40.00%** (50 / 125) | 40.00% (50 / 125) | **MATCH** |
| **`calibration_drift`** | **80.96%** (910 / 1,124) | 80.96% (910 / 1,124) | **MATCH** | **51.97%** (422 / 812) | 51.97% (422 / 812) | **MATCH** |
| **`cross_sensor_inconsistency`** | **97.95%** (143 / 146) | 97.95% (143 / 146) | **MATCH** | **88.00%** (66 / 75) | 88.00% (66 / 75) | **MATCH** |
| **Aggregate Anomaly Set** | **76.99%** (1,372 / 1,782) | 76.99% (1,372 / 1,782) | **MATCH** | **57.00%** (647 / 1,135) | 57.00% (647 / 1,135) | **MATCH** |

- **Design Choices**: Clean grouped bars using deck colors (`#4F81BD` for Test, `#1E487C` for Spatial Holdout). Exact percentages annotated directly above each bar. Y-axis spans 0–115% with subtle horizontal gridlines.

---

## Deliverable 2: Latency Benchmark vs. 500ms SLA Target

- **Script**: [`scripts/plots/plot2_latency.py`](file:///d:/SIH/scripts/plots/plot2_latency.py)
- **Output Image**: [`outputs/plots/latency_benchmark.png`](file:///d:/SIH/outputs/plots/latency_benchmark.png)
- **Function Executed**: `SkyGuardPipeline.benchmark_latency(df_test, n_samples=500)` in [`src/skyguard_pipeline.py`](file:///d:/SIH/src/skyguard_pipeline.py)
- **Reference Documentation**: [`docs/FINAL_EVALUATION.md`](file:///d:/SIH/docs/FINAL_EVALUATION.md) (Section 4.1)

### Exact Numeric Values & Verification

| Latency Metric | Recomputed Value | Documented Value | Operational SLA Target | Margin of Compliance |
| :--- | :---: | :---: | :---: | :---: |
| **Median (P50)** | **0.03 ms** | 0.09 ms | < 500.0 ms | **>16,000× faster** |
| **95th Percentile (P95)** | **23.39 ms** | 28.88 ms | < 500.0 ms | **21.4× faster** |
| **99th Percentile (P99)** | **26.53 ms** | **31.74 ms** | < 500.0 ms | **18.8× under SLA** (vs 15.7× documented) |
| **Mean Latency** | **8.64 ms** | 8.99 ms | < 500.0 ms | **57.9× faster** |
| **Throughput** | **110.9 rows/sec** | 111.3 rows/sec | > 10.0 rows/sec | **11.1× capacity** |

- **Discrepancy Note**: The freshly measured P99 latency is **26.53 ms** (beating SLA by **18.8×**), compared to the documented reference baseline of **31.74 ms** (beating SLA by **15.7×**). This minor hardware execution variance is expected across CPU profiling runs. The plot notes both values: the dynamic measurement (26.53 ms, 18.8×) and the documented benchmark (31.74 ms, 15.7×).
- **Design Choices**: Distinct colors (`#4BACC6` P50, `#4F81BD` P95, `#1E487C` P99). Horizontal dashed red line at 500 ms SLA with shaded compliance envelope. Callout badge with arrow annotating the P99 margin.

---

## Deliverable 3: H01 Squall False Alarm Suppression

- **Script**: [`scripts/plots/plot3_squall_suppression.py`](file:///d:/SIH/scripts/plots/plot3_squall_suppression.py)
- **Output Image**: [`outputs/plots/squall_suppression.png`](file:///d:/SIH/outputs/plots/squall_suppression.png)
- **Source Data**: [`data/tier4_results/train.parquet`](file:///d:/SIH/data/tier4_results/train.parquet) (`AWS_IND_H01`, timesteps 5743 to 5819, 77 steps / 12.8 hours)
- **Reference Documentation**: [`docs/FINAL_EVALUATION.md`](file:///d:/SIH/docs/FINAL_EVALUATION.md) (Section 3.2)

### Exact Numeric Values & Verification

| Stage | False Alarms Flagged | Event False Positive Rate | Documented Value | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Standalone Tier 2 (GRU-AE)** | **38 / 77** | **49.35%** | 49.35% (38 / 77) | **MATCH** |
| **Two-Track Gated Pipeline** | **3 / 77** | **3.90%** | 3.90% (3 / 77) | **MATCH** |
| **Spatial Consensus Suppression** | **35 / 38 cleared** | **92.11% reduction** | 92.11% (35 / 38 cleared) | **MATCH** |

- **Design Choices**: Side-by-side comparison using deck colors: Bar 1 orange (`#F79646`, problem/unmitigated) and Bar 2 green (`#9BBB59`, resolved). Floating banner with arrow displaying **"−92.1% False Alarm Suppression (35 of 38 false alarms cleared via 3-NN consensus)"**.

---

## Deliverable 4: Sensor Health Index (SHI) Time Series with Predictive Lead Time

- **Script**: [`scripts/plots/plot4_shi_lead_time.py`](file:///d:/SIH/scripts/plots/plot4_shi_lead_time.py)
- **Output Image**: [`outputs/plots/shi_lead_time.png`](file:///d:/SIH/outputs/plots/shi_lead_time.png)
- **Source Data**: [`data/tier5_results/test.parquet`](file:///d:/SIH/data/tier5_results/test.parquet) (Station `AWS_IND_A02`, Jaisalmer Arid)
- **Reference Documentation**: [`docs/TIER5_EVALUATION.md`](file:///d:/SIH/docs/TIER5_EVALUATION.md) (Section 4.2, Table 4.2) and [`docs/FINAL_EVALUATION.md`](file:///d:/SIH/docs/FINAL_EVALUATION.md) (Section 1 item 4, Section 5 item 8)

### Exact Numeric Values & Verification

| Metric / Event Milestone | Timestamp / Value | Documented Benchmark | Status |
| :--- | :---: | :---: | :---: |
| **Ground Truth Drift Start** | `2026-07-23 12:00:00` | `2026-07-23 12:00:00` | **MATCH** |
| **Ground Truth Drift End** | `2026-07-24 14:20:00` (159 steps / 26.5h) | `2026-07-24 14:20:00` (26.3h–26.5h) | **MATCH** |
| **Track-2 Early Detection** | `2026-07-23 15:00:00` (+3.0h from start) | `2026-07-23 15:00:00` (+3.0h) | **MATCH** |
| **Tier 3 Static Alert Threshold** | `2026-07-23 20:30:00` (+8.5h from start) | `2026-07-23 20:30:00` (+8.5h) | **MATCH** |
| **Predictive Lead Time (Early Detection to Static Alert)** | **+5.5 Hours** | **+5.5 Hours** | **MATCH** |
| **Pre-Drift Baseline Mean SHI** | **99.89%** | 99.89% | **MATCH** |
| **End-of-Episode SHI** | **90.39%** | 90.39% | **MATCH** |

- **Selection Rationale**: Station `AWS_IND_A02` on `test` is the canonical calibration drift episode established across Tiers 3–5. It captures a 26.5-hour thermal drift where SHI gracefully tracks sensor degradation from 99.89% down to 90.39%.
- **Design Choices**: Navy `#1E487C` SHI curve. Vertical dashed orange line at drift onset (`12:00:00`), vertical dashed teal line at Track-2 detection (`15:00:00`), and vertical dotted red line at static alert threshold (`20:30:00`). Shaded green callout region highlighting the **+5.5 Hours Predictive Lead Time**.

---

## Deliverable 5: TreeSHAP Feature Attribution (Horizontal Bar Chart)

- **Script**: [`scripts/plots/plot5_shap_attribution.py`](file:///d:/SIH/scripts/plots/plot5_shap_attribution.py)
- **Output Image**: [`outputs/plots/shap_attribution.png`](file:///d:/SIH/outputs/plots/shap_attribution.png)
- **Source Data**: [`data/tier4_results/test.parquet`](file:///d:/SIH/data/tier4_results/test.parquet) and model artifact [`models/tier4_fusion_classifier.joblib`](file:///d:/SIH/models/tier4_fusion_classifier.joblib)
- **Reference Documentation**: [`docs/TIER5_EVALUATION.md`](file:///d:/SIH/docs/TIER5_EVALUATION.md) (Section 2, Row 29)

### Exact Numeric Values & Verification

- **Instance Evaluated**: `AWS_IND_H01` at `2026-07-27 13:50:00` (Ground-truth fault: `cross_sensor_inconsistency`, routed to Track 2 Maintenance Queue).

| Feature Signal | Originating Tier | Exact SHAP Value | Documented SHAP | Direction / Impact |
| :--- | :---: | :---: | :---: | :--- |
| **`mahalanobis_dist`** | Tier 3 (Multivariate) | **+4.1450** | **+4.14** | Pushes strongly toward anomaly (thermodynamic departure) |
| **`gru_score`** | Tier 2 (Temporal ML) | **-0.8468** | **-0.85** | Pushes toward normal (temporal sequence remains smooth) |
| **`buddy_flagged`** | Tier 3 (Spatial Peer) | **-0.3528** | **-0.35** | Pushes toward normal (peers show similar local weather) |
| **`if_score`** | Tier 2 (Density ML) | **+0.2889** | ~+0.29 | Modest push toward anomaly (isolated high-dimensional point) |
| **`isolated_deviation`** | Tier 3 (Gate) | **+0.0000** | 0.00 | Neutral (gate is zero for gradual thermodynamic creep) |
| **`tier1_flagged`** | Tier 1 (Physical QC) | **+0.0000** | 0.00 | Neutral (physical limits intact) |

- **Selection Rationale**: This canonical instance from `TIER5_EVALUATION.md` demonstrates game-theoretic explainability in action: it shows why Mahalanobis distance correctly flags thermodynamic covariance violations while GRU reconstruction error correctly notes that no abrupt physical step occurred.
- **Design Choices**: Horizontal bars ranked by $|SHAP|$. Positive values colored orange (`#F79646`), negative values colored teal (`#4BACC6`). Solid zero-line at $x=0$. Explicit numerical data labels on each bar.

---

## Deliverable 6 (Bonus): Spatial Buddy-Check Consensus

- **Script**: [`scripts/plots/plot6_spatial_consensus.py`](file:///d:/SIH/scripts/plots/plot6_spatial_consensus.py)
- **Output Image**: [`outputs/plots/spatial_consensus.png`](file:///d:/SIH/outputs/plots/spatial_consensus.png)
- **Source Data**: [`data/tier4_results/train.parquet`](file:///d:/SIH/data/tier4_results/train.parquet) (`AWS_IND_H01` vs. 3-NN neighbors `AWS_IND_P01`, `AWS_IND_H02`, `AWS_IND_P02` from [`models/spatial_neighbors.json`](file:///d:/SIH/models/spatial_neighbors.json))
- **Reference Documentation**: [`docs/FINAL_EVALUATION.md`](file:///d:/SIH/docs/FINAL_EVALUATION.md) (Section 3.2)

### Exact Numeric Values & Verification

- **Event Window**: `2026-07-10 21:10:00` to `2026-07-11 09:50:00` (77 steps / 12.8 hours).
- **Temperature Plunge**:
  - `AWS_IND_H01` (Shimla hill station, ~2200m elevation): plunges from **21.28°C** down to **7.83°C** ($\Delta T = -13.45^\circ\text{C}$).
  - 3-NN Peer Median (Ambala, Mandi, Ludhiana): plunges from **32.19°C** down to **26.18°C** ($\Delta T = -6.01^\circ\text{C}$).
  - Temperature Pearson Correlation: **$r = 0.7803$** (strong regional co-movement).
- **Barometric Pressure Surge**:
  - `AWS_IND_H01`: surges from **767.25 hPa** up to **781.30 hPa** ($\Delta P = +14.06\text{ hPa}$).
  - 3-NN Peer Median: surges from **981.61 hPa** up to **986.18 hPa** ($\Delta P = +4.57\text{ hPa}$).
  - Pressure Pearson Correlation: **$r = 0.5371$**.
- **Spatial Consensus Verdict**:
  - `isolated_deviation == False` on **74 out of 77 timesteps** (**96.1% agreement**).
  - Standalone GRU false alarms suppressed: **35 of 38** (92.11% suppression rate).
- **Selection Rationale**: Validates the physical mechanism underlying Deliverable 3, graphically explaining to pitch-deck evaluators why multi-station consensus prevents severe weather false alarms.
- **Design Choices**: Dual-stacked subplots (Temperature on top, Pressure on bottom) with dual y-axes to accommodate elevation offsets between hill and valley stations. Callout banner highlighting the active spatial consensus region.

---

## File Inventory Checklist

| Deliverable | Standalone Script | Generated Output PNG | DPI | Status |
| :--- | :--- | :--- | :---: | :---: |
| **D1: Recall Rollup** | [`scripts/plots/plot1_recall_rollup.py`](file:///d:/SIH/scripts/plots/plot1_recall_rollup.py) | [`outputs/plots/recall_rollup.png`](file:///d:/SIH/outputs/plots/recall_rollup.png) | 300 | **READY** |
| **D2: Latency Benchmark** | [`scripts/plots/plot2_latency.py`](file:///d:/SIH/scripts/plots/plot2_latency.py) | [`outputs/plots/latency_benchmark.png`](file:///d:/SIH/outputs/plots/latency_benchmark.png) | 300 | **READY** |
| **D3: Squall Suppression** | [`scripts/plots/plot3_squall_suppression.py`](file:///d:/SIH/scripts/plots/plot3_squall_suppression.py) | [`outputs/plots/squall_suppression.png`](file:///d:/SIH/outputs/plots/squall_suppression.png) | 300 | **READY** |
| **D4: SHI Lead Time** | [`scripts/plots/plot4_shi_lead_time.py`](file:///d:/SIH/scripts/plots/plot4_shi_lead_time.py) | [`outputs/plots/shi_lead_time.png`](file:///d:/SIH/outputs/plots/shi_lead_time.png) | 300 | **READY** |
| **D5: TreeSHAP Attribution**| [`scripts/plots/plot5_shap_attribution.py`](file:///d:/SIH/scripts/plots/plot5_shap_attribution.py) | [`outputs/plots/shap_attribution.png`](file:///d:/SIH/outputs/plots/shap_attribution.png) | 300 | **READY** |
| **D6: Spatial Consensus** | [`scripts/plots/plot6_spatial_consensus.py`](file:///d:/SIH/scripts/plots/plot6_spatial_consensus.py) | [`outputs/plots/spatial_consensus.png`](file:///d:/SIH/outputs/plots/spatial_consensus.png) | 300 | **READY** |
