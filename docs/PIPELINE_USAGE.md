# SkyGuard AI: Operational Pipeline Usage Manual
**SIH 2026 — Problem Statement 26073**  
*Automated Quality Control & Diagnostic Anomaly Detection for AWS Surface Telemetry*

---

## 1. Executive Overview

`src/skyguard_pipeline.py` provides the production-grade, deployable `SkyGuardPipeline` class for automated telemetry quality control (QC), temporal and spatial fault detection, explainable root-cause attribution, and predictive maintenance.

The pipeline integrates all five SkyGuard AI architectural tiers into a single, callable, low-latency engine designed for real-time Automated Weather Station (AWS) ingestion:
- **Tier 1: Physical QC** (deterministic range, sentinel, and flatline screening).
- **Tier 2: Temporal ML** (Isolation Forest rate-of-change + GRU-Autoencoder reconstruction).
- **Tier 3: Multivariate & Spatial Consensus** (Local/Zone Mahalanobis distance + 3-NN Haversine buddy check).
- **Tier 4: Two-Track Fusion Engine** (Track 1 Gated Hard Rule for immediate operational alerts + Track 2 Learned LightGBM Meta-Classifier for maintenance queuing).
- **Tier 5: TreeSHAP Explainability & Predictive Maintenance** (local feature attribution rationales, EMA Sensor Health Index, and 3-NN buddy-median value reconstruction).

---

## 2. Installation & Quickstart

### 2.1 Dependencies
Ensure the environment contains Python 3.10+ and the required dependencies:
```bash
pip install torch lightgbm shap scikit-learn pandas numpy joblib
```

### 2.2 Minimal Python Invocation

```python
from src.skyguard_pipeline import SkyGuardPipeline

# 1. Initialize pipeline once at application startup (loads all models into RAM)
pipeline = SkyGuardPipeline()

# 2. Ingest an incoming telemetry record
raw_telemetry = {
    "station_id": "AWS_IND_C02",
    "timestamp": "2026-07-22 01:20:00",
    "temperature_c": 26.10,
    "pressure_hpa": 1001.94,
    "humidity_pct": 95.99
}

# 3. Process the row
decision = pipeline.process_row(raw_telemetry)

# 4. Inspect diagnostic output
print(f"Alert Status: {decision['is_anomaly']}")
print(f"Fault Category: {decision['predicted_type']}")
print(f"Routing Track: Track {decision['which_track']}")
print(f"Rationale: {decision['shap_rationale_text']}")
print(f"Sensor Health: {decision['sensor_health_index']}/100.0")
```

---

## 3. Architecture & Two-Track Routing Logic

To solve the dual requirements of operational meteorological systems—**zero false alarms during genuine extreme weather** (e.g. convective squalls) while maintaining **high sensitivity to silent hardware degradations** (calibration drift and frozen sensors)—SkyGuard AI deploys a **Two-Track Routing Architecture**:

```
                              Incoming Raw Telemetry Row
                                          │
                                          ▼
                             [ Tier 1: Physical QC ] ────────────────┐
                             (Sentinels, Limits, Comm)               │ Fail (Sentinel/Dropout)
                                          │                          ▼
                                      Pass QC               [ Track 1 Emergency Alert ]
                                          │                 (Confidence: 1.0, Latency: 0.1µs)
                                          ▼
                      [ Tiers 2 & 3: Multi-Model Inference ]
                      - Isolation Forest + GRU-Autoencoder
                      - Mahalanobis Distance + 3-NN Buddy
                                          │
                                          ▼
                      [ Spatial Consensus Isolation Gate ]
                        (peer_diverged & own_delta_large)
                                          │
                     ┌────────────────────┴────────────────────┐
                     ▼                                         ▼
            isolated_deviation == True               isolated_deviation == False
                     │                                         │
        [ Track 1: Gated Hard Rule ]                  [ Track 2: Learned Classifier ]
        - Immediate Operational Alert                 - Asynchronous Maintenance Queue
        - Target Types:                               - Target Types:
          * data_corruption                             * frozen_sensor
          * communication_dropout                       * calibration_drift
          * spike_or_drop                               * cross_sensor_inconsistency
          * power_fluctuation_glitch                  - Weather Squall False Alarms Suppressed
```

### Track 1: Operational Emergency Alert (Real-Time Synchronous)
- **Purpose**: Urgent alerts sent to meteorologists and automated forecasting ingestion pipelines.
- **Criteria**: Tier 1 physical failures OR (Tier 2/3 flags verified by `isolated_deviation == True`).
- **Target Types**: `data_corruption`, `communication_dropout`, `spike_or_drop`, `power_fluctuation_glitch`.
- **False Positive Guarantee**: Extreme convective weather squalls are mathematically suppressed because regional neighbor stations agree with rapid atmospheric pressure/temperature shifts (`isolated_deviation == False`).

### Track 2: Maintenance Queue (Asynchronous Diagnostic Review)
- **Purpose**: Secondary triage queue reviewed by field instrumentation technicians.
- **Criteria**: Low 10-minute rate-of-change faults that cannot pass the spatial rate-of-change gate.
- **Target Types**: `frozen_sensor` (zero delta), `calibration_drift` (slow multi-day creep), `cross_sensor_inconsistency` (thermodynamic psychrometric mismatch).

---

## 4. Input & Output Specifications

### 4.1 Input Schema (`process_row`)
The `process_row` method expects a dictionary with the following keys:

| Field Name | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `station_id` | `str` | Yes | Unique AWS station identifier (e.g. `'AWS_IND_C02'`). |
| `timestamp` | `str` or `pd.Timestamp` | Yes | Telemetry observation time (ISO format or Timestamp). |
| `temperature_c` | `float` or `None` | Yes | Dry-bulb ambient air temperature in degrees Celsius (°C). |
| `pressure_hpa` | `float` or `None` | Yes | Station barometric surface pressure in hectopascals (hPa). |
| `humidity_pct` | `float` or `None` | Yes | Relative humidity percentage (0.0 to 100.0%). |

*Optional argument*: `concurrent_neighbors` (`Dict[str, dict]`, optional):
Dictionary mapping neighbor station IDs to their current observation dictionaries `{'t': float, 'p': float, 'rh': float}`. If omitted, the pipeline automatically looks up the latest known readings from internal network cache (`self.network_latest`).

### 4.2 Output Payload Schema
Every invocation of `process_row` returns a comprehensive diagnostic dictionary:

```json
{
  "station_id": "AWS_IND_H04",
  "timestamp": "2026-07-26 02:20:00",
  "is_anomaly": true,
  "predicted_type": "power_fluctuation_glitch",
  "confidence": 0.95,
  "which_track": "1",
  "shap_rationale_text": "Flagged primarily due to elevated Isolation Forest score (IF=0.420, SHAP: +2.72) and GRU error (GRU=0.15), indicating transient hardware spike/glitch.",
  "sensor_health_index": 98.8,
  "corrected_value_suggestion": {
    "temperature_c": 26.52,
    "pressure_hpa": 1006.20,
    "humidity_pct": 91.84
  },
  "tier_signals": {
    "tier1_flagged": false,
    "tier1_reason": null,
    "if_score": 0.4201,
    "gru_score": 0.1512,
    "mahalanobis_dist": 2.45,
    "buddy_flagged": true,
    "isolated_deviation": true
  },
  "diagnostics": {
    "latency_ms": 26.69,
    "history_length": 31,
    "neighbor_fallback_used": false
  }
}
```

#### Field Explanations:
1. **`is_anomaly`** (`bool`): `True` if flagged as an operational alert (Track 1) or confirmed maintenance fault (Track 2); `False` if normal or suppressed.
2. **`predicted_type`** (`str`): Categorical classification:
   - `'normal'`
   - `'data_corruption'`
   - `'communication_dropout'`
   - `'spike_or_drop'`
   - `'power_fluctuation_glitch'`
   - `'frozen_sensor'`
   - `'calibration_drift'`
   - `'cross_sensor_inconsistency'`
   - `'insufficient_context'` (cold-start rows)
3. **`confidence`** (`float`): Model certainty (1.0 for deterministic physical QC; calibrated class probability for learned models).
4. **`which_track`** (`str`): Routing destination: `'1'` (Operational Alert), `'2'` (Maintenance Queue), or `'insufficient_context'`.
5. **`shap_rationale_text`** (`str`): Natural language diagnostic summary generated via TreeSHAP feature importance ranking.
6. **`sensor_health_index`** (`float`): 0.0 to 100.0 health metric tracking long-term degradation via an Exponential Moving Average (EMA).
7. **`corrected_value_suggestion`** (`dict` or `None`): Spatial 3-NN buddy median replacement values `{'temperature_c', 'pressure_hpa', 'humidity_pct'}` provided whenever a station reading is anomalous and neighbor stations are available.
8. **`tier_signals`** (`dict`): Raw underlying model outputs across Tiers 1 through 3 for operational auditing.
9. **`diagnostics`** (`dict`): Per-row execution latency in milliseconds, current rolling history buffer depth, and whether spatial fallback mode was engaged.

---

## 5. Edge Cases & Robustness Handlers

### 5.1 Cold Start (< 24 History Steps)
- **Problem**: Higher-order temporal models (GRU-Autoencoder 6-step sequences, 1-hour/6-hour rolling volatility ratios) require historical context.
- **Handling**: 
  - For the first 23 steps following station deployment or reboot, Tier 1 Physical QC remains 100% active.
  - If a Tier 1 physical violation occurs, it is flagged immediately (`which_track='1'`, confidence 1.0).
  - If Tier 1 passes, the row is safely passed through with:
    - `predicted_type = "insufficient_context"`
    - `is_anomaly = False`
    - `which_track = "insufficient_context"`
    - `shap_rationale_text = "Warmup window lookback buffer (<24 steps); insufficient context for higher tiers."`
  - Telemetry is buffered until the 24-step threshold is attained, at which point Tiers 2–5 automatically activate.

### 5.2 Missing Concurrent Neighbor Telemetry (Isolated Stations)
- **Problem**: In sparse networks or during telecommunication blackouts, neighboring AWS stations may not report concurrently.
- **Handling**:
  - The pipeline checks `concurrent_neighbors` (if passed) followed by `self.network_latest`.
  - If no neighbors have reported within the active window, the pipeline engages the **Single-Station Spatial Fallback Mode**:
    - `delta_T_buddy = 0.0`
    - `delta_P_cluster = 0.0`
    - `fallback_used = True` (recorded in `diagnostics['neighbor_fallback_used']`)
  - The pipeline gracefully continues inference using temporal models (Isolation Forest + GRU-AE) and single-station physical Mahalanobis checks without throwing exceptions or blocking telemetry.

### 5.3 Climate-Zone Fallback for Unseen Stations
- **Problem**: When newly commissioned stations not present in the training set report telemetry, station-specific Mahalanobis mean/covariance matrices do not exist.
- **Handling**:
  - The pipeline automatically resolves the station's geographic coordinates to its regional climate zone (`"plains"`, `"hills"`, `"coastal"`, `"arid"`).
  - Applies the pre-computed climate-zone hourly background covariance matrix (`zone_hour`), preventing false-positive spikes on holdout stations.

---

## 6. Batch Processing & Utility APIs

### 6.1 Processing Batches (`process_batch`)
To process a DataFrame or a list of dictionaries sequentially:

```python
import pandas as pd
from src.skyguard_pipeline import SkyGuardPipeline

pipeline = SkyGuardPipeline()
df_incoming = pd.read_parquet("data/splits/test.parquet").head(100)

results_df = pipeline.process_batch(df_incoming)
print(results_df[["station_id", "timestamp", "is_anomaly", "predicted_type", "which_track"]])
```

### 6.2 Clearing Station State (`reset_state`)
To reset the rolling window history, network cache, or Sensor Health Index (e.g. after a physical sensor replacement or recalibration):

```python
# Reset a specific station's buffer and health index
pipeline.reset_state(station_id="AWS_IND_C02")

# Reset all stations across the entire network
pipeline.reset_state()
```

### 6.3 Real-Time Latency Benchmarking (`benchmark_latency`)
Measure empirical performance against the operational <500ms SLA:

```python
sample_rows = df_incoming.to_dict(orient="records")
metrics = pipeline.benchmark_latency(sample_rows, n_runs=1)

print(f"Mean Latency: {metrics['mean_ms']:.2f} ms")
print(f"P95 Latency:  {metrics['p95_ms']:.2f} ms")
print(f"Throughput:   {metrics['throughput_rows_sec']:.1f} rows/sec")
print(f"Target Met (<500ms): {metrics['target_met']}")
```

---

## 7. Example Test Script

A full demonstration script exercising all 9 representative meteorological scenarios (normal, 7 fault types, and genuine extreme convective storm squall) is available at:
```bash
python examples/run_pipeline_demo.py
```
