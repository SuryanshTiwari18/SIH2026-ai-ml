# SkyGuard AI: Physics-Informed Synthetic AWS Sensor Data Generator

**Smart India Hackathon 2026 | Problem Statement 26073**  
*AI/ML-Based Intelligent Anomaly Detection for Automatic Weather Stations (IMD/MoES)*  
*Theme: Disaster Management*

---

## Overview

**SkyGuard AI** implements a 5-tier machine learning pipeline:
1. **Physical QC**: Physical plausibility, step-change limits, and thermodynamic bounds.
2. **Univariate Anomaly Detection**: Isolation Forest + GRU-Autoencoder.
3. **Multivariate & Spatial QC**: Mahalanobis distance, Clausius-Clapeyron consistency, and spatial buddy-check.
4. **Fusion Classifier**: Ensemble voting / gradient boosting meta-classifier.
5. **Explainability & Health Index**: Sensor Health Index (SHI) + TreeSHAP feature attributions.

This repository provides the **Physics-Informed Synthetic AWS Sensor Data Generator** that produces multi-station time series for **Temperature (°C)**, **Atmospheric Pressure (hPa)**, and **Relative Humidity (%)**.

---

## Key Features

1. **Multi-Station Indian Network**:
   - 16 real Indian observatories grounded in IMD climatological normals across 4 climate zones: **Coastal**, **Arid/Desert**, **Hill/Montane**, and **Gangetic Plains**.
   - Elevation-derived barometric baselines using the International Standard Atmosphere (ISA) formula.

2. **Rigorous Physics-Informed Signals**:
   - Asymmetric solar diurnal heating curve (trough at ~05:30 IST, peak at ~14:30 IST).
   - Semi-diurnal atmospheric thermal solar tides ($S_2(P)$ ~12-hour cycle with peaks at 10:00 & 22:00).
   - Inversely correlated diurnal relative humidity dynamics.
   - Coupled AR(1) synoptic drift (passing depressions, ridges, moisture influx).
   - August-Roche-Magnus approximation enforcing $T_{\text{dew}} \le T_{\text{air}}$ unconditionally in normal conditions.

3. **Genuine Extreme Weather (Ground Truth: `is_anomaly = False`)**:
   - Heatwaves (sustained high temperature + depressed RH).
   - Convective storms / Monsoon squalls (rapid barometric drop, cold downdraft rain cooling, saturated 99% RH).
   - Radiation fog & inversions (saturated RH, suppressed diurnal range, high pressure).
   - *Prevents candidate models from raising false alarms on real severe weather.*

4. **Realistic 7-Fault Sensor Taxonomy (Ground Truth: `is_anomaly = True`)**:
   - `spike_or_drop`: ADC impulses and discontinuous steps.
   - `frozen_sensor`: Sensor ADC stuck at exact constant float (zero variance).
   - `communication_dropout`: Telemetry packet loss resulting in `NaN` values.
   - `calibration_drift`: Slow accumulating linear bias over days to weeks.
   - `power_fluctuation_glitch`: Degraded battery voltage ripple and high-frequency jitter.
   - `data_corruption`: Nonsensical / out-of-range sentinel tokens (`-999.0`, `9999.0`).
   - `cross_sensor_inconsistency`: Subtly invalid thermodynamic combinations ($T_{\text{dew}} > T_{\text{air}}$).
   - **Station-Local Guarantee**: Neighboring stations remain unaffected to enable spatial buddy-check validation.

5. **Chronological & Spatial Splits**:
   - Clean temporal splits: Train (70%), Validation (15%), Test (15%) with zero time leakage.
   - Spatial holdout split: 3 stations completely withheld for zero-shot spatial testing.

---

## Quick Start

### 1. Installation

```bash
# Activate virtual environment
.venv\Scripts\activate

# Install dependencies (if not already installed)
pip install -r requirements.txt
```

### 2. Generate Dataset

```bash
# Run with default configuration (60 days, 16 stations, 10-min cadence, 3.5% anomaly rate)
python generate.py --plot

# Or customize duration and anomaly rate via CLI
python generate.py --days 30 --anomaly-rate 0.03 --seed 42 --plot
```

### 3. Run Validation & Verification Suite

```bash
python validate.py
```

### 4. Inspect Visual Sanity-Check Plots

The plots are generated in `data/plots/`:
- `data/plots/climate_zones_comparison.png`: Diurnal temperature, pressure, and humidity across climate zones.
- `data/plots/extreme_weather_cases.png`: Severe weather dynamics showing physical coupling and normal labels.
- `data/plots/sensor_fault_taxonomy.png`: High-resolution gallery illustrating all 7 fault types with ground-truth overlays.

---

## Configuration Reference (`configs/default_config.yaml`)

```yaml
generation:
  start_date: "2026-06-01 00:00:00"  # Start of monsoon season
  duration_days: 60                 # Days of continuous reporting
  sampling_interval_minutes: 10     # Standard AWS cadence (144 readings/day)
  random_seed: 42                   # Deterministic reproducibility

stations:
  include_climate_zones: [coastal, arid, hill, plains]
  spatial_holdout_station_ids: [AWS_IND_C04, AWS_IND_A03, AWS_IND_H03]

extreme_weather:
  enabled: true
  event_probability_per_station: 0.35
  event_types: [heatwave, convective_storm_squall, temperature_inversion_fog]

anomalies:
  enabled: true
  target_anomaly_rate: 0.035
  station_local: true
  fault_type_weights:
    spike_or_drop: 0.25
    frozen_sensor: 0.18
    communication_dropout: 0.15
    calibration_drift: 0.12
    power_fluctuation_glitch: 0.12
    data_corruption: 0.08
    cross_sensor_inconsistency: 0.10

splits:
  temporal:
    train_ratio: 0.70
    val_ratio: 0.15
    test_ratio: 0.15
```

---

## Output Contract & Schema

The dataset is saved to `data/raw/aws_telemetry_master.parquet` (and `.csv`):

```
station_id, timestamp, latitude, longitude, temperature_c, pressure_hpa, humidity_pct, is_anomaly, anomaly_type, anomaly_severity
```

See [docs/DATA_DICTIONARY.md](file:///d:/SIH/docs/DATA_DICTIONARY.md) for full column descriptions and [docs/METEOROLOGICAL_MODELING.md](file:///d:/SIH/docs/METEOROLOGICAL_MODELING.md) for the mathematical formulations.
