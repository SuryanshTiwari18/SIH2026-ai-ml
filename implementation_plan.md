# Implementation Plan: Physics-Informed Synthetic AWS Sensor Data Generator for SkyGuard AI

## Overview
SkyGuard AI requires a physics-informed synthetic data generator for Automatic Weather Stations (AWS) to train and benchmark a 5-tier anomaly detection pipeline (Physical QC → Isolation Forest + GRU-Autoencoder → Mahalanobis/Clausius-Clapeyron consistency + spatial buddy-check → Fusion Classifier → Sensor Health Index + TreeSHAP). The generator will model three critical meteorological parameters:
1. **Air Temperature (°C)**
2. **Atmospheric Surface Pressure (hPa)**
3. **Relative Humidity (%)**

The dataset will feature realistic multi-station dynamics across 4 Indian climate zones, genuine extreme weather events (labeled as normal, `is_anomaly=False`), and a comprehensive suite of 7 realistic sensor fault types (labeled with `anomaly_type` and `anomaly_severity`).

---

## User Review Required

> [!IMPORTANT]
> - **Default Sampling & Duration**: Default configuration will generate **16 stations** at **10-minute reporting intervals** (144 readings/day/station) over **60 days** (~138,240 records total), producing an optimal balance between statistical depth, seasonal/diurnal variability, and lightweight file sizes (< 15 MB Parquet). This is fully configurable via CLI or YAML/JSON.
> - **Station Holdout vs Chronological Split**: Both temporal splits (70% train, 15% val, 15% test) and spatial splits (e.g. 3 held-out stations for cross-station generalization testing) will be provided out of the box.

---

## Proposed Architecture & Component Design

```
d:/SIH/
├── configs/
│   └── default_config.yaml            # Configurable parameters (stations, dates, anomaly rates, seed)
├── data/                              # Generated output datasets
│   ├── raw/                           # Full generated dataset (parquet + csv)
│   ├── splits/                        # Chronological & spatial splits (train/val/test/spatial_holdout)
│   └── plots/                         # Validation and sanity-check time series plots
├── docs/
│   ├── DATA_DICTIONARY.md             # Canonical Telemetry Contract schema & specifications
│   └── METEOROLOGICAL_MODELING.md     # Equations & physical modeling formulation
├── src/
│   ├── __init__.py
│   ├── config.py                      # Dataclass / YAML config parser
│   ├── stations.py                    # 16 Indian AWS stations with realistic lat, lon, elevation, normals
│   ├── physics.py                     # Barometric formula, August-Roche-Magnus dew point & Clausius-Clapeyron
│   ├── base_signals.py                # Diurnal cycles, semi-diurnal tides, seasonal shift, AR(1) synoptic drift
│   ├── extreme_weather.py             # Genuine extreme weather injection (heatwaves, squalls/depressions, inversions)
│   ├── anomaly_injector.py            # 7 sensor fault types (spikes, frozen, dropout, drift, jitter, corrupt, cross-inconsistent)
│   └── pipeline.py                    # Master dataset synthesizer, splitting, and metadata export
├── generate.py                        # CLI entrypoint for running the generator
├── validate.py                        # Automated physical & statistical validation suite
├── visualize.py                       # Plotting script for multi-station anomaly time series inspection
└── README.md                          # Comprehensive user & developer guide
```

---

## Technical Specifications

### 1. Indian AWS Stations & Climate Zones (`src/stations.py`)
16 real Indian AWS observatories grounded in IMD climatological normals:
- **Coastal / Maritime** (Low diurnal temperature range 4–8°C, high base humidity 65–85%, sea-level pressure ~1010–1013 hPa):
  - `AWS_IND_C01`: Mumbai (Colaba) — $18.90^\circ\text{N}, 72.81^\circ\text{E}$, 11m ASL
  - `AWS_IND_C02`: Chennai (Meenambakkam) — $13.00^\circ\text{N}, 80.18^\circ\text{E}$, 16m ASL
  - `AWS_IND_C03`: Kochi (Willingdon Island) — $9.96^\circ\text{N}, 76.27^\circ\text{E}$, 3m ASL
  - `AWS_IND_C04`: Puri (Odisha Coast) — $19.81^\circ\text{N}, 85.83^\circ\text{E}$, 9m ASL
- **Arid / Semi-Arid** (High diurnal temperature range 14–18°C, low base humidity 20–45%, dry heat):
  - `AWS_IND_A01`: Jodhpur (Thar Desert) — $26.26^\circ\text{N}, 73.05^\circ\text{E}$, 224m ASL
  - `AWS_IND_A02`: Jaisalmer — $26.91^\circ\text{N}, 70.90^\circ\text{E}$, 225m ASL
  - `AWS_IND_A03`: Bikaner — $28.02^\circ\text{N}, 73.31^\circ\text{E}$, 242m ASL
  - `AWS_IND_A04`: Rajkot (Saurashtra) — $22.30^\circ\text{N}, 70.80^\circ\text{E}$, 138m ASL
- **Hill / Montane** (Reduced baseline pressure via barometric formula ~780–850 hPa, cooler temperatures, orographic effects):
  - `AWS_IND_H01`: Shimla (Himachal Pradesh) — $31.10^\circ\text{N}, 77.17^\circ\text{E}$, 2205m ASL (~776 hPa baseline)
  - `AWS_IND_H02`: Srinagar (Kashmir Valley) — $34.08^\circ\text{N}, 74.80^\circ\text{E}$, 1587m ASL (~836 hPa baseline)
  - `AWS_IND_H03`: Shillong (Meghalaya) — $25.57^\circ\text{N}, 91.89^\circ\text{E}$, 1496m ASL (~845 hPa baseline)
  - `AWS_IND_H04`: Ooty (Nilgiris) — $11.41^\circ\text{N}, 76.70^\circ\text{E}$, 2240m ASL (~773 hPa baseline)
- **Gangetic Plains / Continental** (High seasonal swings, moderate-high DTR 10–14°C, monsoon shifts):
  - `AWS_IND_P01`: New Delhi (Safdarjung) — $28.58^\circ\text{N}, 77.21^\circ\text{E}$, 211m ASL (~989 hPa baseline)
  - `AWS_IND_P02`: Lucknow (Amausi) — $26.76^\circ\text{N}, 80.88^\circ\text{E}$, 128m ASL (~998 hPa baseline)
  - `AWS_IND_P03`: Patna — $25.60^\circ\text{N}, 85.10^\circ\text{E}$, 53m ASL (~1007 hPa baseline)
  - `AWS_IND_P04`: Nagpur (Central India) — $21.15^\circ\text{N}, 79.08^\circ\text{E}$, 310m ASL (~978 hPa baseline)

### 2. Meteorological & Physical Formulation (`src/physics.py`, `src/base_signals.py`)
- **Barometric Elevation Offset**:
  $$P_{\text{base}}(h) = P_0 \left(1 - \frac{L \cdot h}{T_0}\right)^{\frac{g M}{R_0 L}}$$
- **Atmospheric Thermal Tides**: Tropical/subtropical semi-diurnal pressure oscillation ($\Delta P \approx 1.5 \sin(2\pi (t - 4)/12)$ hPa with peaks at ~10:00 and 22:00 local solar time).
- **Diurnal Solar Cycle**:
  - $T(t)$ modelled with modified asymmetric diurnal curve (trough at ~05:30 sunrise, peak at ~14:30).
  - $RH(t)$ modelled via thermal vapor saturation response, reaching minimum when $T$ peaks and maximum at dawn.
- **Synoptic Weather Component**:
  - Continuous autoregressive AR(1) process with decay parameter $\phi \approx 0.98$ (correlation scale ~2–4 days) simulating passing depressions/ridges.
  - Temperature, pressure, and moisture are coupled (low pressure corresponds to cloudy, humid conditions with compressed DTR).
- **August-Roche-Magnus Vapor Pressure & Dew Point**:
  $$e_s(T) = 6.112 \exp\left(\frac{17.67 T}{T + 243.5}\right)\text{ hPa}, \quad e = e_s(T) \cdot \frac{RH}{100}$$
  $$T_d = \frac{243.5 \cdot \ln(e / 6.112)}{17.67 - \ln(e / 6.112)}$$
  Enforcing $T_d \le T$ and $0\% < RH \le 100\%$ unconditionally in normal weather.

### 3. Genuine Extreme Weather Events (`src/extreme_weather.py`)
Labeled as **`is_anomaly = False`**, testing the ML pipeline's false-positive rejection:
1. **Heatwave**: +4°C to +8°C temperature anomaly for 3–5 consecutive days, depressed RH down to 12–20%, slight barometric thinning.
2. **Monsoon Low-Pressure Squall / Convective Storm**: Rapid barometric drop (-8 to -15 hPa), sudden rain cooling (-5°C to -10°C in under 40 mins due to downdrafts), RH saturation surging to 95–100%.
3. **Winter Fog / Temperature Inversion**: Prolonged RH near 98–100%, suppressed diurnal temperature range (< 3°C), cold baseline, high stable barometric pressure.

### 4. Sensor Fault & Anomaly Injection (`src/anomaly_injector.py`)
Labeled as **`is_anomaly = True`**, with exact `anomaly_type` and `anomaly_severity` (0.0 to 1.0):
1. `spike_or_drop`: Sudden 1–3 sample discontinuous jump (e.g. +18°C or -50 hPa).
2. `frozen_sensor`: Sensor ADC stuck; 0-variance repeated values over 2–24 hours.
3. `communication_dropout`: Data packet loss resulting in `NaN` readings for 20m to 6h.
4. `calibration_drift`: Subtle accumulated linear/exponential ramp ($+0.15^\circ\text{C/day}$ or $-0.4\text{ hPa/day}$) over 7–14 days.
5. `power_fluctuation_glitch`: Degraded battery/solar voltage causing massive high-frequency jitter/noise bursts ($\sigma \times 10$) for 30–90 min.
6. `data_corruption`: Nonsensical / out-of-range sensor readings (e.g. -999.0, 9999.0, RH = -15% or 150%).
7. `cross_sensor_inconsistency`: Subtly invalid multi-sensor combinations (e.g., $RH = 98\%$ during a 45°C afternoon, or $T_d > T_{\text{air}}$ violating Clausius-Clapeyron, while each univariate value is within individual historical ranges).
- **Spatial Locality Guarantee**: Faults are injected station-by-station. Neighboring stations remain unaffected, providing ground truth for spatial buddy checks.

### 5. Canonical Telemetry Contract Schema (`docs/DATA_DICTIONARY.md`)
| Column | Type | Description |
|---|---|---|
| `station_id` | string | Unique station identifier (e.g., `AWS_IND_C01`) |
| `timestamp` | timestamp (UTC/IST) | ISO-8601 observation timestamp at 10-minute cadence |
| `latitude` | float64 | Latitude in decimal degrees |
| `longitude` | float64 | Longitude in decimal degrees |
| `temperature_c` | float64 | Dry-bulb air temperature in degrees Celsius (can be NaN on dropout) |
| `pressure_hpa` | float64 | Station surface atmospheric pressure in hPa (can be NaN on dropout) |
| `humidity_pct` | float64 | Relative humidity percentage [0, 100] (can be NaN on dropout) |
| `is_anomaly` | boolean | Binary ground truth label (True if faulty, False if normal or extreme weather) |
| `anomaly_type` | string / null | Specific fault category (or null if normal) |
| `anomaly_severity` | float64 / null | Normalized fault severity score in range (0.0, 1.0] |

---

## Verification Plan

### Automated Tests (`validate.py`)
1. **Physical Consistency Checks**:
   - Verify normal rows have $0 \le RH \le 100\%$ and calculated $T_d \le T$.
   - Verify barometric altitudes scale monotonically across sea-level vs hill stations.
2. **Fault Label Fidelity**:
   - Verify `frozen_sensor` sequences have standard deviation = 0.0.
   - Verify `communication_dropout` records contain `NaN`.
   - Verify `cross_sensor_inconsistency` violations are mathematically detected.
3. **Data Leakage & Split Verification**:
   - Verify $\max(\text{train\_time}) < \min(\text{val\_time}) < \min(\text{test\_time})$.
   - Verify zero station overlap for spatial holdouts.
4. **Reproducibility**:
   - Verify identical output hashes across two runs with identical random seeds.

### Visual Sanity Checks (`visualize.py`)
- Multi-panel publication-grade plots saved in `data/plots/`:
  1. Diurnal & synoptic multi-day trace comparing Coastal vs Hill vs Arid station.
  2. Extreme weather trace showing storm pressure drop & rain cooling labeled as normal.
  3. Zoomed-in panels highlighting each of the 7 sensor fault types with distinct color overlays.
