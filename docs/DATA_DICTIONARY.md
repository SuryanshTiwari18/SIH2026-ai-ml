# SkyGuard AI: Canonical Telemetry Contract & Data Dictionary

**Smart India Hackathon 2026 | Problem Statement 26073**  
**AI/ML-Based Intelligent Anomaly Detection for Automatic Weather Stations (IMD/MoES)**

---

## 1. Schema Definition

The generated dataset follows the **Canonical Telemetry Contract**, standardizing multi-station weather station observations for machine learning pipelines.

| Column Name | Data Type | Nullable | Units | Expected Bounds | Description |
|---|---|---|---|---|---|
| `station_id` | `string` | No | - | E.g. `AWS_IND_C01` | Unique station code |
| `timestamp` | `timestamp[ns]` | No | ISO-8601 (IST) | `2026-06-01` onward | Observation timestamp at 10-minute cadence |
| `latitude` | `float64` | No | Decimal Degrees | `8.0` to `36.0` | Station geographic latitude |
| `longitude` | `float64` | No | Decimal Degrees | `68.0` to `97.0` | Station geographic longitude |
| `temperature_c` | `float64` | Yes (on dropout) | °Celsius | `-20.0` to `55.0` | Surface dry-bulb air temperature |
| `pressure_hpa` | `float64` | Yes (on dropout) | hPa | `550.0` to `1060.0` | Station surface barometric pressure |
| `humidity_pct` | `float64` | Yes (on dropout) | % | `0.0` to `100.0` | Relative humidity percentage |
| `is_anomaly` | `boolean` | No | - | `True` / `False` | Ground-truth binary fault label |
| `anomaly_type` | `string` | Yes (null if normal) | - | Categorical (7 types) | Specific sensor fault category |
| `anomaly_severity` | `float64` | Yes (null if normal) | Score [0, 1] | `0.1` to `1.0` | Normalized fault severity score |

---

## 2. Ground-Truth Anomaly Taxonomy (PS 26073)

| `anomaly_type` | Physical Hardware / Network Cause | Typical Duration | Characteristics & Signature |
|---|---|---|---|
| `spike_or_drop` | Analog-to-Digital Converter (ADC) glitch, impulse electromagnetic surge | 10–30 min (1–3 steps) | Discontinuous jump ($\Delta T > \pm 12^\circ\text{C}$, $\Delta P > \pm 25\text{ hPa}$, $\Delta RH > \pm 40\%$) |
| `frozen_sensor` | Firmware deadlock, sensor signal bus lockup | 3–24 hours (18–144 steps) | Zero variance: exact floating-point value repeated consecutively |
| `communication_dropout` | Cellular/GPRS telemetry drop, battery exhaustion, antenna disconnect | 30 min – 6 hours (3–36 steps) | Missing sensor readings (`NaN` across one or all parameters) |
| `calibration_drift` | Transducer aging, dust accumulation, hygroscopic degradation | 2–6 days (288–864 steps) | Accumulating monotonic bias ramp without sudden discontinuities |
| `power_fluctuation_glitch` | Solar charge controller failure, low battery ripple voltage | 40 min – 2.5 hours (4–15 steps) | High-frequency jitter burst ($\sigma \times 15-25$), extreme noise variance |
| `data_corruption` | Packet parity bitflip, memory buffer corruption | 10–40 min (1–4 steps) | Nonsensical/out-of-range sentinel tokens (`-999.0`, `9999.0`, `RH = -25%`) |
| `cross_sensor_inconsistency` | Decoupled sensor failure, humidity capsule contamination | 1–6 hours (6–36 steps) | Thermodynamic violation: $T_{\text{dew}} > T_{\text{air}}$ or impossible $P/T$ pair |

---

## 3. Genuine Extreme Weather Distinction (`is_anomaly = False`)

A primary innovation of this generator is preventing **false positives** on genuine atmospheric events:
- **Heatwaves**: Sustained synoptic high temperature (+4.5°C to +8°C above baseline) coupled with depressed RH (down to 12–25%). Labeled `is_anomaly = False`.
- **Convective Storm / Monsoon Squall**: Sudden sharp barometric drop (-7 to -14 hPa), cold downdraft rain cooling (-6°C to -10°C in under 45 minutes), and RH saturation surge (96–100%). Labeled `is_anomaly = False`.
- **Temperature Inversion & Radiation Fog**: Prolonged saturated RH (>98%), suppressed diurnal range (< 2.5°C), cold baseline, high stable pressure. Labeled `is_anomaly = False`.

---

## 4. File Structure & Splits

```
data/
├── raw/
│   ├── aws_telemetry_master.parquet       # Complete dataset across all stations
│   ├── aws_telemetry_master.csv           # Full CSV export for inspection
│   └── generation_metadata.json           # Audit trail of parameters & split timestamps
├── splits/
│   ├── train.parquet                      # Chronological Train Split (70%)
│   ├── val.parquet                        # Chronological Validation Split (15%)
│   ├── test.parquet                       # Chronological Test Split (15%)
│   ├── test.csv                           # Test split in CSV format
│   └── spatial_holdout_stations.parquet   # 3 entirely held-out stations for spatial testing
└── plots/
    ├── climate_zones_comparison.png       # Diurnal cycles across Coastal, Arid, Hill, Plains
    ├── extreme_weather_cases.png          # Severe weather vs anomaly separation
    └── sensor_fault_taxonomy.png          # Visual gallery of all 7 fault types
```
