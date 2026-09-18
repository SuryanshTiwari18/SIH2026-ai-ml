# SkyGuard AI: Physics-Informed AWS Sensor Data Generator Walkthrough

**Smart India Hackathon 2026 | Problem Statement 26073**  
*AI/ML-Based Intelligent Anomaly Detection for Automatic Weather Stations (IMD/MoES)*  
*Theme: Disaster Management*

---

## 1. Executive Summary

We have built, verified, and benchmarked a configurable, physics-informed synthetic Automatic Weather Station (AWS) sensor data generator tailored for **SkyGuard AI**. The generator synthesizes realistic, high-fidelity multivariate telemetry for:
- **Temperature (°C)**
- **Atmospheric Pressure (hPa)**
- **Relative Humidity (%)**

The dataset strictly avoids synthetic faker abstractions and instead implements real atmospheric thermodynamic equations (ISA barometric formula, August-Roche-Magnus approximation, $S_2(P)$ semi-diurnal thermal tides, and coupled synoptic AR(1) processes).

---

## 2. Multi-Station Network & Climate Zones

Ground truth was calibrated using India Meteorological Department (IMD) climatological normals across 16 real observatories spanning 4 major climate zones:

| Climate Zone | Stations | Lat Range | Lon Range | Elevation Range | Baseline Pressure | Diurnal Temp Range |
|---|---|---|---|---|---|---|
| **Coastal** | Mumbai (Colaba), Chennai, Kochi, Puri | $9.96^\circ\text{N} - 19.81^\circ\text{N}$ | $72.81^\circ\text{E} - 85.83^\circ\text{E}$ | $3\text{m} - 16\text{m}$ | $1011 - 1013\text{ hPa}$ | $5.5^\circ\text{C} - 7.5^\circ\text{C}$ |
| **Arid / Desert** | Jodhpur, Jaisalmer, Bikaner, Rajkot | $22.30^\circ\text{N} - 28.02^\circ\text{N}$ | $70.80^\circ\text{E} - 73.31^\circ\text{E}$ | $138\text{m} - 242\text{m}$ | $985 - 997\text{ hPa}$ | $13.5^\circ\text{C} - 18.0^\circ\text{C}$ |
| **Hill / Montane** | Shimla, Srinagar, Shillong, Ooty | $11.41^\circ\text{N} - 34.08^\circ\text{N}$ | $74.80^\circ\text{E} - 91.89^\circ\text{E}$ | $1496\text{m} - 2240\text{m}$ | $773 - 845\text{ hPa}$ | $8.0^\circ\text{C} - 12.0^\circ\text{C}$ |
| **Gangetic Plains** | New Delhi, Lucknow, Patna, Nagpur | $21.09^\circ\text{N} - 28.58^\circ\text{N}$ | $77.21^\circ\text{E} - 85.09^\circ\text{E}$ | $53\text{m} - 310\text{m}$ | $977 - 1007\text{ hPa}$ | $11.0^\circ\text{C} - 13.8^\circ\text{C}$ |

![Multi-Station Diurnal Signals Across Indian Climate Zones](C:/Users/Asus/.gemini/antigravity-ide/brain/c1ed1cc5-3e60-416e-a4ac-5f8ed8fe983b/climate_zones_comparison.png)

---

## 3. Genuine Extreme Weather Modeling (`is_anomaly = False`)

To prevent false alarms in the 5-tier pipeline, the generator injects genuine atmospheric extreme events:
- **Monsoon Squalls / Depressions**: Fast barometric drops (-7 to -14 hPa), cold rain downdrafts (-6°C to -10°C), and humidity surging to 99%.
- **Heatwaves**: Multi-day elevated baseline (+4.5°C to +8°C) and depressed humidity (down to 10–25%).
- **Temperature Inversions & Fog**: Saturated RH (>98%) and collapsed diurnal range.

All variables maintain physical self-consistency ($T_{\text{dew}} \le T_{\text{air}}$), and ground truth is explicitly labeled as **normal** (`is_anomaly = False`).

![Genuine Extreme Weather Dynamics](C:/Users/Asus/.gemini/antigravity-ide/brain/c1ed1cc5-3e60-416e-a4ac-5f8ed8fe983b/extreme_weather_cases.png)

---

## 4. Sensor Fault Taxonomy (`is_anomaly = True`)

The generator injects every fault mode specified in Problem Statement 26073 with high statistical fidelity:

1. `spike_or_drop`: ADC impulses (1–3 readings).
2. `frozen_sensor`: Deadlock resulting in zero-variance floating-point constant value over 3–24 hours.
3. `communication_dropout`: Packet loss leading to `NaN` readings over 30 min – 6 hours.
4. `calibration_drift`: Slow accumulating linear bias over 2–6 days.
5. `power_fluctuation_glitch`: Solar battery voltage ripple causing extreme jitter ($\sigma \times 15-25$).
6. `data_corruption`: Sentinel tokens (`-999.0`, `9999.0`).
7. `cross_sensor_inconsistency`: Subtle thermodynamic violations ($T_{\text{dew}} > T_{\text{air}}$).

**Station Locality**: All faults are localized to single stations. Neighboring stations remain unaffected, providing ground truth for Tier 3 spatial buddy-checks.

![Sensor Fault Taxonomy Visual Gallery](C:/Users/Asus/.gemini/antigravity-ide/brain/c1ed1cc5-3e60-416e-a4ac-5f8ed8fe983b/sensor_fault_taxonomy.png)

---

## 5. Verification & Validation Results

The automated test suite `validate.py` was executed directly on the generated dataset. All checks passed:

```
================================================================================
                   SkyGuard AI - Dataset Validation & QC Suite                  
================================================================================
Loaded master telemetry dataset: 138,240 records.
Testing physical plausibility of non-anomalous telemetry...
  [PASS] Physical bounds and thermodynamic consistency verified for all normal rows.
Testing internal consistency of anomaly labels...
  [PASS] Verified 397 communication dropout records contain NaN.
  [PASS] Verified frozen sensor zero-variance sequences.
  [PASS] Verified data corruption sentinel values.
  [PASS] Verified station-locality of injected faults.
Testing train/val/test splits and spatial holdouts...
  [PASS] Strict chronological boundaries: Train < Val < Test.
  [PASS] Zero station leakage in spatial holdouts ({'AWS_IND_C04', 'AWS_IND_A03', 'AWS_IND_H03'}).
Testing generator reproducibility with fixed random seed...
  [PASS] Full deterministic reproducibility confirmed across identical seeds.
================================================================================
          ALL VALIDATION CHECKS PASSED: DATASET IS METEOROLOGICALLY VALID       
================================================================================
```

---

## 6. Generated Artifacts & Directory Layout

```
d:/SIH/
├── configs/
│   └── default_config.yaml                      # Fully configurable generation settings
├── data/
│   ├── raw/
│   │   ├── aws_telemetry_master.parquet         # 138,240 records (4.2 MB)
│   │   ├── aws_telemetry_master.csv             # Full CSV export (15.5 MB)
│   │   └── generation_metadata.json             # Parameter audit & split timestamps
│   ├── splits/
│   │   ├── train.parquet                        # 78,624 records (70% chronological train)
│   │   ├── val.parquet                          # 16,848 records (15% chronological val)
│   │   ├── test.parquet                         # 16,848 records (15% chronological test)
│   │   ├── test.csv                             # Test split in CSV format
│   │   └── spatial_holdout_stations.parquet     # 25,920 records (3 held-out stations)
│   └── plots/
│       ├── climate_zones_comparison.png         # Multi-station diurnal trace
│       ├── extreme_weather_cases.png            # Extreme weather vs anomaly traces
│       └── sensor_fault_taxonomy.png            # Complete 7-fault gallery
├── docs/
│   ├── DATA_DICTIONARY.md                       # Canonical Telemetry Contract specification
│   └── METEOROLOGICAL_MODELING.md               # Governing thermodynamic equations
├── src/
│   ├── config.py                                # Dataclass / YAML config parser
│   ├── stations.py                              # 16 Indian AWS station catalog
│   ├── physics.py                               # Barometric & August-Roche-Magnus physics
│   ├── base_signals.py                          # Diurnal, synoptic AR(1), and tides
│   ├── extreme_weather.py                       # Physically coupled severe weather
│   ├── anomaly_injector.py                      # 7-fault station-local injector
│   └── pipeline.py                              # Master orchestrator & splitting
├── generate.py                                  # CLI entrypoint
├── validate.py                                  # Automated QC and validation suite
├── visualize.py                                 # Matplotlib visualization script
└── README.md                                    # Comprehensive documentation
```
