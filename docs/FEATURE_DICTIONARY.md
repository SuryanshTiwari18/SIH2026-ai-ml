# SkyGuard AI: Feature Engineering Dictionary

This dictionary defines every engineered feature produced by [`src/features.py`](file:///d:/SIH/src/features.py) for consumption by Tiers 2–4 of the SkyGuard AI anomaly detection pipeline (SIH 2026, PS 26073).

All features are mathematically grounded in Section 8 of [`docs/EDA_INSIGHTS.md`](file:///d:/SIH/docs/EDA_INSIGHTS.md).

---

## 1. Pipeline Overview & Scaling Integrity

- **Raw vs Scaled Representation**: For every engineered feature `F`, two representations exist:
  1. `F`: Raw physical value preserving meteorological units (°C, hPa, %, dimensionless).
  2. `F_scaled`: Standardized z-score transformed via `StandardScaler`.
- **Zero-Leakage Invariant**: The scaler is fitted **strictly on normal rows (`is_anomaly == False`) of the `train` split** ($N = 70,905$). It is applied unchanged across `train`, `val`, `test`, and `spatial_holdout`.
- **Tier 1 Exclusion**: Telemetry belonging to `communication_dropout` or `data_corruption` episodes are routed to `data/features/tier1_excluded_rows.parquet` and excluded from model training splits to prevent corruption of continuous derivatives and variance windows.
- **Zero-NaN Invariant**: Within the retained feature datasets (`data/features/{train,val,test,spatial_holdout}.parquet`), all `*_scaled` columns contain **0 NaN values**.

---

## 2. Feature Definitions & Catalog

| Feature Name | Scaled Column | Formula / Calculation | Units | Targeted Fault Modes ("Catches") | Train Split Range `[Min, Max]` | Train Mean ± Std |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `delta_T` | `delta_T_scaled` | $T_t - T_{t-1}$ (contiguous segment) | °C | `spike_or_drop`, `power_fluctuation_glitch` | `[-15.674, 1014.217]` | `0.016 ± 3.814` |
| `delta_P` | `delta_P_scaled` | $P_t - P_{t-1}$ (contiguous segment) | hPa | `spike_or_drop`, `power_fluctuation_glitch` | `[-9221.981, 1772.824]` | `-0.073 ± 35.359` |
| `delta_RH` | `delta_RH_scaled` | $RH_t - RH_{t-1}$ (contiguous segment) | % | `spike_or_drop`, `power_fluctuation_glitch` | `[-48.954, 98.816]` | `0.005 ± 1.276` |
| `var_T_1h` | `var_T_1h_scaled` | $\mathrm{Var}(T_{t-5:t})$ (6-step rolling variance) | °C² | `frozen_sensor` ($\sigma^2 \to 0$), `power_fluctuation_glitch` | `[0.000, 274449.542]` | `18.005 ± 2135.806` |
| `var_T_6h` | `var_T_6h_scaled` | $\mathrm{Var}(T_{t-35:t})$ (36-step rolling variance) | °C² | `frozen_sensor`, persistent instability | `[0.000, 55633.187]` | `30.901 ± 1208.141` |
| `vol_ratio_T` | `vol_ratio_T_scaled` | $\frac{\sigma_{1\mathrm{h}}(T)}{\sigma_{24\mathrm{h}}(T) + \epsilon}$ ($\epsilon = 10^{-4}$) | ratio | `power_fluctuation_glitch` ($\mathrm{VolRatio} \gg 3$) | `[0.000, 4.380]` | `0.097 ± 0.105` |
| `var_P_1h` | `var_P_1h_scaled` | $\mathrm{Var}(P_{t-5:t})$ (6-step rolling variance) | hPa² | `frozen_sensor` ($\sigma^2 \to 0$), `power_fluctuation_glitch` | `[0.000, 14175521.209]` | `1065.936 ± 117957.825` |
| `var_P_6h` | `var_P_6h_scaled` | $\mathrm{Var}(P_{t-35:t})$ (36-step rolling variance) | hPa² | `frozen_sensor`, persistent barometric anomalies | `[0.000, 2362341.043]` | `1269.851 ± 52054.520` |
| `vol_ratio_P` | `vol_ratio_P_scaled` | $\frac{\sigma_{1\mathrm{h}}(P)}{\sigma_{24\mathrm{h}}(P) + \epsilon}$ ($\epsilon = 10^{-4}$) | ratio | `power_fluctuation_glitch` ($\mathrm{VolRatio} \gg 3$) | `[0.000, 4.900]` | `0.235 ± 0.173` |
| `var_RH_1h` | `var_RH_1h_scaled` | $\mathrm{Var}(RH_{t-5:t})$ (6-step rolling variance) | %² | `frozen_sensor` ($\sigma^2 \to 0$), `power_fluctuation_glitch` | `[0.000, 2631.151]` | `2.019 ± 30.727` |
| `var_RH_6h` | `var_RH_6h_scaled` | $\mathrm{Var}(RH_{t-35:t})$ (36-step rolling variance) | %² | `frozen_sensor`, diurnal humidity failure | `[0.000, 681.091]` | `26.044 ± 34.242` |
| `vol_ratio_RH` | `vol_ratio_RH_scaled` | $\frac{\sigma_{1\mathrm{h}}(RH)}{\sigma_{24\mathrm{h}}(RH) + \epsilon}$ ($\epsilon = 10^{-4}$) | ratio | `power_fluctuation_glitch` ($\mathrm{VolRatio} \gg 3$) | `[0.000, 3.792]` | `0.115 ± 0.114` |
| `sin_hour` | `sin_hour_scaled` | $\sin\left(\frac{2\pi \cdot (H + M/60)}{24}\right)$ | ratio | Enables autoencoder diurnal modeling without overfitting | `[-1.000, 1.000]` | `-0.000 ± 0.707` |
| `cos_hour` | `cos_hour_scaled` | $\cos\left(\frac{2\pi \cdot (H + M/60)}{24}\right)$ | ratio | Enables autoencoder diurnal modeling without overfitting | `[-1.000, 1.000]` | `-0.000 ± 0.707` |
| `dew_point_dep` | `dew_point_dep_scaled` | $T - T_{\mathrm{dew}}$ (August-Roche-Magnus) | °C | `cross_sensor_inconsistency`, physical violations ($T < T_d$) | `[0.000, 90.603]` | `11.286 ± 9.203` |
| `vpd` | `vpd_scaled` | $e_s(T) \cdot \left(1 - \frac{RH}{100}\right)$ | hPa | `cross_sensor_inconsistency` (e.g. 42°C with 85% RH) | `[0.000, 125.412]` | `22.207 ± 20.676` |
| `mahalanobis_dist` | `mahalanobis_dist_scaled` | $\sqrt{(x - \mu_{s, h})^T \Sigma_{s, h}^{-1} (x - \mu_{s, h})}$ | $D_M$ | `cross_sensor_inconsistency` ($D_M > 7.0$) | `[0.028, 82.704]` | `1.808 ± 2.471` |
| `delta_T_buddy` | `delta_T_buddy_scaled` | $\|T_{\mathrm{station}} - T_{\mathrm{nearest}}\|$ | °C | `calibration_drift`, separates sensor faults from extreme weather | `[0.000, 1029.049]` | `6.990 ± 9.513` |
| `delta_P_cluster` | `delta_P_cluster_scaled` | $P_{\mathrm{station}} - \mathrm{median}(P_{\mathrm{3\_neighbors}})$ | hPa | `calibration_drift`, separates synoptic fronts from sensor drift | `[-287.122, 218.070]` | `-44.265 ± 93.454` |

---

## 3. Metadata & Label Columns Preserved Unscaled

The following columns are preserved in their original, unscaled format in the output feature files:

1. **`station_id`** (string): Unique Indian AWS observatory identifier (e.g. `AWS_IND_C01`).
2. **`timestamp`** (timestamp[ns]): Observation date and time (10-minute cadence).
3. **`latitude`**, **`longitude`** (float64): Station coordinates in decimal degrees.
4. **`temperature_c`** (float64): Unscaled ambient surface temperature in °C.
5. **`pressure_hpa`** (float64): Unscaled barometric surface pressure in hPa.
6. **`humidity_pct`** (float64): Unscaled relative humidity in %.
7. **`near_dropout_gap`** (boolean): Flag set to `True` for any observation occurring within $\le 60$ minutes of a `communication_dropout` episode on that station.
8. **`is_anomaly`** (boolean): Ground truth binary label.
9. **`anomaly_type`** (string): Ground truth fault category (`calibration_drift`, `frozen_sensor`, `cross_sensor_inconsistency`, `power_fluctuation_glitch`, `spike_or_drop`, etc.).
10. **`anomaly_severity`** (float64): Normalized severity score $[0.0, 1.0]$.

---

## 4. Persisted Artifacts for Production Inference

1. **[`models/feature_scaler.joblib`](file:///d:/SIH/models/feature_scaler.joblib)**: Fitted `sklearn.preprocessing.StandardScaler` across all 19 engineered features.
2. **[`models/mahalanobis_stats.joblib`](file:///d:/SIH/models/mahalanobis_stats.joblib)**: Precomputed mean vector $\mu$ and inverse covariance $\Sigma^{-1}$ for 288 $(station, hour)$ buckets + hourly and global fallbacks with hydrostatic elevation compensation.
3. **[`models/spatial_neighbors.json`](file:///d:/SIH/models/spatial_neighbors.json)**: Exact 3-nearest neighbor station mapping derived via Haversine distance matrix.
