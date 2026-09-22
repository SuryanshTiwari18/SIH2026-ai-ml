"""SkyGuard AI - Feature Engineering Pipeline (Tiers 2-4 Feature Store).

Module: src.features
Author: SkyGuard AI Team (SIH 2026, PS 26073)

Grounding:
Formulas and transformations strictly implemented per EDA_INSIGHTS.md Section 8:
1. Temporal derivatives: delta_T, delta_P, delta_RH (segment-bounded, true gap restarts)
2. Rolling volatility: 1h/6h rolling variance per sensor, VolRatio = std_1h / (std_24h + epsilon)
3. Diurnal harmonic phase embeddings: sin_hour, cos_hour from timestamp
4. Dew point depression: T - T_dew via August-Roche-Magnus (reusing src.physics)
5. Vapor Pressure Deficit (VPD): e_s(T) * (1 - RH/100) (reusing src.physics)
6. Per-(station, hour) dynamic Mahalanobis distance fitted ONLY on normal train rows
7. Spatial buddy-check features: delta_T_buddy and delta_P_cluster via Haversine 3-NN

Leakage & Data Quality Guarantees:
- Scaler (StandardScaler) fitted strictly on TRAIN split where is_anomaly == False.
- True gap segmentation: any interval > 10 min starts a new segment.
- Zero lookback reach across exclusion gaps or segment boundaries.
- Full window min_periods: segment start rows missing lookback history get NaN and
  are routed to data/features/gap_edge_excluded_rows.parquet.
- Tier 1 excluded rows (communication_dropout, data_corruption) segregated to
  data/features/tier1_excluded_rows.parquet.
- near_excluded_gap flag set for observations within 1 hour of any exclusion boundary.
- Reference telemetry for spatial buddy check is cleansed of Tier 1 corruptions.
- Retained feature parquet files have ZERO NaNs in any feature column.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.physics import (
    calculate_baseline_pressure_hpa,
    calculate_dew_point_c,
    calculate_saturation_vapor_pressure_hpa,
)
from src.stations import INDIAN_AWS_STATIONS, get_station_catalog

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("SkyGuard.Features")

# Sensor columns canonical definition
SENSOR_COLS = ["temperature_c", "pressure_hpa", "humidity_pct"]

# Names of raw engineered features (to be scaled)
ENGINEERED_FEATURE_COLS = [
    "delta_T",
    "delta_P",
    "delta_RH",
    "var_T_1h",
    "var_T_6h",
    "vol_ratio_T",
    "var_P_1h",
    "var_P_6h",
    "vol_ratio_P",
    "var_RH_1h",
    "var_RH_6h",
    "vol_ratio_RH",
    "sin_hour",
    "cos_hour",
    "dew_point_dep",
    "vpd",
    "mahalanobis_dist",
    "delta_T_buddy",
    "delta_P_cluster",
]

EPSILON = 1e-4  # VolRatio numerical stabilizer


def compute_haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes great-circle distance between two geographic coordinates in kilometers."""
    r_earth = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return float(r_earth * c)


def build_spatial_neighbor_mapping(
    train_station_ids: Optional[List[str]] = None,
    k_neighbors: int = 3,
) -> Dict[str, List[str]]:
    """Builds spatial neighbor mapping for all stations in catalog.
    
    For any station, its k nearest reference neighbors are selected from the
    operational monitoring network (train stations).
    """
    catalog = get_station_catalog()
    all_stations = list(catalog.keys())
    ref_stations = train_station_ids if train_station_ids is not None else all_stations

    neighbor_mapping: Dict[str, List[str]] = {}
    for st_id in all_stations:
        st_meta = catalog[st_id]
        distances: List[Tuple[float, str]] = []
        for ref_id in ref_stations:
            if ref_id == st_id:
                continue
            ref_meta = catalog[ref_id]
            dist = compute_haversine_distance_km(
                st_meta.latitude, st_meta.longitude, ref_meta.latitude, ref_meta.longitude
            )
            distances.append((dist, ref_id))
        distances.sort(key=lambda x: x[0])
        neighbor_mapping[st_id] = [n[1] for n in distances[:k_neighbors]]

    return neighbor_mapping


def compute_diurnal_harmonics(df: pd.DataFrame) -> pd.DataFrame:
    """Computes continuous diurnal phase embeddings (sin/cos) from timestamps.
    
    Formula:
        hour_cont = hour + minute / 60.0
        sin_hour = sin(2 * pi * hour_cont / 24)
        cos_hour = cos(2 * pi * hour_cont / 24)
    """
    timestamps = pd.to_datetime(df["timestamp"])
    hour_cont = timestamps.dt.hour + timestamps.dt.minute / 60.0
    phase = 2.0 * np.pi * hour_cont / 24.0

    df["sin_hour"] = np.sin(phase).astype(np.float64)
    df["cos_hour"] = np.cos(phase).astype(np.float64)
    return df


def compute_thermodynamic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes physical dew point depression and vapor pressure deficit.
    
    Formulas:
        T_dew = calculate_dew_point_c(T, RH)  (August-Roche-Magnus)
        dew_point_dep = max(0, T - T_dew)
        VPD = e_s(T) * (1 - RH/100)
    """
    t = df["temperature_c"].to_numpy(dtype=np.float64)
    rh = df["humidity_pct"].to_numpy(dtype=np.float64)

    # August-Roche-Magnus Dew Point from physics module
    t_dew = calculate_dew_point_c(t, rh)
    dew_dep = np.maximum(0.0, t - t_dew)

    # Saturation Vapor Pressure & VPD
    es = calculate_saturation_vapor_pressure_hpa(t)
    vpd = np.maximum(0.0, es * (1.0 - (rh / 100.0)))

    df["dew_point_dep"] = dew_dep
    df["vpd"] = vpd
    return df


def compute_temporal_and_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes temporal first differences and rolling volatility per segment.
    
    Segment Boundary Guarantees:
    - Segments are partitioned strictly at any time gap > 10 min or station boundary.
    - NEVER diffs or rolls across a segment boundary.
    - Full-window min_periods enforcement:
        delta_*: first row of segment is NaN
        var_*_1h: first 5 rows of segment are NaN (min_periods=6)
        var_*_6h: first 35 rows of segment are NaN (min_periods=36)
        vol_ratio_*: first 143 rows of segment are NaN (min_periods=144)
    """
    df = df.copy()
    timestamps = pd.to_datetime(df["timestamp"])
    df["_ts"] = timestamps
    df = df.sort_values(["station_id", "_ts"]).reset_index(drop=True)

    # Identify true temporal gaps (>10 min interval or station boundary)
    dt_sec = df.groupby("station_id")["_ts"].diff().dt.total_seconds().fillna(600.0)
    station_changed = df["station_id"] != df["station_id"].shift(1)
    df["_is_gap"] = (dt_sec != 600.0) | station_changed
    df["_segment_id"] = df.groupby("station_id")["_is_gap"].cumsum()

    # Pre-allocate output arrays with NaN
    n_rows = len(df)
    delta_T = np.full(n_rows, np.nan, dtype=np.float64)
    delta_P = np.full(n_rows, np.nan, dtype=np.float64)
    delta_RH = np.full(n_rows, np.nan, dtype=np.float64)

    var_T_1h = np.full(n_rows, np.nan, dtype=np.float64)
    var_T_6h = np.full(n_rows, np.nan, dtype=np.float64)
    vol_ratio_T = np.full(n_rows, np.nan, dtype=np.float64)

    var_P_1h = np.full(n_rows, np.nan, dtype=np.float64)
    var_P_6h = np.full(n_rows, np.nan, dtype=np.float64)
    vol_ratio_P = np.full(n_rows, np.nan, dtype=np.float64)

    var_RH_1h = np.full(n_rows, np.nan, dtype=np.float64)
    var_RH_6h = np.full(n_rows, np.nan, dtype=np.float64)
    vol_ratio_RH = np.full(n_rows, np.nan, dtype=np.float64)

    for _, seg_df in df.groupby(["station_id", "_segment_id"]):
        idx = seg_df.index

        # Temperature
        t_vals = seg_df["temperature_c"]
        dt = t_vals.diff().to_numpy()
        v1_t = t_vals.rolling(6, min_periods=6).var().to_numpy()
        v6_t = t_vals.rolling(36, min_periods=36).var().to_numpy()
        s1_t = t_vals.rolling(6, min_periods=6).std().to_numpy()
        s24_t = t_vals.rolling(144, min_periods=144).std().to_numpy()
        vr_t = s1_t / (s24_t + EPSILON)

        delta_T[idx] = dt
        var_T_1h[idx] = v1_t
        var_T_6h[idx] = v6_t
        vol_ratio_T[idx] = vr_t

        # Pressure
        p_vals = seg_df["pressure_hpa"]
        dp = p_vals.diff().to_numpy()
        v1_p = p_vals.rolling(6, min_periods=6).var().to_numpy()
        v6_p = p_vals.rolling(36, min_periods=36).var().to_numpy()
        s1_p = p_vals.rolling(6, min_periods=6).std().to_numpy()
        s24_p = p_vals.rolling(144, min_periods=144).std().to_numpy()
        vr_p = s1_p / (s24_p + EPSILON)

        delta_P[idx] = dp
        var_P_1h[idx] = v1_p
        var_P_6h[idx] = v6_p
        vol_ratio_P[idx] = vr_p

        # Humidity
        rh_vals = seg_df["humidity_pct"]
        drh = rh_vals.diff().to_numpy()
        v1_rh = rh_vals.rolling(6, min_periods=6).var().to_numpy()
        v6_rh = rh_vals.rolling(36, min_periods=36).var().to_numpy()
        s1_rh = rh_vals.rolling(6, min_periods=6).std().to_numpy()
        s24_rh = rh_vals.rolling(144, min_periods=144).std().to_numpy()
        vr_rh = s1_rh / (s24_rh + EPSILON)

        delta_RH[idx] = drh
        var_RH_1h[idx] = v1_rh
        var_RH_6h[idx] = v6_rh
        vol_ratio_RH[idx] = vr_rh

    df.drop(columns=["_ts", "_is_gap", "_segment_id"], inplace=True)

    df["delta_T"] = delta_T
    df["delta_P"] = delta_P
    df["delta_RH"] = delta_RH

    df["var_T_1h"] = var_T_1h
    df["var_T_6h"] = var_T_6h
    df["vol_ratio_T"] = vol_ratio_T

    df["var_P_1h"] = var_P_1h
    df["var_P_6h"] = var_P_6h
    df["vol_ratio_P"] = vol_ratio_P

    df["var_RH_1h"] = var_RH_1h
    df["var_RH_6h"] = var_RH_6h
    df["vol_ratio_RH"] = vol_ratio_RH

    return df


def compute_excluded_gap_proximity(
    df: pd.DataFrame,
    excluded_timestamps_by_station: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """Adds boolean flag near_excluded_gap: True for rows within 1h of ANY exclusion boundary
    (data_corruption or communication_dropout).
    """
    df = df.copy()
    timestamps = pd.to_datetime(df["timestamp"]).to_numpy()
    stations = df["station_id"].to_numpy()

    near_gap = np.zeros(len(df), dtype=bool)

    for st_id, st_indices in df.groupby("station_id").groups.items():
        if st_id not in excluded_timestamps_by_station:
            continue
        excl_ts = excluded_timestamps_by_station[st_id]
        if len(excl_ts) == 0:
            continue

        st_ts = timestamps[st_indices]
        # shape: (n_rows, n_exclusions)
        diff_matrix = np.abs(st_ts[:, np.newaxis] - excl_ts[np.newaxis, :])
        min_diff_sec = diff_matrix.min(axis=1) / np.timedelta64(1, "s")
        is_near = min_diff_sec <= 3600.0  # 1 hour

        near_gap[st_indices] = is_near

    df["near_excluded_gap"] = near_gap
    return df


def fit_mahalanobis_model(train_df: pd.DataFrame) -> Dict[str, Any]:
    """Fits multivariate mu and inverse covariance per (station_id, hour) on NORMAL train rows.
    
    Leakage Rule: Only train rows where is_anomaly == False are used.
    Also computes global hour-of-day fallback stats for unseen holdout stations.
    """
    normal_train = train_df[train_df["is_anomaly"] == False].dropna(subset=SENSOR_COLS).copy()
    # Filter out sentinel corruptions if any
    normal_train = normal_train[normal_train["temperature_c"] > -50.0]
    normal_train["hour"] = pd.to_datetime(normal_train["timestamp"]).dt.hour

    stats: Dict[str, Any] = {
        "station_hour": {},
        "hour_fallback": {},
        "global_fallback": {},
    }

    # Global fallback
    x_global = normal_train[SENSOR_COLS].to_numpy(dtype=np.float64)
    mu_global = np.mean(x_global, axis=0)
    cov_global = np.cov(x_global, rowvar=False) + 1e-4 * np.eye(3)
    stats["global_fallback"] = {
        "mu": mu_global,
        "cov": cov_global,
        "inv_cov": np.linalg.pinv(cov_global),
    }

    # Per-hour fallback
    for hour, h_grp in normal_train.groupby("hour"):
        x_h = h_grp[SENSOR_COLS].to_numpy(dtype=np.float64)
        mu_h = np.mean(x_h, axis=0)
        cov_h = np.cov(x_h, rowvar=False) + 1e-4 * np.eye(3)
        stats["hour_fallback"][int(hour)] = {
            "mu": mu_h,
            "cov": cov_h,
            "inv_cov": np.linalg.pinv(cov_h),
        }

    # Per (station_id, hour)
    for (station_id, hour), grp in normal_train.groupby(["station_id", "hour"]):
        x_sh = grp[SENSOR_COLS].to_numpy(dtype=np.float64)
        if len(x_sh) >= 5:
            mu_sh = np.mean(x_sh, axis=0)
            cov_sh = np.cov(x_sh, rowvar=False) + 1e-4 * np.eye(3)
            inv_cov_sh = np.linalg.pinv(cov_sh)
        else:
            mu_sh = stats["hour_fallback"][int(hour)]["mu"]
            cov_sh = stats["hour_fallback"][int(hour)]["cov"]
            inv_cov_sh = stats["hour_fallback"][int(hour)]["inv_cov"]

        stats["station_hour"][(str(station_id), int(hour))] = {
            "mu": mu_sh,
            "cov": cov_sh,
            "inv_cov": inv_cov_sh,
        }

    logger.info(
        "Fitted Mahalanobis stats: %d (station, hour) buckets from %d normal train rows.",
        len(stats["station_hour"]),
        len(normal_train),
    )
    return stats


def apply_mahalanobis_distance(
    df: pd.DataFrame,
    mahalanobis_stats: Dict[str, Any],
    neighbor_mapping: Optional[Dict[str, List[str]]] = None,
) -> pd.DataFrame:
    """Computes Mahalanobis distance DM(x) per row using pre-fitted parameters.
    
    Formula:
        DM(x) = sqrt( (x - mu)^T * Sigma^-1 * (x - mu) )
    For unseen stations, utilizes the station's nearest training peer or hour fallback,
    with hydrostatic altitude compensation for pressure baseline differences.
    """
    df = df.copy()
    hours = pd.to_datetime(df["timestamp"]).dt.hour.to_numpy()
    stations = df["station_id"].to_numpy()
    x_matrix = df[SENSOR_COLS].to_numpy(dtype=np.float64)

    dm_values = np.zeros(len(df), dtype=np.float64)
    station_hour_stats = mahalanobis_stats["station_hour"]
    hour_fallback = mahalanobis_stats["hour_fallback"]
    global_fallback = mahalanobis_stats["global_fallback"]

    catalog = get_station_catalog()

    for i in range(len(df)):
        x_i = x_matrix[i]
        if np.isnan(x_i).any():
            dm_values[i] = 0.0
            continue

        st_id = stations[i]
        h = hours[i]
        key = (st_id, h)

        if key in station_hour_stats:
            bucket = station_hour_stats[key]
            mu = bucket["mu"]
        elif neighbor_mapping and st_id in neighbor_mapping:
            peer_st = neighbor_mapping[st_id][0]
            peer_key = (peer_st, h)
            bucket = station_hour_stats.get(peer_key, hour_fallback.get(h, global_fallback))
            mu = bucket["mu"].copy()
            # Hydrostatic altitude adjustment for pressure baseline
            if st_id in catalog and peer_st in catalog:
                st_alt = catalog[st_id].altitude_m
                peer_alt = catalog[peer_st].altitude_m
                dp_hydro = calculate_baseline_pressure_hpa(st_alt) - calculate_baseline_pressure_hpa(peer_alt)
                mu[1] += dp_hydro
        else:
            bucket = hour_fallback.get(h, global_fallback)
            mu = bucket["mu"]

        diff = x_i - mu
        dm_sq = float(diff @ bucket["inv_cov"] @ diff)
        dm_values[i] = math.sqrt(max(0.0, dm_sq))

    df["mahalanobis_dist"] = dm_values
    return df


def compute_spatial_buddy_features(
    df: pd.DataFrame,
    neighbor_mapping: Dict[str, List[str]],
    reference_telemetry: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Computes spatial peer delta features joined on matching timestamp.
    
    Formulas:
        delta_T_buddy = |T_station - T_nearest|
        delta_P_cluster = P_station - median(P_of_3_neighbors)
        
    Guarantees:
    - Excludes corrupted rows from reference telemetry so invalid sentinels (-999.0)
      never leak into peer delta calculations.
    """
    df = df.copy()
    ref = df if reference_telemetry is None else reference_telemetry

    # Strictly filter out corrupted / dropout rows from reference
    ref_mask = (
        ref["anomaly_type"].isin(["communication_dropout", "data_corruption"])
        | ref["temperature_c"].isna()
        | (ref["temperature_c"] < -50.0)
    )
    ref_clean = ref[~ref_mask].dropna(subset=SENSOR_COLS)

    # Pivot reference temperature and pressure by (timestamp, station_id)
    piv_t = ref_clean.pivot_table(index="timestamp", columns="station_id", values="temperature_c")
    piv_p = ref_clean.pivot_table(index="timestamp", columns="station_id", values="pressure_hpa")

    n_rows = len(df)
    delta_T_buddy = np.zeros(n_rows, dtype=np.float64)
    delta_P_cluster = np.zeros(n_rows, dtype=np.float64)

    timestamps = df["timestamp"].values
    stations = df["station_id"].values
    t_vals = df["temperature_c"].values
    p_vals = df["pressure_hpa"].values

    for i in range(n_rows):
        ts = timestamps[i]
        st = stations[i]
        t_i = t_vals[i]
        p_i = p_vals[i]

        if np.isnan(t_i) or np.isnan(p_i):
            delta_T_buddy[i] = 0.0
            delta_P_cluster[i] = 0.0
            continue

        neighbors = neighbor_mapping.get(st, [])
        if not neighbors:
            delta_T_buddy[i] = 0.0
            delta_P_cluster[i] = 0.0
            continue

        # 1. delta_T_buddy: |T_station - T_nearest|
        t_near = np.nan
        for n_st in neighbors:
            if ts in piv_t.index and n_st in piv_t.columns:
                val = piv_t.at[ts, n_st]
                if not np.isnan(val):
                    t_near = val
                    break

        if not np.isnan(t_near):
            delta_T_buddy[i] = abs(float(t_i - t_near))
        else:
            delta_T_buddy[i] = 0.0

        # 2. delta_P_cluster: P_station - median(P_of_3_neighbors)
        p_cluster_vals = []
        for n_st in neighbors[:3]:
            if ts in piv_p.index and n_st in piv_p.columns:
                p_val = piv_p.at[ts, n_st]
                if not np.isnan(p_val):
                    p_cluster_vals.append(p_val)

        if p_cluster_vals:
            cluster_median = float(np.median(p_cluster_vals))
            delta_P_cluster[i] = float(p_i - cluster_median)
        else:
            delta_P_cluster[i] = 0.0

    df["delta_T_buddy"] = delta_T_buddy
    df["delta_P_cluster"] = delta_P_cluster
    return df


def engineer_split_features(
    df: pd.DataFrame,
    neighbor_mapping: Dict[str, List[str]],
    mahalanobis_stats: Dict[str, Any],
    excluded_timestamps_by_station: Dict[str, np.ndarray],
    reference_telemetry: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Applies all raw feature engineering transformations to a cleansed split dataframe."""
    logger.info("Computing diurnal harmonics...")
    df = compute_diurnal_harmonics(df)

    logger.info("Computing thermodynamic features (August-Roche-Magnus)...")
    df = compute_thermodynamic_features(df)

    logger.info("Computing temporal derivatives & rolling volatility (contiguous segments)...")
    df = compute_temporal_and_volatility_features(df)

    logger.info("Computing excluded gap proximity flag (near_excluded_gap)...")
    df = compute_excluded_gap_proximity(df, excluded_timestamps_by_station)

    logger.info("Computing dynamic Mahalanobis distance...")
    df = apply_mahalanobis_distance(df, mahalanobis_stats, neighbor_mapping)

    logger.info("Computing spatial buddy features...")
    df = compute_spatial_buddy_features(df, neighbor_mapping, reference_telemetry)

    return df


def fit_feature_scaler(train_retained_df: pd.DataFrame) -> StandardScaler:
    """Fits StandardScaler strictly on normal (is_anomaly == False) train rows.
    
    Leakage Rule:
    - Never fit on anomalous rows.
    - Never fit on validation, test, or holdout splits.
    - Fitted ONLY on retained rows (free of NaNs and lookback gaps).
    """
    normal_mask = train_retained_df["is_anomaly"] == False
    fit_rows = train_retained_df[normal_mask][ENGINEERED_FEATURE_COLS]

    scaler = StandardScaler()
    scaler.fit(fit_rows)
    logger.info(
        "Fitted StandardScaler on %d train normal rows across %d features.",
        len(fit_rows),
        len(ENGINEERED_FEATURE_COLS),
    )
    return scaler


def apply_feature_scaling(
    df: pd.DataFrame,
    scaler: StandardScaler,
) -> pd.DataFrame:
    """Applies fitted scaler to engineered columns, generating *_scaled columns.
    
    Leaves raw sensor columns and labels strictly unscaled.
    """
    df = df.copy()
    scaled_array = scaler.transform(df[ENGINEERED_FEATURE_COLS])
    for idx, col in enumerate(ENGINEERED_FEATURE_COLS):
        df[f"{col}_scaled"] = scaled_array[:, idx]
    return df


def run_feature_pipeline(
    splits_dir: str | Path = "data/splits",
    output_features_dir: str | Path = "data/features",
    models_dir: str | Path = "models",
) -> Dict[str, pd.DataFrame]:
    """Executes the complete end-to-end feature engineering pipeline with gap isolation.
    
    Steps:
    1. Load raw splits: train, val, test, spatial_holdout_stations.
    2. Segregate Tier 1 excluded rows (communication_dropout, data_corruption) FIRST.
    3. Partition clean series into contiguous segments at true temporal gaps (>10 min).
    4. Compute all derivative and rolling features per segment only.
    5. Route lookback-deficient edge rows (NaNs at segment starts) to
       data/features/gap_edge_excluded_rows.parquet.
    6. Fit StandardScaler strictly on normal train retained rows.
    7. Scale retained features across all splits.
    8. Save all parquet datasets and fitted model artifacts.
    """
    splits_path = Path(splits_dir)
    features_path = Path(output_features_dir)
    models_path = Path(models_dir)

    features_path.mkdir(parents=True, exist_ok=True)
    models_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading raw split parquet files from %s...", splits_path)
    raw_splits = {
        "train": pd.read_parquet(splits_path / "train.parquet"),
        "val": pd.read_parquet(splits_path / "val.parquet"),
        "test": pd.read_parquet(splits_path / "test.parquet"),
        "spatial_holdout": pd.read_parquet(splits_path / "spatial_holdout_stations.parquet"),
    }

    train_stations = sorted(raw_splits["train"]["station_id"].unique().tolist())
    logger.info("Identified %d operational training stations: %s", len(train_stations), train_stations)

    # 1. Build & Save Spatial Neighbors
    neighbor_mapping = build_spatial_neighbor_mapping(train_station_ids=train_stations, k_neighbors=3)
    neighbors_file = models_path / "spatial_neighbors.json"
    with open(neighbors_file, "w", encoding="utf-8") as f:
        json.dump(neighbor_mapping, f, indent=2)
    logger.info("Saved spatial neighbor mapping to %s", neighbors_file)

    # 2. Fit Mahalanobis Model on Normal Train Rows
    mahalanobis_stats = fit_mahalanobis_model(raw_splits["train"])
    mahalanobis_file = models_path / "mahalanobis_stats.joblib"
    joblib.dump(mahalanobis_stats, mahalanobis_file)
    logger.info("Saved Mahalanobis model stats to %s", mahalanobis_file)

    # 3. Segregate Tier 1 exclusions and map exclusion timestamps per split
    tier1_excluded_list: List[pd.DataFrame] = []
    clean_splits: Dict[str, pd.DataFrame] = {}
    excluded_timestamps_by_station: Dict[str, List[pd.Timestamp]] = {}

    for split_name, df_raw in raw_splits.items():
        is_tier1 = (
            df_raw["anomaly_type"].isin(["communication_dropout", "data_corruption"])
            | df_raw["temperature_c"].isna()
            | df_raw["pressure_hpa"].isna()
            | df_raw["humidity_pct"].isna()
        )
        tier1_excl = df_raw[is_tier1].copy()
        tier1_excl["split"] = split_name
        tier1_excluded_list.append(tier1_excl)

        for _, row in tier1_excl.iterrows():
            st = row["station_id"]
            ts = pd.to_datetime(row["timestamp"])
            excluded_timestamps_by_station.setdefault(st, []).append(ts)

        clean = df_raw[~is_tier1].copy()
        clean["timestamp"] = pd.to_datetime(clean["timestamp"])
        clean_splits[split_name] = clean.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    # Convert exclusion timestamps to numpy datetime64 arrays for fast search
    excl_ts_array_by_st = {
        st: np.array([ts.to_datetime64() for ts in tss])
        for st, tss in excluded_timestamps_by_station.items()
    }

    # Clean operational reference telemetry for buddy checks
    op_clean_ref = pd.concat(
        [clean_splits["train"], clean_splits["val"], clean_splits["test"]],
        axis=0,
        ignore_index=True,
    )

    # 4. Engineer features for each clean split
    engineered_splits: Dict[str, pd.DataFrame] = {}
    for split_name, clean_df in clean_splits.items():
        ref = op_clean_ref if split_name == "spatial_holdout" else clean_df
        logger.info("Engineering features for split '%s' (%d clean rows)...", split_name, len(clean_df))
        eng = engineer_split_features(
            clean_df,
            neighbor_mapping,
            mahalanobis_stats,
            excl_ts_array_by_st,
            reference_telemetry=ref,
        )
        engineered_splits[split_name] = eng

    # 5. Route edge rows (missing lookback context) to gap_edge_excluded_rows.parquet
    gap_edge_excluded_list: List[pd.DataFrame] = []
    retained_splits: Dict[str, pd.DataFrame] = {}

    for split_name, eng_df in engineered_splits.items():
        is_edge = eng_df[ENGINEERED_FEATURE_COLS].isna().any(axis=1)
        edge_excl = eng_df[is_edge].copy()
        edge_excl["split"] = split_name
        gap_edge_excluded_list.append(edge_excl)

        retained = eng_df[~is_edge].copy()
        retained_splits[split_name] = retained
        logger.info(
            "Split '%s': %d retained rows, %d gap edge rows excluded.",
            split_name,
            len(retained),
            len(edge_excl),
        )

    # 6. Fit StandardScaler strictly on normal train retained rows
    scaler = fit_feature_scaler(retained_splits["train"])
    scaler_file = models_path / "feature_scaler.joblib"
    joblib.dump(scaler, scaler_file)
    logger.info("Saved fitted feature scaler to %s", scaler_file)

    # 7. Apply scaling across all retained splits
    final_scaled_splits: Dict[str, pd.DataFrame] = {}
    for split_name, ret_df in retained_splits.items():
        logger.info("Applying feature scaling to '%s' retained split...", split_name)
        scaled_df = apply_feature_scaling(ret_df, scaler)
        final_scaled_splits[split_name] = scaled_df

        # Save split parquet
        out_file = features_path / f"{split_name}.parquet"
        scaled_df.to_parquet(out_file, index=False)
        logger.info("Saved %s to %s (%d rows, %d cols)", split_name, out_file, len(scaled_df), len(scaled_df.columns))

    # 8. Save excluded row datasets
    tier1_excluded_df = pd.concat(tier1_excluded_list, axis=0, ignore_index=True)
    tier1_file = features_path / "tier1_excluded_rows.parquet"
    tier1_excluded_df.to_parquet(tier1_file, index=False)
    logger.info("Saved %d Tier 1 excluded rows to %s", len(tier1_excluded_df), tier1_file)

    gap_edge_excluded_df = pd.concat(gap_edge_excluded_list, axis=0, ignore_index=True)
    gap_edge_file = features_path / "gap_edge_excluded_rows.parquet"
    gap_edge_excluded_df.to_parquet(gap_edge_file, index=False)
    logger.info("Saved %d gap edge excluded rows to %s", len(gap_edge_excluded_df), gap_edge_file)

    # 9. Audit total row count integrity
    logger.info("=== Final Row Count Balance Audit ===")
    for split_name in raw_splits.keys():
        n_raw = len(raw_splits[split_name])
        n_tier1 = len(tier1_excluded_df[tier1_excluded_df["split"] == split_name])
        n_edge = len(gap_edge_excluded_df[gap_edge_excluded_df["split"] == split_name])
        n_retained = len(final_scaled_splits[split_name])
        total_accounted = n_tier1 + n_edge + n_retained
        logger.info(
            "Split '%s': raw=%d == tier1(%d) + gap_edge(%d) + retained(%d) = %d [MATCH: %s]",
            split_name,
            n_raw,
            n_tier1,
            n_edge,
            n_retained,
            total_accounted,
            n_raw == total_accounted,
        )
        assert n_raw == total_accounted, f"Row count mismatch in split {split_name}!"

    logger.info("Feature engineering pipeline completed successfully with full gap isolation!")
    return final_scaled_splits


if __name__ == "__main__":
    run_feature_pipeline()
