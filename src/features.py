"""SkyGuard AI - Feature Engineering Pipeline (Tiers 2-4 Feature Store).

Module: src.features
Author: SkyGuard AI Team (SIH 2026, PS 26073)

Grounding:
Formulas and transformations strictly implemented per EDA_INSIGHTS.md Section 8:
1. Temporal derivatives: delta_T, delta_P, delta_RH (station-bounded, restart at dropout)
2. Rolling volatility: 1h/6h rolling variance per sensor, VolRatio = std_1h / (std_24h + epsilon)
3. Diurnal harmonic phase embeddings: sin_hour, cos_hour from timestamp
4. Dew point depression: T - T_dew via August-Roche-Magnus (reusing src.physics)
5. Vapor Pressure Deficit (VPD): e_s(T) * (1 - RH/100) (reusing src.physics)
6. Per-(station, hour) dynamic Mahalanobis distance fitted ONLY on normal train rows
7. Spatial buddy-check features: delta_T_buddy and delta_P_cluster via Haversine 3-NN

Leakage & Data Quality Guarantees:
- Scaler (StandardScaler) fitted strictly on TRAIN split where is_anomaly == False.
- Never forward-fill across dropout gaps. Hard boundary segmentation.
- Tier 1 excluded rows (communication_dropout, data_corruption) segregated to
  data/features/tier1_excluded_rows.parquet.
- Output parquet files retain raw sensors unscaled alongside scaled engineered columns.
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
    operational monitoring network (train stations). If train_station_ids is None,
    all stations in the catalog are used.
    
    Args:
        train_station_ids: List of operational training station IDs.
        k_neighbors: Number of nearest neighbors to retain (default 3).
        
    Returns:
        Dict mapping station_id -> list of k nearest neighbor station_ids.
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
    """Computes temporal first differences and rolling volatility per station.
    
    Boundary Guarantees:
    - Never diffs across station changes.
    - Never diffs or rolls across communication_dropout episodes or gaps.
    - Identifies contiguous valid segments: first row of segment resets derivative to 0.0
      and min_periods=1 ensures no NaNs are produced.
    """
    df = df.sort_values(["station_id", "timestamp"]).copy()
    timestamps = pd.to_datetime(df["timestamp"])

    # Detect dropout / NaN gaps
    is_nan_sensor = (
        df["temperature_c"].isna() | df["pressure_hpa"].isna() | df["humidity_pct"].isna()
    )
    is_dropout_type = df["anomaly_type"].fillna("") == "communication_dropout"
    is_dropout = is_nan_sensor | is_dropout_type

    # Detect temporal discontinuities (>10 min interval or station boundary)
    dt_sec = timestamps.diff().dt.total_seconds().fillna(600.0)
    station_changed = df["station_id"] != df["station_id"].shift(1)
    time_gap = (dt_sec != 600.0) & (~station_changed)

    # Contiguous segment identifier: increments whenever boundary is encountered
    boundary_marker = station_changed | time_gap | is_dropout | is_dropout.shift(1, fill_value=False)
    segment_id = boundary_marker.cumsum()

    # Pre-allocate output feature arrays
    n_rows = len(df)
    delta_T = np.zeros(n_rows, dtype=np.float64)
    delta_P = np.zeros(n_rows, dtype=np.float64)
    delta_RH = np.zeros(n_rows, dtype=np.float64)

    var_T_1h = np.zeros(n_rows, dtype=np.float64)
    var_T_6h = np.zeros(n_rows, dtype=np.float64)
    vol_ratio_T = np.zeros(n_rows, dtype=np.float64)

    var_P_1h = np.zeros(n_rows, dtype=np.float64)
    var_P_6h = np.zeros(n_rows, dtype=np.float64)
    vol_ratio_P = np.zeros(n_rows, dtype=np.float64)

    var_RH_1h = np.zeros(n_rows, dtype=np.float64)
    var_RH_6h = np.zeros(n_rows, dtype=np.float64)
    vol_ratio_RH = np.zeros(n_rows, dtype=np.float64)

    # Compute strictly within contiguous non-dropout segments
    df["_segment_id"] = segment_id
    for _, seg_df in df[~is_dropout].groupby("_segment_id"):
        idx = seg_df.index

        # Temperature derivatives & rolling volatility
        t_vals = seg_df["temperature_c"]
        dt = t_vals.diff().fillna(0.0).to_numpy()
        delta_T[df.index.get_indexer(idx)] = dt

        v1_t = t_vals.rolling(6, min_periods=1).var().fillna(0.0).to_numpy()
        v6_t = t_vals.rolling(36, min_periods=1).var().fillna(0.0).to_numpy()
        s1_t = t_vals.rolling(6, min_periods=1).std().fillna(0.0).to_numpy()
        s24_t = t_vals.rolling(144, min_periods=1).std().fillna(0.0).to_numpy()
        vr_t = s1_t / (s24_t + EPSILON)

        var_T_1h[df.index.get_indexer(idx)] = v1_t
        var_T_6h[df.index.get_indexer(idx)] = v6_t
        vol_ratio_T[df.index.get_indexer(idx)] = vr_t

        # Pressure derivatives & rolling volatility
        p_vals = seg_df["pressure_hpa"]
        dp = p_vals.diff().fillna(0.0).to_numpy()
        delta_P[df.index.get_indexer(idx)] = dp

        v1_p = p_vals.rolling(6, min_periods=1).var().fillna(0.0).to_numpy()
        v6_p = p_vals.rolling(36, min_periods=1).var().fillna(0.0).to_numpy()
        s1_p = p_vals.rolling(6, min_periods=1).std().fillna(0.0).to_numpy()
        s24_p = p_vals.rolling(144, min_periods=1).std().fillna(0.0).to_numpy()
        vr_p = s1_p / (s24_p + EPSILON)

        var_P_1h[df.index.get_indexer(idx)] = v1_p
        var_P_6h[df.index.get_indexer(idx)] = v6_p
        vol_ratio_P[df.index.get_indexer(idx)] = vr_p

        # Humidity derivatives & rolling volatility
        rh_vals = seg_df["humidity_pct"]
        drh = rh_vals.diff().fillna(0.0).to_numpy()
        delta_RH[df.index.get_indexer(idx)] = drh

        v1_rh = rh_vals.rolling(6, min_periods=1).var().fillna(0.0).to_numpy()
        v6_rh = rh_vals.rolling(36, min_periods=1).var().fillna(0.0).to_numpy()
        s1_rh = rh_vals.rolling(6, min_periods=1).std().fillna(0.0).to_numpy()
        s24_rh = rh_vals.rolling(144, min_periods=1).std().fillna(0.0).to_numpy()
        vr_rh = s1_rh / (s24_rh + EPSILON)

        var_RH_1h[df.index.get_indexer(idx)] = v1_rh
        var_RH_6h[df.index.get_indexer(idx)] = v6_rh
        vol_ratio_RH[df.index.get_indexer(idx)] = vr_rh

    df.drop(columns=["_segment_id"], inplace=True)

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


def compute_dropout_proximity(df: pd.DataFrame) -> pd.DataFrame:
    """Adds boolean flag near_dropout_gap: True for rows within 1h of a dropout episode."""
    df = df.copy()
    timestamps = pd.to_datetime(df["timestamp"])
    is_dropout = (
        df["anomaly_type"].fillna("") == "communication_dropout"
    ) | df["temperature_c"].isna()

    near_gap = np.zeros(len(df), dtype=bool)

    # Process each station independently
    for station_id, st_indices in df.groupby("station_id").groups.items():
        st_dropouts = timestamps.loc[st_indices][is_dropout.loc[st_indices]]
        if st_dropouts.empty:
            continue

        st_ts = timestamps.loc[st_indices].to_numpy()
        drop_ts = st_dropouts.to_numpy()

        # Vectorized time difference to nearest dropout episode
        # shape: (n_station_rows, n_dropouts)
        diff_matrix = np.abs(st_ts[:, np.newaxis] - drop_ts[np.newaxis, :])
        min_diff_sec = diff_matrix.min(axis=1) / np.timedelta64(1, "s")
        is_near = min_diff_sec <= 3600.0  # 1 hour

        near_gap[df.index.get_indexer(st_indices)] = is_near

    df["near_dropout_gap"] = near_gap
    return df


def fit_mahalanobis_model(train_df: pd.DataFrame) -> Dict[str, Any]:
    """Fits multivariate mu and inverse covariance per (station_id, hour) on NORMAL train rows.
    
    Leakage Rule: Only train rows where is_anomaly == False are used.
    Also computes global hour-of-day fallback stats for unseen holdout stations.
    """
    normal_train = train_df[train_df["is_anomaly"] == False].dropna(subset=SENSOR_COLS).copy()
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
    For unseen stations, utilizes the station's nearest training peer or hour fallback.
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

    # Iterate row-by-row
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
            # Unseen station: look up nearest training neighbor
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
        
    Args:
        df: Target dataframe.
        neighbor_mapping: Mapping of station_id -> list of 3 nearest neighbor station_ids.
        reference_telemetry: Optional dataframe containing telemetry of operational stations
            (useful when processing spatial holdout stations that reference train stations).
            If None, df itself is used as reference.
    """
    df = df.copy()
    ref = df if reference_telemetry is None else reference_telemetry

    # Pivot reference temperature and pressure by (timestamp, station_id)
    ref_clean = ref.dropna(subset=SENSOR_COLS)
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
        # If nearest neighbor has NaN / dropout at ts, try next closest available neighbor
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
    reference_telemetry: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Applies all raw feature engineering transformations to a split dataframe."""
    logger.info("Computing diurnal harmonics...")
    df = compute_diurnal_harmonics(df)

    logger.info("Computing thermodynamic features (August-Roche-Magnus)...")
    df = compute_thermodynamic_features(df)

    logger.info("Computing temporal derivatives & rolling volatility (contiguous blocks)...")
    df = compute_temporal_and_volatility_features(df)

    logger.info("Computing dropout proximity flag (near_dropout_gap)...")
    df = compute_dropout_proximity(df)

    logger.info("Computing dynamic Mahalanobis distance...")
    df = apply_mahalanobis_distance(df, mahalanobis_stats, neighbor_mapping)

    logger.info("Computing spatial buddy features...")
    df = compute_spatial_buddy_features(df, neighbor_mapping, reference_telemetry)

    return df


def fit_feature_scaler(train_df: pd.DataFrame) -> StandardScaler:
    """Fits StandardScaler strictly on normal (is_anomaly == False) train rows.
    
    Leakage Rule:
    - Never fit on anomalous rows.
    - Never fit on validation, test, or holdout splits.
    """
    normal_mask = (train_df["is_anomaly"] == False) & (
        ~train_df["anomaly_type"].isin(["communication_dropout", "data_corruption"])
    )
    fit_rows = train_df[normal_mask][ENGINEERED_FEATURE_COLS]

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
    """Executes the complete end-to-end feature engineering pipeline.
    
    Steps:
    1. Load splits: train, val, test, spatial_holdout_stations.
    2. Build spatial neighbor mapping (k=3) from station coordinates.
    3. Fit Mahalanobis model strictly on normal train rows.
    4. Compute engineered features for all splits.
    5. Fit StandardScaler strictly on normal train rows.
    6. Scale engineered features across all splits.
    7. Segregate Tier 1 excluded rows (communication_dropout, data_corruption)
       into data/features/tier1_excluded_rows.parquet.
    8. Save all parquet datasets and fitted model artifacts.
    """
    splits_path = Path(splits_dir)
    features_path = Path(output_features_dir)
    models_path = Path(models_dir)

    features_path.mkdir(parents=True, exist_ok=True)
    models_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading raw split parquet files from %s...", splits_path)
    train_raw = pd.read_parquet(splits_path / "train.parquet")
    val_raw = pd.read_parquet(splits_path / "val.parquet")
    test_raw = pd.read_parquet(splits_path / "test.parquet")
    holdout_raw = pd.read_parquet(splits_path / "spatial_holdout_stations.parquet")

    train_stations = sorted(train_raw["station_id"].unique().tolist())
    logger.info("Identified %d operational training stations: %s", len(train_stations), train_stations)

    # 1. Build & Save Spatial Neighbors
    neighbor_mapping = build_spatial_neighbor_mapping(train_station_ids=train_stations, k_neighbors=3)
    neighbors_file = models_path / "spatial_neighbors.json"
    with open(neighbors_file, "w", encoding="utf-8") as f:
        json.dump(neighbor_mapping, f, indent=2)
    logger.info("Saved spatial neighbor mapping to %s", neighbors_file)

    # 2. Fit Mahalanobis Model on Normal Train Rows
    mahalanobis_stats = fit_mahalanobis_model(train_raw)
    mahalanobis_file = models_path / "mahalanobis_stats.joblib"
    joblib.dump(mahalanobis_stats, mahalanobis_file)
    logger.info("Saved Mahalanobis model stats to %s", mahalanobis_file)

    # Combine operational splits for reference buddy telemetry
    operational_ref = pd.concat([train_raw, val_raw, test_raw], axis=0, ignore_index=True)

    # 3. Engineer Features for All Splits
    logger.info("Engineering features for TRAIN split (%d rows)...", len(train_raw))
    train_eng = engineer_split_features(train_raw, neighbor_mapping, mahalanobis_stats, train_raw)

    logger.info("Engineering features for VAL split (%d rows)...", len(val_raw))
    val_eng = engineer_split_features(val_raw, neighbor_mapping, mahalanobis_stats, val_raw)

    logger.info("Engineering features for TEST split (%d rows)...", len(test_raw))
    test_eng = engineer_split_features(test_raw, neighbor_mapping, mahalanobis_stats, test_raw)

    logger.info("Engineering features for SPATIAL HOLDOUT split (%d rows)...", len(holdout_raw))
    holdout_eng = engineer_split_features(holdout_raw, neighbor_mapping, mahalanobis_stats, operational_ref)

    # 4. Fit Scaler ONLY on Normal Train Rows
    scaler = fit_feature_scaler(train_eng)
    scaler_file = models_path / "feature_scaler.joblib"
    joblib.dump(scaler, scaler_file)
    logger.info("Saved fitted feature scaler to %s", scaler_file)

    # 5. Apply Scaling to All Splits
    logger.info("Applying feature scaling across all splits...")
    train_scaled = apply_feature_scaling(train_eng, scaler)
    val_scaled = apply_feature_scaling(val_eng, scaler)
    test_scaled = apply_feature_scaling(test_eng, scaler)
    holdout_scaled = apply_feature_scaling(holdout_eng, scaler)

    # 6. Segregate Tier 1 Excluded Rows
    tier1_fault_types = ["communication_dropout", "data_corruption"]
    tier1_excluded_list: List[pd.DataFrame] = []

    split_dfs = {
        "train": train_scaled,
        "val": val_scaled,
        "test": test_scaled,
        "spatial_holdout": holdout_scaled,
    }

    final_splits: Dict[str, pd.DataFrame] = {}

    for split_name, s_df in split_dfs.items():
        is_tier1 = s_df["anomaly_type"].isin(tier1_fault_types) | s_df["temperature_c"].isna()
        excluded = s_df[is_tier1].copy()
        excluded["split"] = split_name
        tier1_excluded_list.append(excluded)

        retained = s_df[~is_tier1].copy()
        final_splits[split_name] = retained
        logger.info(
            "Split '%s': %d rows retained, %d rows excluded for Tier 1.",
            split_name,
            len(retained),
            len(excluded),
        )

    # Save Tier 1 Excluded Rows
    tier1_excluded_df = pd.concat(tier1_excluded_list, axis=0, ignore_index=True)
    tier1_excluded_file = features_path / "tier1_excluded_rows.parquet"
    tier1_excluded_df.to_parquet(tier1_excluded_file, index=False)
    logger.info("Saved Tier 1 excluded rows (%d rows) to %s", len(tier1_excluded_df), tier1_excluded_file)

    # Save Final Feature Splits
    for split_name, s_df in final_splits.items():
        out_file = features_path / f"{split_name}.parquet"
        s_df.to_parquet(out_file, index=False)
        logger.info("Saved %s features to %s (%d rows, %d cols)", split_name, out_file, len(s_df), len(s_df.columns))

    logger.info("Feature engineering pipeline completed successfully!")
    return final_splits


if __name__ == "__main__":
    run_feature_pipeline()
