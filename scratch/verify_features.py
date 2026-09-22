"""Verification script for SkyGuard AI Feature Engineering Pipeline."""

import json
import math
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.features import (
    ENGINEERED_FEATURE_COLS,
    SENSOR_COLS,
    compute_haversine_distance_km,
)
from src.physics import calculate_dew_point_c, calculate_saturation_vapor_pressure_hpa


def run_verification():
    print("=" * 80)
    print("SKYGUARD AI FEATURE ENGINEERING VERIFICATION")
    print("=" * 80)

    # 1. Load Parquet Files & Fitted Artifacts
    train_feat = pd.read_parquet("data/features/train.parquet")
    val_feat = pd.read_parquet("data/features/val.parquet")
    test_feat = pd.read_parquet("data/features/test.parquet")
    holdout_feat = pd.read_parquet("data/features/spatial_holdout.parquet")
    tier1_excluded = pd.read_parquet("data/features/tier1_excluded_rows.parquet")

    scaler: StandardScaler = joblib.load("models/feature_scaler.joblib")
    mahalanobis_stats = joblib.load("models/mahalanobis_stats.joblib")
    with open("models/spatial_neighbors.json", "r", encoding="utf-8") as f:
        spatial_neighbors = json.load(f)

    # CHECK 1: Confirm zero NaN in any *_scaled column for rows NOT in tier1_excluded
    print("\n--- CHECK 1: ZERO NaN VERIFICATION IN SCALED COLUMNS ---")
    scaled_cols = [f"{c}_scaled" for c in ENGINEERED_FEATURE_COLS]
    all_zero_nan = True
    for name, df in [
        ("train", train_feat),
        ("val", val_feat),
        ("test", test_feat),
        ("spatial_holdout", holdout_feat),
    ]:
        nan_counts = df[scaled_cols].isna().sum()
        total_nans = nan_counts.sum()
        print(f"[{name}] Rows: {len(df)}, Total NaNs in {len(scaled_cols)} scaled cols: {total_nans}")
        if total_nans > 0:
            print(f"  WARNING: NaNs found in {name}:\n{nan_counts[nan_counts > 0]}")
            all_zero_nan = False
    assert all_zero_nan, "Failed Check 1: Found NaNs in scaled columns!"
    print(">>> CHECK 1 PASSED: Zero NaNs across all scaled columns in all splits.")

    # CHECK 2: Confirm scaler fit sample count matches known normal train rows
    print("\n--- CHECK 2: SCALER SAMPLE COUNT & LEAKAGE CHECK ---")
    train_raw = pd.read_parquet("data/splits/train.parquet")
    known_normal_rows = (train_raw["is_anomaly"] == False).sum()
    print(f"Known train normal rows: {known_normal_rows}")
    print(f"Scaler n_features_in_: {scaler.n_features_in_}")
    print(f"Scaler feature names: {scaler.feature_names_in_.tolist()}")
    print(f"Scaler n_samples_seen_: {scaler.n_samples_seen_}")
    assert (
        scaler.n_samples_seen_ == known_normal_rows
    ), f"Mismatch! Scaler fitted on {scaler.n_samples_seen_}, expected {known_normal_rows}"
    print(f">>> CHECK 2 PASSED: Scaler strictly fitted on exactly {known_normal_rows} normal train rows.")

    # CHECK 3: Hand-Calculation Spot Check on 3 Random Train Rows
    print("\n--- CHECK 3: HAND-CALCULATION SPOT CHECK ON 3 RANDOM ROWS ---")
    np.random.seed(42)
    # Pick 3 random indices from train features where is_anomaly == False
    normal_indices = train_feat[train_feat["is_anomaly"] == False].index.to_numpy()
    sample_indices = np.random.choice(normal_indices, size=3, replace=False)

    for i, idx in enumerate(sample_indices, 1):
        row = train_feat.loc[idx]
        st_id = row["station_id"]
        ts = pd.to_datetime(row["timestamp"])
        t_val = float(row["temperature_c"])
        rh_val = float(row["humidity_pct"])
        p_val = float(row["pressure_hpa"])
        hour = ts.hour

        print(f"\n[Spot Check Sample {i}] Station: {st_id}, Timestamp: {ts}")
        print(f"  Inputs: T = {t_val:.4f} °C, RH = {rh_val:.4f} %, P = {p_val:.4f} hPa")

        # 1. Dew point depression
        # August-Roche-Magnus formula
        rh_safe = max(rh_val, 0.1)
        es_hand = 6.112 * np.exp((17.67 * t_val) / (t_val + 243.5))
        e_hand = es_hand * (rh_safe / 100.0)
        gamma_hand = np.log(e_hand / 6.112)
        td_hand = (243.5 * gamma_hand) / (17.67 - gamma_hand)
        dew_dep_hand = max(0.0, t_val - td_hand)
        dew_dep_file = float(row["dew_point_dep"])
        print(f"  Dew Point Depression:")
        print(f"    Hand formula: T - T_dew = {t_val:.4f} - {td_hand:.4f} = {dew_dep_hand:.6f}")
        print(f"    Dataset file: {dew_dep_file:.6f}")
        diff_dep = abs(dew_dep_hand - dew_dep_file)
        print(f"    Difference: {diff_dep:.2e}")
        assert diff_dep < 1e-5, f"Dew point depression mismatch on sample {i}!"

        # 2. Delta T
        # Lookup previous row for same station in train_raw
        st_rows = train_raw[train_raw["station_id"] == st_id].sort_values("timestamp")
        curr_loc = st_rows[st_rows["timestamp"] == row["timestamp"]].index[0]
        pos = st_rows.index.get_loc(curr_loc)
        if pos > 0:
            prev_row = st_rows.iloc[pos - 1]
            prev_t = float(prev_row["temperature_c"])
            delta_t_hand = t_val - prev_t
        else:
            delta_t_hand = 0.0
        delta_t_file = float(row["delta_T"])
        print(f"  Delta T (First Difference):")
        print(f"    Hand formula: T_t - T_{{t-1}} = {t_val:.4f} - {prev_t:.4f} = {delta_t_hand:.6f}")
        print(f"    Dataset file: {delta_t_file:.6f}")
        diff_dt = abs(delta_t_hand - delta_t_file)
        print(f"    Difference: {diff_dt:.2e}")
        assert diff_dt < 1e-5, f"Delta T mismatch on sample {i}!"

        # 3. Mahalanobis Distance
        bucket = mahalanobis_stats["station_hour"][(st_id, hour)]
        mu = bucket["mu"]
        inv_cov = bucket["inv_cov"]
        x_vec = np.array([t_val, p_val, rh_val], dtype=np.float64)
        diff_vec = x_vec - mu
        dm_sq_hand = float(diff_vec @ inv_cov @ diff_vec)
        dm_hand = math.sqrt(max(0.0, dm_sq_hand))
        dm_file = float(row["mahalanobis_dist"])
        print(f"  Mahalanobis Distance D_M:")
        print(f"    mu vector: [{mu[0]:.2f}, {mu[1]:.2f}, {mu[2]:.2f}]")
        print(f"    diff vector: [{diff_vec[0]:.2f}, {diff_vec[1]:.2f}, {diff_vec[2]:.2f}]")
        print(f"    Hand formula sqrt(diff @ inv_cov @ diff): {dm_hand:.6f}")
        print(f"    Dataset file: {dm_file:.6f}")
        diff_dm = abs(dm_hand - dm_file)
        print(f"    Difference: {diff_dm:.2e}")
        assert diff_dm < 1e-5, f"Mahalanobis distance mismatch on sample {i}!"

    print("\n>>> CHECK 3 PASSED: All 3 spot-checked rows match hand calculations with zero error (<1e-5).")

    # CHECK 4: Feature Value Ranges per Split (Sanity Check)
    print("\n--- CHECK 4: FEATURE VALUE RANGES PER SPLIT ---")
    for name, df in [
        ("train", train_feat),
        ("val", val_feat),
        ("test", test_feat),
        ("spatial_holdout", holdout_feat),
    ]:
        print(f"\n[{name.upper()} SPLIT - Raw & Scaled Feature Summary]")
        summary_cols = [
            "temperature_c",
            "delta_T",
            "delta_T_scaled",
            "dew_point_dep",
            "dew_point_dep_scaled",
            "vpd",
            "vpd_scaled",
            "mahalanobis_dist",
            "mahalanobis_dist_scaled",
            "delta_T_buddy",
            "delta_T_buddy_scaled",
            "delta_P_cluster",
            "delta_P_cluster_scaled",
        ]
        stats_table = []
        for col in summary_cols:
            vals = df[col]
            stats_table.append(
                {
                    "Feature": col,
                    "Min": f"{vals.min():.3f}",
                    "Mean": f"{vals.mean():.3f}",
                    "Max": f"{vals.max():.3f}",
                    "Std": f"{vals.std():.3f}",
                }
            )
        print(pd.DataFrame(stats_table).to_string(index=False))

    # CHECK 5: Excluded Tier 1 breakdown
    print("\n--- CHECK 5: TIER 1 EXCLUDED ROWS BREAKDOWN ---")
    print(f"Total Tier 1 Excluded rows: {len(tier1_excluded)}")
    print(tier1_excluded.groupby(["split", "anomaly_type"]).size().to_frame("count"))

    print("\n" + "=" * 80)
    print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_verification()
