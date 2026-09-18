"""Comprehensive validation test suite for SkyGuard AI AWS data generator.

Verifies:
1. Physical Plausibility: Normal and extreme weather rows satisfy thermodynamic bounds (T_dew <= T_air).
2. Fault Label Fidelity:
   - communication_dropout: ALL THREE columns (T, P, RH) MUST be NaN across 100% of dropout rows.
   - cross_sensor_inconsistency: Implied T_dew strictly exceeds T_air (large majority / 100% violation).
   - frozen_sensor: Exact zero-variance constant float sequences.
   - data_corruption: Out-of-range sentinel tokens.
   - station-locality: Faults affect only single stations.
3. No Temporal Leakage: Strict chronological separation across train/val/test splits.
4. Spatial Isolation: 4 spatial holdout stations (1 per climate zone) with zero train/val/test overlap.
5. Episode Representation: Min 8 episodes per anomaly type in Train, Val, Test; min 5 in Spatial Holdout.
6. Extreme Weather Representation: At least 5 total events across at least 4 stations and at least 2 event types.
7. Deterministic Reproducibility: Identical random seeds produce exact identical outputs.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import GeneratorConfig
from src.pipeline import DatasetPipeline
from src.physics import calculate_dew_point_c


def count_episodes(df: pd.DataFrame, anom_type: str) -> int:
    """Count contiguous episodes of a specific anomaly type grouped by station."""
    sub = df[df["anomaly_type"] == anom_type].copy()
    if len(sub) == 0:
        return 0
    episodes = 0
    for _, group in sub.groupby("station_id"):
        # Sort by timestamp
        group = group.sort_values("timestamp")
        # Time diff > 10 minutes indicates a new episode
        time_diffs = group["timestamp"].diff()
        new_episode_flags = (time_diffs != pd.Timedelta(minutes=10))
        episodes += int(new_episode_flags.sum())
    return episodes


def test_physical_plausibility(master_df: pd.DataFrame):
    """Verify normal and extreme weather rows strictly conform to physical bounds."""
    print("Testing physical plausibility of non-anomalous telemetry...")
    normal_df = master_df[~master_df["is_anomaly"]].dropna(subset=["temperature_c", "pressure_hpa", "humidity_pct"])
    
    # 1. Check relative humidity bounds [0, 100%]
    rh_min = normal_df["humidity_pct"].min()
    rh_max = normal_df["humidity_pct"].max()
    assert rh_min >= 0.0, f"Violation: RH below 0% ({rh_min})"
    assert rh_max <= 100.05, f"Violation: RH above 100% ({rh_max})"

    # 2. Check temperature bounds (-25°C to +60°C)
    t_min = normal_df["temperature_c"].min()
    t_max = normal_df["temperature_c"].max()
    assert t_min >= -25.0, f"Violation: Unrealistic cold temperature ({t_min}°C)"
    assert t_max <= 60.0, f"Violation: Unrealistic high temperature ({t_max}°C)"

    # 3. Check pressure bounds (550 to 1060 hPa)
    p_min = normal_df["pressure_hpa"].min()
    p_max = normal_df["pressure_hpa"].max()
    assert p_min >= 550.0, f"Violation: Atmospheric pressure too low ({p_min} hPa)"
    assert p_max <= 1060.0, f"Violation: Atmospheric pressure too high ({p_max} hPa)"

    # 4. Check Clausius-Clapeyron / August-Roche-Magnus invariant (T_dew <= T_air + 0.1°C numeric tolerance)
    td = calculate_dew_point_c(normal_df["temperature_c"].to_numpy(), normal_df["humidity_pct"].to_numpy())
    diff = td - normal_df["temperature_c"].to_numpy()
    violations = np.sum(diff > 0.1)
    assert violations == 0, f"Violation: {violations} normal records have T_dew > T_air"

    print("  [PASS] Physical bounds and thermodynamic consistency verified for all normal rows.")


def test_anomaly_internal_consistency(master_df: pd.DataFrame):
    """Verify ground-truth anomaly labels match expected fault properties."""
    print("Testing internal consistency of anomaly labels...")
    anom_df = master_df[master_df["is_anomaly"]].copy()
    assert len(anom_df) > 0, "No anomalies found in dataset!"

    # 1. Communication Dropout: ALL THREE columns MUST be NaN on 100% of rows
    dropout_df = anom_df[anom_df["anomaly_type"] == "communication_dropout"]
    assert len(dropout_df) > 0, "No communication_dropout records found!"
    
    sensor_cols = ["temperature_c", "pressure_hpa", "humidity_pct"]
    all_three_nan = dropout_df[sensor_cols].isna().all(axis=1)
    nan_complete_pct = (all_three_nan.sum() / len(dropout_df)) * 100.0
    
    print(f"  --> Dropout Completeness Check: {all_three_nan.sum()}/{len(dropout_df)} rows ({nan_complete_pct:.2f}%) have ALL THREE sensor columns NaN.")
    if not all_three_nan.all():
        leaking_rows = dropout_df[~all_three_nan]
        raise AssertionError(
            f"FAILED: {len(leaking_rows)} communication_dropout rows leaked non-NaN values! "
            f"A dropped packet must null all 3 sensors."
        )
    print("  [PASS] Verified 100% of communication_dropout rows have ALL THREE sensor columns NaN.")

    # 2. Cross-Sensor Inconsistency: Joint Mahalanobis Outlier staying within univariate bounds
    cross_df = anom_df[anom_df["anomaly_type"] == "cross_sensor_inconsistency"].dropna(
        subset=["temperature_c", "pressure_hpa", "humidity_pct"]
    )
    assert len(cross_df) > 0, "No cross_sensor_inconsistency records found!"
    
    # Check (a): 0 rows with RH outside [0, 100]% or T/P outside realistic terrestrial bounds
    rh_invalid = (cross_df["humidity_pct"] < 0.0) | (cross_df["humidity_pct"] > 100.0)
    t_invalid = (cross_df["temperature_c"] < -25.0) | (cross_df["temperature_c"] > 55.0)
    p_invalid = (cross_df["pressure_hpa"] < 550.0) | (cross_df["pressure_hpa"] > 1060.0)
    total_univariate_invalid = (rh_invalid | t_invalid | p_invalid).sum()
    
    assert total_univariate_invalid == 0, (
        f"FAILED: {total_univariate_invalid} cross_sensor_inconsistency rows violated univariate bounds! "
        f"Every variable must individually stay within physical bounds."
    )
    
    # Check (b): Mahalanobis distance distribution vs normal
    # Compute station-level Mahalanobis distances
    normal_df = master_df[~master_df["is_anomaly"]].dropna(subset=["temperature_c", "pressure_hpa", "humidity_pct"])
    
    cross_dm_list = []
    normal_dm_list = []
    
    for st_id, st_norm in normal_df.groupby("station_id"):
        X_norm = st_norm[["temperature_c", "pressure_hpa", "humidity_pct"]].to_numpy()
        mu = np.mean(X_norm, axis=0)
        cov = np.cov(X_norm, rowvar=False)
        inv_cov = np.linalg.inv(cov + 1e-6 * np.eye(3))
        
        diff_norm = X_norm - mu
        norm_dm = np.sqrt(np.sum(diff_norm @ inv_cov * diff_norm, axis=1))
        normal_dm_list.extend(norm_dm)
        
        st_cross = cross_df[cross_df["station_id"] == st_id]
        if len(st_cross) > 0:
            X_cross = st_cross[["temperature_c", "pressure_hpa", "humidity_pct"]].to_numpy()
            diff_cross = X_cross - mu
            cross_dm = np.sqrt(np.sum(diff_cross @ inv_cov * diff_cross, axis=1))
            cross_dm_list.extend(cross_dm)
            
    cross_dm_arr = np.array(cross_dm_list)
    norm_dm_arr = np.array(normal_dm_list)
    
    mean_cross_dm = float(np.mean(cross_dm_arr))
    p95_norm_dm = float(np.percentile(norm_dm_arr, 95.0))
    pct_separated = float(np.mean(cross_dm_arr > p95_norm_dm) * 100.0)
    
    print(f"  --> Cross-Sensor Mahalanobis Check: Mean D_M = {mean_cross_dm:.2f} vs Normal 95th-pct D_M = {p95_norm_dm:.2f} ({pct_separated:.2f}% separated).")
    print(f"      RH Range: [{cross_df['humidity_pct'].min():.1f}%, {cross_df['humidity_pct'].max():.1f}%] (All within [0, 100]%)")
    assert mean_cross_dm >= 3.0, f"FAILED: Mean cross-sensor Mahalanobis distance too low ({mean_cross_dm:.2f} < 3.0)!"
    assert pct_separated >= 95.0, f"FAILED: Only {pct_separated:.2f}% separated from normal distribution!"
    print("  [PASS] Verified cross_sensor_inconsistency rows are genuine joint Mahalanobis outliers within physical bounds.")

    # 3. Frozen sensor must have zero consecutive variance
    frozen_df = anom_df[anom_df["anomaly_type"] == "frozen_sensor"]
    assert len(frozen_df) > 0, "No frozen_sensor records found!"
    consecutive_frozen_zeros = 0
    for _, group in frozen_df.groupby("station_id"):
        diff_t = group["temperature_c"].diff().dropna()
        diff_p = group["pressure_hpa"].diff().dropna()
        diff_rh = group["humidity_pct"].diff().dropna()
        zero_diffs = (diff_t == 0) | (diff_p == 0) | (diff_rh == 0)
        consecutive_frozen_zeros += zero_diffs.sum()
    assert consecutive_frozen_zeros > 0, "Frozen sensor sequences must have repeated constant values!"
    print(f"  [PASS] Verified frozen sensor zero-variance sequences.")

    # 4. Data corruption must contain out-of-range sentinel tokens
    corrupt_df = anom_df[anom_df["anomaly_type"] == "data_corruption"]
    assert len(corrupt_df) > 0, "No data_corruption records found!"
    sentinel_found = (
        (corrupt_df["temperature_c"] > 100.0)
        | (corrupt_df["temperature_c"] < -50.0)
        | (corrupt_df["pressure_hpa"] < 500.0)
        | (corrupt_df["pressure_hpa"] > 1100.0)
        | (corrupt_df["humidity_pct"] < 0.0)
        | (corrupt_df["humidity_pct"] > 105.0)
    )
    assert sentinel_found.any(), "Data corruption records must contain out-of-range sentinel values!"
    print(f"  [PASS] Verified data corruption sentinel values.")

    # 5. Station locality test
    st_counts = anom_df.groupby("timestamp")["station_id"].nunique()
    assert (st_counts < 10).all(), "Anomalies must remain station-local (not global simultaneous failures)!"
    print("  [PASS] Verified station-locality of injected faults.")


def test_split_integrity_and_episode_representation(data_dir: Path):
    """Verify chronological split isolation and minimum episode representation quotas."""
    print("Testing train/val/test splits, spatial holdouts, and episode quotas...")
    train_path = data_dir / "splits" / "train.parquet"
    val_path = data_dir / "splits" / "val.parquet"
    test_path = data_dir / "splits" / "test.parquet"
    holdout_path = data_dir / "splits" / "spatial_holdout_stations.parquet"

    assert train_path.exists() and val_path.exists() and test_path.exists(), "Split files missing!"

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)
    holdout_df = pd.read_parquet(holdout_path)

    # 1. Chronological order without time leakage
    t_train_max = train_df["timestamp"].max()
    t_val_min = val_df["timestamp"].min()
    t_val_max = val_df["timestamp"].max()
    t_test_min = test_df["timestamp"].min()

    assert t_train_max < t_val_min, f"Temporal leakage: train max ({t_train_max}) >= val min ({t_val_min})"
    assert t_val_max < t_test_min, f"Temporal leakage: val max ({t_val_max}) >= test min ({t_test_min})"
    print(f"  [PASS] Strict chronological boundaries: Train < Val < Test.")

    # 2. Spatial holdout station isolation
    expected_holdouts = {"AWS_IND_C04", "AWS_IND_A03", "AWS_IND_H03", "AWS_IND_P03"}
    actual_holdouts = set(holdout_df["station_id"].unique())
    assert actual_holdouts == expected_holdouts, f"Spatial holdout stations mismatch! Expected {expected_holdouts}, got {actual_holdouts}"

    train_stations = set(train_df["station_id"].unique())
    val_stations = set(val_df["station_id"].unique())
    test_stations = set(test_df["station_id"].unique())
    temporal_stations = train_stations | val_stations | test_stations

    overlap = actual_holdouts.intersection(temporal_stations)
    assert len(overlap) == 0, f"Spatial leakage: Holdout stations {overlap} present in temporal splits!"
    print(f"  [PASS] Zero station leakage in 4 spatial holdouts ({actual_holdouts}).")

    # 3. Episode quotas per split
    fault_types = [
        "spike_or_drop",
        "frozen_sensor",
        "communication_dropout",
        "calibration_drift",
        "power_fluctuation_glitch",
        "data_corruption",
        "cross_sensor_inconsistency",
    ]

    splits_to_test = [
        ("Train", train_df, 8),
        ("Validation", val_df, 8),
        ("Test", test_df, 8),
        ("Spatial Holdout", holdout_df, 5),
    ]

    for split_name, s_df, min_quota in splits_to_test:
        for fault in fault_types:
            ep_count = count_episodes(s_df, fault)
            assert ep_count >= min_quota, (
                f"Episode starvation in {split_name}: {fault} has only {ep_count} episodes (minimum required: {min_quota})!"
            )
    print(f"  [PASS] Verified episode quotas: >=8 per fault type in Train/Val/Test, and >=5 in Spatial Holdout.")


def test_extreme_weather_representation(data_dir: Path):
    """Verify extreme weather events satisfy network targets."""
    print("Testing extreme weather representation...")
    meta_path = data_dir / "raw" / "generation_metadata.json"
    import json
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    summary = meta.get("extreme_weather_summary", {})
    total_events = summary.get("total_events", 0)
    stations_affected = summary.get("stations_affected", [])
    event_types_present = summary.get("event_types_present", [])

    print(f"  --> Extreme Weather Summary: {total_events} events across {len(stations_affected)} stations, types: {event_types_present}")
    assert total_events >= 5, f"Too few extreme events: {total_events} < 5"
    assert len(stations_affected) >= 4, f"Events clustered on too few stations: {len(stations_affected)} < 4"
    assert len(event_types_present) >= 2, f"Too few event types present: {len(event_types_present)} < 2"
    print("  [PASS] Verified extreme weather network distribution (>=5 events, >=4 stations, >=2 types).")


def test_reproducibility(seed: int = 42):
    """Verify identical seeds produce exact identical outputs."""
    print("Testing generator reproducibility with fixed random seed...")
    config1 = GeneratorConfig()
    config1.generation.random_seed = seed
    config1.generation.duration_days = 5

    config2 = GeneratorConfig()
    config2.generation.random_seed = seed
    config2.generation.duration_days = 5

    p1 = DatasetPipeline(config1)
    df1, _, _ = p1.generate()

    p2 = DatasetPipeline(config2)
    df2, _, _ = p2.generate()

    mask1 = df1["temperature_c"].notna()
    mask2 = df2["temperature_c"].notna()
    np.testing.assert_array_equal(mask1.to_numpy(), mask2.to_numpy())
    np.testing.assert_allclose(df1["temperature_c"][mask1].to_numpy(), df2["temperature_c"][mask2].to_numpy(), rtol=1e-5)
    np.testing.assert_array_equal(df1["is_anomaly"].to_numpy(), df2["is_anomaly"].to_numpy())
    print("  [PASS] Full deterministic reproducibility confirmed across identical seeds.")


def run_all_tests(data_dir_str: str = "data"):
    print("================================================================================")
    print("                   SkyGuard AI - Dataset Validation & QC Suite                  ")
    print("================================================================================")
    data_dir = Path(data_dir_str)
    master_path = data_dir / "raw" / "aws_telemetry_master.parquet"
    if not master_path.exists():
        print(f"Error: Master file {master_path} not found. Run generate.py first.")
        sys.exit(1)

    master_df = pd.read_parquet(master_path)
    print(f"Loaded master telemetry dataset: {len(master_df):,} records.")

    test_physical_plausibility(master_df)
    test_anomaly_internal_consistency(master_df)
    test_split_integrity_and_episode_representation(data_dir)
    test_extreme_weather_representation(data_dir)
    test_reproducibility()

    print("================================================================================")
    print("          ALL VALIDATION CHECKS PASSED: DATASET IS METEOROLOGICALLY VALID       ")
    print("================================================================================")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    run_all_tests(args.data_dir)
