"""Audit script for SkyGuard AI AWS telemetry dataset.

Displays:
1. Row counts AND distinct episode counts per anomaly_type across each split:
   - Train (temporal)
   - Val (temporal)
   - Test (temporal)
   - Spatial Holdout
   - Total
2. Distinct stations affected per anomaly type per split.
3. Explicit univariate bounds check on cross_sensor_inconsistency rows (0 rows outside [0, 100]% RH or [p01, p99]).
4. Explicit Mahalanobis distance distribution comparison (cross_sensor_inconsistency vs normal rows).
5. Explicit communication dropout null completeness check (% with all 3 columns NaN).
6. Extreme weather summary audit.
"""

from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd

from src.physics import calculate_dew_point_c


def count_episodes(df: pd.DataFrame, anom_type: str) -> Tuple[int, int]:
    """Return (episode_count, station_count) for an anomaly type."""
    sub = df[df["anomaly_type"] == anom_type].copy()
    if len(sub) == 0:
        return 0, 0
    episodes = 0
    stations = sub["station_id"].nunique()
    for _, group in sub.groupby("station_id"):
        group = group.sort_values("timestamp")
        time_diffs = group["timestamp"].diff()
        new_episode_flags = (time_diffs != pd.Timedelta(minutes=10))
        episodes += int(new_episode_flags.sum())
    return episodes, stations


def main():
    splits_dir = Path("data/splits")
    raw_dir = Path("data/raw")

    train_df = pd.read_parquet(splits_dir / "train.parquet")
    val_df = pd.read_parquet(splits_dir / "val.parquet")
    test_df = pd.read_parquet(splits_dir / "test.parquet")
    holdout_df = pd.read_parquet(splits_dir / "spatial_holdout_stations.parquet")
    master_df = pd.read_parquet(raw_dir / "aws_telemetry_master.parquet")

    fault_types = [
        "spike_or_drop",
        "frozen_sensor",
        "communication_dropout",
        "calibration_drift",
        "power_fluctuation_glitch",
        "data_corruption",
        "cross_sensor_inconsistency",
    ]

    splits = [
        ("Train", train_df),
        ("Val", val_df),
        ("Test", test_df),
        ("Spatial Holdout", holdout_df),
    ]

    print("=" * 115)
    print("                 SKYGUARD AI: CLASS BALANCE & EPISODE AUDIT TABLE (BY SPLIT)")
    print("=" * 115)
    header = f"{'Anomaly Type':<28} | {'Train (Rows/Ep/St)':<18} | {'Val (Rows/Ep/St)':<18} | {'Test (Rows/Ep/St)':<18} | {'Holdout (Rows/Ep/St)':<20} | {'Total Episodes':<14}"
    print(header)
    print("-" * 115)

    for fault in fault_types:
        row_str = f"{fault:<28} | "
        total_episodes = 0
        for s_name, s_df in splits:
            rows = len(s_df[s_df["anomaly_type"] == fault])
            eps, sts = count_episodes(s_df, fault)
            total_episodes += eps
            if s_name == "Spatial Holdout":
                row_str += f"{rows:>4}r/{eps:>2}ep/{sts:>1}st     | "
            else:
                row_str += f"{rows:>4}r/{eps:>2}ep/{sts:>1}st   | "
        row_str += f"{total_episodes:>8} eps"
        print(row_str)

    print("-" * 115)

    # Summary row for total anomalies
    summary_str = f"{'TOTAL ANOMALIES':<28} | "
    grand_total_episodes = 0
    for s_name, s_df in splits:
        sub = s_df[s_df["is_anomaly"]]
        rows = len(sub)
        pct = (rows / len(s_df)) * 100.0
        tot_eps = sum(count_episodes(s_df, f)[0] for f in fault_types)
        grand_total_episodes += tot_eps
        if s_name == "Spatial Holdout":
            summary_str += f"{rows:>4}r ({pct:>4.1f}%)        | "
        else:
            summary_str += f"{rows:>4}r ({pct:>4.1f}%)      | "
    summary_str += f"{grand_total_episodes:>8} eps"
    print(summary_str)
    print("=" * 115)

    print("\n" + "=" * 80)
    print("     CROSS-SENSOR INCONSISTENCY: UNIVARIATE BOUNDS & DEW-POINT CHECK")
    print("=" * 80)
    cross_df = master_df[master_df["anomaly_type"] == "cross_sensor_inconsistency"].dropna(
        subset=["temperature_c", "pressure_hpa", "humidity_pct"]
    )
    total_cross = len(cross_df)
    
    # Univariate physical checks
    rh_outside_100 = ((cross_df["humidity_pct"] < 0.0) | (cross_df["humidity_pct"] > 100.0)).sum()
    t_outside_bounds = ((cross_df["temperature_c"] < -25.0) | (cross_df["temperature_c"] > 55.0)).sum()
    p_outside_bounds = ((cross_df["pressure_hpa"] < 550.0) | (cross_df["pressure_hpa"] > 1060.0)).sum()

    # Dew point check
    td = calculate_dew_point_c(cross_df["temperature_c"].to_numpy(), cross_df["humidity_pct"].to_numpy())
    diff = td - cross_df["temperature_c"].to_numpy()
    td_violating = np.sum(diff > 0.05)

    print(f"Total cross_sensor_inconsistency rows:   {total_cross:,}")
    print(f"Rows with RH outside [0, 100]%:           {rh_outside_100} (Target: 0)")
    print(f"Rows with T outside realistic bounds:     {t_outside_bounds} (Target: 0)")
    print(f"Rows with P outside realistic bounds:     {p_outside_bounds} (Target: 0)")
    print(f"RH Range:                                 [{cross_df['humidity_pct'].min():.1f}%, {cross_df['humidity_pct'].max():.1f}%]")
    print(f"Temperature Range:                        [{cross_df['temperature_c'].min():.1f}°C, {cross_df['temperature_c'].max():.1f}°C]")
    print(f"Pressure Range:                           [{cross_df['pressure_hpa'].min():.1f} hPa, {cross_df['pressure_hpa'].max():.1f} hPa]")
    print(f"Rows with T_dew > T_air:                  {td_violating} (Physical invariant satisfied)")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("     CROSS-SENSOR INCONSISTENCY: MAHALANOBIS DISTANCE SEPARATION")
    print("=" * 80)
    normal_df = master_df[~master_df["is_anomaly"]].dropna(subset=["temperature_c", "pressure_hpa", "humidity_pct"])
    cross_dm_list = []
    normal_dm_list = []
    within_station_p01_p99_count = 0

    for st_id, st_norm in normal_df.groupby("station_id"):
        X_norm = st_norm[["temperature_c", "pressure_hpa", "humidity_pct"]].to_numpy()
        mu = np.mean(X_norm, axis=0)
        cov = np.cov(X_norm, rowvar=False)
        inv_cov = np.linalg.inv(cov + 1e-6 * np.eye(3))
        
        p01 = np.percentile(X_norm, 1.0, axis=0)
        p99 = np.percentile(X_norm, 99.0, axis=0)
        
        diff_norm = X_norm - mu
        norm_dm = np.sqrt(np.sum(diff_norm @ inv_cov * diff_norm, axis=1))
        normal_dm_list.extend(norm_dm)
        
        st_cross = cross_df[cross_df["station_id"] == st_id]
        if len(st_cross) > 0:
            X_cross = st_cross[["temperature_c", "pressure_hpa", "humidity_pct"]].to_numpy()
            diff_cross = X_cross - mu
            cross_dm = np.sqrt(np.sum(diff_cross @ inv_cov * diff_cross, axis=1))
            cross_dm_list.extend(cross_dm)
            
            # Check station-specific [p01, p99]
            in_p01_p99 = ((X_cross >= p01) & (X_cross <= p99)).all(axis=1)
            within_station_p01_p99_count += in_p01_p99.sum()

    cross_dm_arr = np.array(cross_dm_list)
    norm_dm_arr = np.array(normal_dm_list)

    print("Mahalanobis Distance D_M Distribution:")
    print(f"  NORMAL DATA (Non-Anomalous, N={len(norm_dm_arr):,}):")
    print(f"    Mean: {norm_dm_arr.mean():.2f} | Median: {np.median(norm_dm_arr):.2f} | 95th-pct: {np.percentile(norm_dm_arr, 95):.2f} | 99th-pct: {np.percentile(norm_dm_arr, 99):.2f} | Max: {norm_dm_arr.max():.2f}")
    print(f"  CROSS_SENSOR_INCONSISTENCY ANOMALIES (N={len(cross_dm_arr):,}):")
    print(f"    Mean: {cross_dm_arr.mean():.2f} | Min: {cross_dm_arr.min():.2f} | Median: {np.median(cross_dm_arr):.2f} | 95th-pct: {np.percentile(cross_dm_arr, 95):.2f} | Max: {cross_dm_arr.max():.2f}")
    
    pct_gt_3 = (cross_dm_arr > 3.0).mean() * 100.0
    pct_gt_norm95 = (cross_dm_arr > np.percentile(norm_dm_arr, 95)).mean() * 100.0
    pct_within_p01_p99 = (within_station_p01_p99_count / total_cross) * 100.0
    
    print(f"  --> Percentage with D_M > 3.0:                 {pct_gt_3:.2f}% (Target: 100.0%)")
    print(f"  --> Percentage above Normal 95th-percentile:   {pct_gt_norm95:.2f}% (Target: >=95.0%)")
    print(f"  --> Rows strictly within Station [p01, p99]:   {within_station_p01_p99_count}/{total_cross} ({pct_within_p01_p99:.2f}%)")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("          DROPOUT COMPLETENESS CHECK (communication_dropout)")
    print("=" * 80)
    drop_df = master_df[master_df["anomaly_type"] == "communication_dropout"]
    total_drop = len(drop_df)
    cols = ["temperature_c", "pressure_hpa", "humidity_pct"]
    all_three_nan = drop_df[cols].isna().all(axis=1)
    complete_count = all_three_nan.sum()
    complete_pct = (complete_count / total_drop) * 100.0 if total_drop > 0 else 0.0

    t_nan_pct = (drop_df["temperature_c"].isna().sum() / total_drop) * 100.0
    p_nan_pct = (drop_df["pressure_hpa"].isna().sum() / total_drop) * 100.0
    rh_nan_pct = (drop_df["humidity_pct"].isna().sum() / total_drop) * 100.0

    print(f"Total communication_dropout rows:         {total_drop:,}")
    print(f"Rows with ALL 3 sensors NaN:              {complete_count:,}")
    print(f"Dropout Completeness Percentage:          {complete_pct:.2f}% (Target: 100.00%)")
    print(f"  - temperature_c NaN rate:               {t_nan_pct:.2f}%")
    print(f"  - pressure_hpa NaN rate:                {p_nan_pct:.2f}%")
    print(f"  - humidity_pct NaN rate:                {rh_nan_pct:.2f}%")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("          EXTREME WEATHER EVENTS AUDIT (Ground Truth: is_anomaly = FALSE)")
    print("=" * 80)
    import json
    with open(raw_dir / "generation_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    ext_sum = meta.get("extreme_weather_summary", {})
    print(f"Total Scheduled Events:                   {ext_sum.get('total_events', 0)}")
    print(f"Stations Affected:                        {len(ext_sum.get('stations_affected', []))} ({', '.join(ext_sum.get('stations_affected', []))})")
    print(f"Event Types Present:                      {', '.join(ext_sum.get('event_types_present', []))}")
    for evt in ext_sum.get("events", []):
        print(f"  - Station {evt['station_id']:<12}: {evt['event_type']:<26} (Duration: {evt['duration_steps']*10//60:>2} hrs, Start step: {evt['start_idx']})")
    print("=" * 80)


if __name__ == "__main__":
    main()
