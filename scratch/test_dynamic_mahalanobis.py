import sys
import numpy as np
import pandas as pd
sys.path.insert(0, ".")
from src.physics import calculate_dew_point_c

df = pd.read_parquet("data/raw/aws_telemetry_master.parquet")

for st_id in ["AWS_IND_C01", "AWS_IND_A01", "AWS_IND_H01", "AWS_IND_P01"]:
    st = df[(df["station_id"] == st_id) & (~df["is_anomaly"])]
    X = st[["temperature_c", "pressure_hpa", "humidity_pct"]].to_numpy()
    mu = np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)
    inv_cov = np.linalg.inv(cov + 1e-6 * np.eye(3))

    p01 = np.percentile(X, 1, axis=0)
    p99 = np.percentile(X, 99, axis=0)

    # Pick a random 18-step window
    t_chunk = X[100:118, 0].copy()
    p_chunk = X[100:118, 1].copy()
    rh_chunk = X[100:118, 2].copy()

    # Shift towards joint outlier (Mode 1: High T + High RH, breaking anti-correlation)
    # Target values within [p80, p95]
    t_target = np.percentile(X[:, 0], 88)
    p_target = np.percentile(X[:, 1], 20)
    rh_target = np.percentile(X[:, 2], 88)

    t_mod = np.clip(0.6 * t_chunk + 0.4 * t_target + np.random.normal(0, 0.15, 18), p01[0] + 0.5, p99[0] - 0.5)
    p_mod = np.clip(0.6 * p_chunk + 0.4 * p_target + np.random.normal(0, 0.2, 18), p01[1] + 0.5, p99[1] - 0.5)
    rh_mod = np.clip(0.5 * rh_chunk + 0.5 * rh_target + np.random.normal(0, 0.4, 18), max(p01[2] + 1.0, 10.0), min(p99[2] - 1.0, 95.0))

    mod_X = np.column_stack([t_mod, p_mod, rh_mod])
    diffs = mod_X - mu
    dists = np.sqrt(np.sum(diffs @ inv_cov * diffs, axis=1))

    td = calculate_dew_point_c(t_mod, rh_mod)
    td_viol = np.sum(td > t_mod + 0.05)
    within_bounds = ((mod_X >= p01) & (mod_X <= p99)).all()
    rh_bounds = ((rh_mod >= 0.0) & (rh_mod <= 100.0)).all()
    has_zeros_var = (np.diff(t_mod) == 0).any() or (np.diff(rh_mod) == 0).any()

    print(f"Station {st_id}:")
    print(f"  Min DM in episode: {dists.min():.2f}, Mean DM: {dists.mean():.2f}, Max DM: {dists.max():.2f}")
    print(f"  All points in [p01, p99]: {within_bounds}")
    print(f"  RH strictly in [0, 100]: {rh_bounds} (min={rh_mod.min():.1f}%, max={rh_mod.max():.1f}%)")
    print(f"  T_dew > T_air count: {td_viol} (should be 0)")
    print(f"  Zero consecutive variance: {has_zeros_var} (should be False)")
