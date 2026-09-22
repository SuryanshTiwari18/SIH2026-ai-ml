import pandas as pd
import json

splits = ['train', 'val', 'test', 'spatial_holdout']
fault_types = [
    "spike_or_drop",
    "frozen_sensor",
    "communication_dropout",
    "calibration_drift",
    "power_fluctuation_glitch",
    "data_corruption",
    "cross_sensor_inconsistency",
]

def count_episodes(df: pd.DataFrame, anom_type: str):
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

print("=" * 80)
print("AUDIT REPORT: VOLATILITY BASELINE & GAP EDGE RECOVERY")
print("=" * 80)

# 1. Gap Edge Excluded Breakdown
gap_df = pd.read_parquet("data/features/gap_edge_excluded_rows.parquet")
print(f"\n1. Gap Edge Excluded Rows:")
print(f"   Total rows: {len(gap_df):,d} (Previous: 13,267 -> {((13267 - len(gap_df))/13267)*100:.1f}% reduction!)")
print(f"   Anomalous rows: {gap_df['is_anomaly'].sum():,d} (Previous: 833 -> {((833 - gap_df['is_anomaly'].sum())/833)*100:.1f}% reduction!)")
print(f"   Normal rows: {(gap_df['is_anomaly'] == False).sum():,d} (Previous: 12,434)")
print("\n   Anomaly Type Breakdown in Gap Edge Excluded:")
print(gap_df['anomaly_type'].value_counts(dropna=False).to_frame("count"))

# 2. Episode Counts per Split Table
print("\n2. Distinct Episode Counts per Split in Retained Feature Data:")
retained_dfs = {s: pd.read_parquet(f"data/features/{s}.parquet") for s in splits}

header = f"{'Anomaly Type':<28} | {'Train':<15} | {'Val':<15} | {'Test':<15} | {'Holdout':<15} | {'Target Balance':<14}"
print("-" * 110)
print(header)
print("-" * 110)

for fault in fault_types:
    ep_strs = []
    for s in splits:
        ep, st = count_episodes(retained_dfs[s], fault)
        ep_strs.append(f"{ep} ep ({st} st)")
    target = "0/0/0/0 (Tier 1)" if fault in ["communication_dropout", "data_corruption"] else "8 / 8 / 8 / 5"
    print(f"{fault:<28} | {ep_strs[0]:<15} | {ep_strs[1]:<15} | {ep_strs[2]:<15} | {ep_strs[3]:<15} | {target:<14}")
print("-" * 110)

# 3. Volatility Ratio Distribution on Train Normal Rows
print("\n3. Volatility Ratio Distribution on Train Normal Rows:")
train_df = retained_dfs["train"]
norm_train = train_df[train_df["is_anomaly"] == False]

for vr_col in ["vol_ratio_T", "vol_ratio_P", "vol_ratio_RH"]:
    r = norm_train[vr_col]
    s = norm_train[f"{vr_col}_scaled"]
    print(f"   {vr_col:<14}: Min={r.min():.4f}, Mean={r.mean():.4f}, Max={r.max():.4f}, Std={r.std():.4f} | Scaled: Min={s.min():.4f}, Mean={s.mean():.4f}, Max={s.max():.4f}, Std={s.std():.4f}")

# 4. Check glitch surge response
glitch_train = train_df[train_df["anomaly_type"] == "power_fluctuation_glitch"]
print("\n   Volatility Ratio Response during Power Fluctuation Glitch (Train):")
for vr_col in ["vol_ratio_T", "vol_ratio_P", "vol_ratio_RH"]:
    r = glitch_train[vr_col]
    s = glitch_train[f"{vr_col}_scaled"]
    print(f"   {vr_col:<14}: Min={r.min():.4f}, Mean={r.mean():.4f} ({r.mean()/norm_train[vr_col].mean():.1f}x surge), Max={r.max():.4f} | Scaled Mean={s.mean():.4f}")

print("\n" + "=" * 80)
