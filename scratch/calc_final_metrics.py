import json
import pandas as pd
import numpy as np

# 1. Load splits and verify canonical sizes
df_train = pd.read_parquet("data/tier4_results/train.parquet")
df_val = pd.read_parquet("data/tier4_results/val.parquet")
df_test = pd.read_parquet("data/tier4_results/test.parquet")
df_hold = pd.read_parquet("data/tier4_results/spatial_holdout.parquet")

print("Canonical split sizes in tier4_results:")
print(f"  train: {len(df_train)} (expected 72576)")
print(f"  val:   {len(df_val)} (expected 15552)")
print(f"  test:  {len(df_test)} (expected 15552)")
print(f"  hold:  {len(df_hold)} (expected 34560)")

assert len(df_train) == 72576, f"Expected 72576, got {len(df_train)}"
assert len(df_val) == 15552, f"Expected 15552, got {len(df_val)}"
assert len(df_test) == 15552, f"Expected 15552, got {len(df_test)}"
assert len(df_hold) == 34560, f"Expected 34560, got {len(df_hold)}"
print("ALL SPLIT SIZES MATCH CANONICAL SIZES EXACTLY!")

# Check columns
print("\nTier 4 columns:")
print(df_test.columns.tolist())

# 2. Check 5-Event Extreme Weather Audit
with open("data/raw/generation_metadata.json", "r") as f:
    meta = json.load(f)

events = meta["extreme_weather_summary"]["events"]
print(f"\nTotal extreme weather events: {len(events)}")

# In which split do these events live?
# Let's concatenate all splits or look up by station_id and start_idx / timestamp
# All 138,240 rows can be matched from the raw dataset or by finding the timestamps in each split
df_all = pd.concat([df_train, df_val, df_test, df_hold], ignore_index=True)
print(f"Combined total rows: {len(df_all)}")

# Let's read raw splits or timestamps to find each event
df_raw_train = pd.read_parquet("data/splits/train.parquet")
df_raw_val = pd.read_parquet("data/splits/val.parquet")
df_raw_test = pd.read_parquet("data/splits/test.parquet")
df_raw_hold = pd.read_parquet("data/splits/spatial_holdout_stations.parquet")
df_raw_all = pd.concat([df_raw_train, df_raw_val, df_raw_test, df_raw_hold], ignore_index=True)

# Let's inspect event details
print("\n--- 5-EVENT EXTREME WEATHER AUDIT DETAILS ---")
total_event_steps = 0
total_t1_fp = 0
total_t2_fp = 0

for i, ev in enumerate(events, 1):
    st = ev["station_id"]
    etype = ev["event_type"]
    dur = ev["duration_steps"]
    s_idx = ev["start_idx"]
    total_event_steps += dur
    
    # In df_all, station records are sorted by timestamp
    st_df = df_all[df_all["station_id"] == st].sort_values("timestamp").reset_index(drop=True)
    ev_slice = st_df.iloc[s_idx : s_idx + dur]
    
    # Check flags in ev_slice
    # Hard rule flagged (Track 1)
    hr_flags = ev_slice["hard_rule_flagged"].sum()
    # Learned classifier flags (Track 2)
    # Check what columns exist for classifier predictions
    lgbm_flags = (ev_slice["fusion_pred"] != "normal").sum() if "fusion_pred" in ev_slice.columns else -1
    
    total_t1_fp += hr_flags
    total_t2_fp += lgbm_flags
    
    print(f"Event {i}: {st} | {etype} | start_idx={s_idx} | dur={dur} | slice_len={len(ev_slice)} | Track 1 FP={hr_flags} ({hr_flags/dur*100:.2f}%) | Track 2 flags={lgbm_flags}")

print(f"\nTotal Extreme Weather Steps: {total_event_steps} (expected 1378)")
print(f"Total Track 1 False Positives: {total_t1_fp} / {total_event_steps} = {total_t1_fp/total_event_steps*100:.4f}%")

# 3. Check H01 Squall Suppression
# Event 2 is H01 convective_storm_squall (77 steps)
# Let's check how many steps had GRU flagged and how many were cleared by isolated_deviation
st_h01 = df_all[df_all["station_id"] == "AWS_IND_H01"].sort_values("timestamp").reset_index(drop=True)
h01_ev = st_h01.iloc[5743 : 5743 + 77]
print("\nH01 Squall (77 steps):")
print("  GRU flagged:", h01_ev["gru_flagged"].sum())
print("  isolated_deviation True:", h01_ev["isolated_deviation"].sum())
print("  isolated_deviation False:", (h01_ev["isolated_deviation"] == False).sum())
print("  hard_rule_flagged:", h01_ev["hard_rule_flagged"].sum())

# What about the 38 squall rows audited in Tier 2/3/4?
# In Tier 3 evaluation:
# df_storm = df[(df["station_id"] == "AWS_IND_H01") & (df["timestamp"] >= "2026-07-10 19:30:00") & (df["timestamp"] <= "2026-07-11 02:00:00")]
h01_window = df_all[(df_all["station_id"] == "AWS_IND_H01") & (df_all["timestamp"] >= "2026-07-10 19:30:00") & (df_all["timestamp"] <= "2026-07-11 02:00:00")]
print(f"\nH01 Storm Window (19:30 to 02:00): {len(h01_window)} rows")
print("  GRU flagged:", h01_window["gru_flagged"].sum())
print("  isolated_deviation False (cleared):", ((h01_window["gru_flagged"] == True) & (h01_window["isolated_deviation"] == False)).sum())
print("  hard_rule_flagged (remaining alarms):", h01_window["hard_rule_flagged"].sum())
