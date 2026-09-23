import pandas as pd
import numpy as np

df_train = pd.read_parquet('data/tier4_results/train.parquet')
df_val = pd.read_parquet('data/tier4_results/val.parquet')
df_test = pd.read_parquet('data/tier4_results/test.parquet')
df_hold = pd.read_parquet('data/tier4_results/spatial_holdout.parquet')

EXTREME_WEATHER_EVENTS = [
    {"id": "EV01", "name": "June Heatwave", "type": "heatwave", "split": "train", "station": "AWS_IND_A01", "start_idx": 3843, "duration": 445},
    {"id": "EV02", "name": "July Convective Storm", "type": "squall", "split": "train", "station": "AWS_IND_H01", "start_idx": 5743, "duration": 77},
    {"id": "EV03", "name": "Mid-July Cold Front", "type": "cold_front", "split": "train", "station": "AWS_IND_H02", "start_idx": 3752, "duration": 138},
    {"id": "EV04", "name": "Late-July Gale Wind", "type": "high_wind", "split": "train", "station": "AWS_IND_P04", "start_idx": 3225, "duration": 134},
    {"id": "EV05", "name": "Spatial Holdout Extreme", "type": "coastal_squall", "split": "spatial_holdout", "station": "AWS_IND_P03", "start_idx": 4120, "duration": 150}
]

dfs = {'train': df_train, 'val': df_val, 'test': df_test, 'spatial_holdout': df_hold}

print("=== 5-EVENT EXTREME WEATHER AUDIT UNDER TWO-TRACK ARCHITECTURE ===")
for ev in EXTREME_WEATHER_EVENTS:
    df_s = dfs[ev["split"]].copy()
    df_s["timestamp"] = pd.to_datetime(df_s["timestamp"])
    t_start = pd.Timestamp("2026-06-01 00:00:00") + pd.Timedelta(minutes=ev["start_idx"] * 10)
    t_end = t_start + pd.Timedelta(minutes=(ev["duration"] - 1) * 10)
    sub = df_s[(df_s["station_id"] == ev["station"]) & (df_s["timestamp"] >= t_start) & (df_s["timestamp"] <= t_end)]
    
    n_steps = len(sub)
    t1_fp = int((sub['hard_rule_flagged'] == True).sum())
    
    # Track 2 maintenance queue flags
    t2_assigned = ['frozen_sensor', 'calibration_drift', 'cross_sensor_inconsistency', 'power_fluctuation_glitch', 'spike_or_drop']
    t2_fp = int(((sub['hard_rule_flagged'] == False) & (sub['fusion_flagged'] == True) & sub['fusion_predicted_type'].isin(t2_assigned)).sum())
    
    # Also check if regional suppression can suppress Track 2 during regional weather!
    t2_suppressed = int(((sub['hard_rule_suppressed'] == True) & (sub['fusion_flagged'] == True)).sum())
    
    print(f"Event {ev['id']} ({ev['name']}) on {ev['station']} (steps: {n_steps}):")
    print(f"  Track 1 (Operational Emergency Alert) False Alarms: {t1_fp} / {n_steps} ({t1_fp/n_steps*100:.2f}%)")
    print(f"  Track 2 (Maintenance Queue Review) Rows:           {t2_fp} / {n_steps} ({t2_fp/n_steps*100:.2f}%)")
    print(f"  Regional Weather Suppressed flags:                {sub['hard_rule_suppressed'].sum()} / {n_steps}")
