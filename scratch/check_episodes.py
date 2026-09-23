import pandas as pd
import numpy as np

def count_episodes(df):
    results = {}
    for atype in ["data_corruption", "communication_dropout", "spike_or_drop", 
                  "power_fluctuation_glitch", "frozen_sensor", "calibration_drift", 
                  "cross_sensor_inconsistency"]:
        sub = df[df["anomaly_type"] == atype]
        ep_count = 0
        for st, st_df in sub.groupby("station_id"):
            st_df = st_df.sort_values("timestamp")
            # diff > 10 min
            time_diff = st_df["timestamp"].diff()
            new_ep = time_diff.isnull() | (time_diff > pd.Timedelta(minutes=10))
            ep_count += new_ep.sum()
        results[atype] = ep_count
    return results

df_test = pd.read_parquet("data/tier4_results/test.parquet")
df_hold = pd.read_parquet("data/tier4_results/spatial_holdout.parquet")

ep_test = count_episodes(df_test)
ep_hold = count_episodes(df_hold)

print("Test Episodes per type:")
for k, v in ep_test.items():
    print(f"  {k:28s}: {v}")
print("Total Test Episodes:", sum(ep_test.values()))

print("\nHoldout Episodes per type:")
for k, v in ep_hold.items():
    print(f"  {k:28s}: {v}")
print("Total Holdout Episodes:", sum(ep_hold.values()))
