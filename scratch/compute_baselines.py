import pandas as pd
import json

# Load all splits combined
train_raw = pd.read_parquet('data/splits/train.parquet')
val_raw = pd.read_parquet('data/splits/val.parquet')
test_raw = pd.read_parquet('data/splits/test.parquet')
hold_raw = pd.read_parquet('data/splits/spatial_holdout_stations.parquet')
all_df = pd.concat([train_raw, val_raw, test_raw, hold_raw], axis=0, ignore_index=True)

# Normal rows only, exclude corruptions/dropouts
normal_mask = (all_df['is_anomaly'] == False) & (all_df['temperature_c'] > -50.0) & all_df['temperature_c'].notna()
normal_df = all_df[normal_mask]

baselines = {}
for st, grp in normal_df.groupby('station_id'):
    baselines[st] = {
        "temperature_c": float(grp['temperature_c'].std()),
        "pressure_hpa": float(grp['pressure_hpa'].std()),
        "humidity_pct": float(grp['humidity_pct'].std()),
    }
    print(f"{st}: std_T={baselines[st]['temperature_c']:.3f}, std_P={baselines[st]['pressure_hpa']:.3f}, std_RH={baselines[st]['humidity_pct']:.3f}")

with open('scratch/test_baselines.json', 'w') as f:
    json.dump(baselines, f, indent=2)
