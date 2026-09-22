import pandas as pd

# Inspect AWS_IND_A02 in val around 2026-07-16 08:20:00
val_raw = pd.read_parquet('data/splits/val.parquet')
val_raw['timestamp'] = pd.to_datetime(val_raw['timestamp'])
a02 = val_raw[(val_raw['station_id'] == 'AWS_IND_A02') & (val_raw['timestamp'] >= '2026-07-16 07:40:00') & (val_raw['timestamp'] <= '2026-07-16 08:40:00')]
print("=== AWS_IND_A02 in val around 2026-07-16 08:20:00 ===")
print(a02[['station_id', 'timestamp', 'temperature_c', 'is_anomaly', 'anomaly_type']])

# Inspect AWS_IND_P03 in spatial_holdout around 2026-07-24 20:30:00
hold_raw = pd.read_parquet('data/splits/spatial_holdout_stations.parquet')
hold_raw['timestamp'] = pd.to_datetime(hold_raw['timestamp'])
p03 = hold_raw[(hold_raw['station_id'] == 'AWS_IND_P03') & (hold_raw['timestamp'] >= '2026-07-24 20:00:00') & (hold_raw['timestamp'] <= '2026-07-24 21:00:00')]
print("\n=== AWS_IND_P03 in spatial_holdout around 2026-07-24 20:30:00 ===")
print(p03[['station_id', 'timestamp', 'temperature_c', 'is_anomaly', 'anomaly_type']])
