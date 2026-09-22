import pandas as pd
import numpy as np

# Load train raw
train_raw = pd.read_parquet('data/splits/train.parquet')
print("Total train rows:", len(train_raw))

# 1. Identify tier1 excluded rows
tier1_mask = train_raw['anomaly_type'].isin(['communication_dropout', 'data_corruption']) | train_raw['temperature_c'].isna()
tier1_excluded = train_raw[tier1_mask].copy()
clean_train = train_raw[~tier1_mask].copy()
print(f"Tier 1 excluded: {len(tier1_excluded)}, Clean rows: {len(clean_train)}")

# 2. Sort by station and timestamp
clean_train['timestamp'] = pd.to_datetime(clean_train['timestamp'])
clean_train = clean_train.sort_values(['station_id', 'timestamp']).reset_index(drop=True)

# 3. Identify gaps > 10 min
dt_sec = clean_train.groupby('station_id')['timestamp'].diff().dt.total_seconds().fillna(600.0)
station_changed = clean_train['station_id'] != clean_train['station_id'].shift(1)
clean_train['is_gap'] = (dt_sec != 600.0) | station_changed
clean_train['segment_id'] = clean_train.groupby('station_id')['is_gap'].cumsum()

# 4. Compute derivative and rolling per segment
delta_T = []
var_T_1h = []
var_T_6h = []
vol_ratio_T = []

for _, seg in clean_train.groupby(['station_id', 'segment_id']):
    t = seg['temperature_c']
    dt = t.diff() # first row is NaN
    v1 = t.rolling(6, min_periods=6).var() # first 5 are NaN
    v6 = t.rolling(36, min_periods=36).var() # first 35 are NaN
    s1 = t.rolling(6, min_periods=6).std()
    s24 = t.rolling(144, min_periods=144).std()
    vr = s1 / (s24 + 1e-4) # first 143 are NaN
    
    seg_df = pd.DataFrame({
        'delta_T': dt,
        'var_T_1h': v1,
        'var_T_6h': v6,
        'vol_ratio_T': vr
    }, index=seg.index)
    delta_T.append(seg_df)

seg_features = pd.concat(delta_T).sort_index()
clean_train[['delta_T', 'var_T_1h', 'var_T_6h', 'vol_ratio_T']] = seg_features

# Check NaNs
has_nan = clean_train[['delta_T', 'var_T_1h', 'var_T_6h', 'vol_ratio_T']].isna().any(axis=1)
edge_excluded = clean_train[has_nan]
retained = clean_train[~has_nan]

print(f"Edge excluded: {len(edge_excluded)}, Retained: {len(retained)}")
print(f"Total check: {len(tier1_excluded)} + {len(edge_excluded)} + {len(retained)} = {len(tier1_excluded) + len(edge_excluded) + len(retained)} (expected {len(train_raw)})")

# Check std on normal retained rows
norm_retained = retained[retained['is_anomaly'] == False]
print("\n--- Corrected Train Normal Retained Statistics ---")
print(f"delta_T std: {norm_retained['delta_T'].std():.4f}, mean: {norm_retained['delta_T'].mean():.4f}, max: {norm_retained['delta_T'].max():.4f}, min: {norm_retained['delta_T'].min():.4f}")
print(f"var_T_1h std: {norm_retained['var_T_1h'].std():.4f}, mean: {norm_retained['var_T_1h'].mean():.4f}, max: {norm_retained['var_T_1h'].max():.4f}, min: {norm_retained['var_T_1h'].min():.4f}")

# Check bug case specifically: AWS_IND_H04 around 2026-06-13 02:20:00
h04_case = clean_train[(clean_train['station_id'] == 'AWS_IND_H04') & (clean_train['timestamp'] >= '2026-06-13 01:50:00') & (clean_train['timestamp'] <= '2026-06-13 02:40:00')]
print("\nAWS_IND_H04 around 2026-06-13 02:20:00:")
print(h04_case[['timestamp', 'temperature_c', 'delta_T', 'var_T_1h', 'is_anomaly', 'anomaly_type']])
