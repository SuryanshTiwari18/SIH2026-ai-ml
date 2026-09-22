import pandas as pd
import numpy as np

def process_split_test(split_name):
    raw = pd.read_parquet(f'data/splits/{split_name}.parquet')
    tier1_mask = raw['anomaly_type'].isin(['communication_dropout', 'data_corruption']) | raw['temperature_c'].isna()
    tier1_excluded = raw[tier1_mask].copy()
    clean = raw[~tier1_mask].copy()
    
    clean['timestamp'] = pd.to_datetime(clean['timestamp'])
    clean = clean.sort_values(['station_id', 'timestamp']).reset_index(drop=True)
    
    dt_sec = clean.groupby('station_id')['timestamp'].diff().dt.total_seconds().fillna(600.0)
    station_changed = clean['station_id'] != clean['station_id'].shift(1)
    clean['is_gap'] = (dt_sec != 600.0) | station_changed
    clean['segment_id'] = clean.groupby('station_id')['is_gap'].cumsum()
    
    delta_T = []
    for _, seg in clean.groupby(['station_id', 'segment_id']):
        t = seg['temperature_c']
        dt = t.diff()
        v1 = t.rolling(6, min_periods=6).var()
        v6 = t.rolling(36, min_periods=36).var()
        s1 = t.rolling(6, min_periods=6).std()
        s24 = t.rolling(144, min_periods=144).std()
        vr = s1 / (s24 + 1e-4)
        
        seg_df = pd.DataFrame({
            'delta_T': dt,
            'var_T_1h': v1,
            'var_T_6h': v6,
            'vol_ratio_T': vr
        }, index=seg.index)
        delta_T.append(seg_df)
        
    seg_features = pd.concat(delta_T).sort_index()
    clean[['delta_T', 'var_T_1h', 'var_T_6h', 'vol_ratio_T']] = seg_features
    
    has_nan = clean[['delta_T', 'var_T_1h', 'var_T_6h', 'vol_ratio_T']].isna().any(axis=1)
    edge_excluded = clean[has_nan]
    retained = clean[~has_nan]
    
    # Check max(|delta_T|) on normal retained
    norm_retained = retained[retained['is_anomaly'] == False]
    max_dt_norm = norm_retained['delta_T'].abs().max()
    print(f"[{split_name}] Retained normal max(|delta_T|): {max_dt_norm:.4f} °C, std: {norm_retained['delta_T'].std():.4f}")
    
    # Also check if any normal row > 15
    huge = norm_retained[norm_retained['delta_T'].abs() > 15.0]
    if len(huge) > 0:
        print(f"  WARNING: {len(huge)} rows > 15C in {split_name}!")
        print(huge[['station_id', 'timestamp', 'temperature_c', 'delta_T']])
    else:
        print(f"  All normal rows <= 15C verified in {split_name}!")

for s in ['train', 'val', 'test', 'spatial_holdout_stations']:
    process_split_test(s)
