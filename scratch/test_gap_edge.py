import pandas as pd
import numpy as np

for name in ['train', 'val', 'test', 'spatial_holdout_stations']:
    df = pd.read_parquet(f'data/splits/{name}.parquet')
    tier1_mask = df['anomaly_type'].isin(['communication_dropout', 'data_corruption']) | df['temperature_c'].isna()
    clean = df[~tier1_mask].copy()
    clean['timestamp'] = pd.to_datetime(clean['timestamp'])
    clean = clean.sort_values(['station_id', 'timestamp'])
    
    clean['dt'] = clean.groupby('station_id')['timestamp'].diff().dt.total_seconds()
    clean['is_gap'] = clean['dt'] > 600.0
    clean['segment_id'] = clean.groupby('station_id')['is_gap'].cumsum()
    
    # Calculate for each window size: 144 vs 36
    for W in [144, 36]:
        # Count rows in each segment that have index < (W - 1)
        clean[f'row_in_seg'] = clean.groupby(['station_id', 'segment_id']).cumcount()
        n_nan = (clean['row_in_seg'] < (W - 1)).sum()
        pct = (n_nan / len(clean)) * 100
        print(f"[{name}] W={W}: {n_nan} edge rows ({pct:.2f}%) routed to gap_edge_excluded, {len(clean) - n_nan} retained")
