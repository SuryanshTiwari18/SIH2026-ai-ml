import pandas as pd

for name in ['train', 'val', 'test', 'spatial_holdout_stations']:
    df = pd.read_parquet(f'data/splits/{name}.parquet')
    print(f'=== {name} ===')
    tier1_mask = df['anomaly_type'].isin(['communication_dropout', 'data_corruption']) | df['temperature_c'].isna()
    clean = df[~tier1_mask].copy()
    clean['timestamp'] = pd.to_datetime(clean['timestamp'])
    clean = clean.sort_values(['station_id', 'timestamp'])
    
    clean['dt'] = clean.groupby('station_id')['timestamp'].diff().dt.total_seconds()
    clean['is_gap'] = clean['dt'] > 600.0
    clean['segment_id'] = clean.groupby('station_id')['is_gap'].cumsum()
    
    n_segments = clean.groupby(['station_id', 'segment_id']).ngroups
    print(f'Total valid rows: {len(clean)}, Total segments: {n_segments}')
    
    seg_lens = clean.groupby(['station_id', 'segment_id']).size()
    print(f'Segment lengths: min={seg_lens.min()}, median={seg_lens.median()}, max={seg_lens.max()}')
    print(f'Segments shorter than 144: {(seg_lens < 144).sum()}')
    print(f'Segments shorter than 36: {(seg_lens < 36).sum()}')
