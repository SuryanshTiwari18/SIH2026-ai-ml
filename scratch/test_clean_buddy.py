import pandas as pd
import numpy as np
import json
from src.features import compute_spatial_buddy_features

with open('models/spatial_neighbors.json') as f:
    neighbor_mapping = json.load(f)

for name in ['train', 'val', 'test', 'spatial_holdout']:
    raw = pd.read_parquet(f'data/splits/{name if name != "spatial_holdout" else "spatial_holdout_stations"}.parquet')
    
    # Exclude tier1 from target and reference
    tier1_mask = raw['anomaly_type'].isin(['communication_dropout', 'data_corruption']) | raw['temperature_c'].isna()
    clean = raw[~tier1_mask].copy()
    
    if name == 'spatial_holdout':
        train_raw = pd.read_parquet('data/splits/train.parquet')
        val_raw = pd.read_parquet('data/splits/val.parquet')
        test_raw = pd.read_parquet('data/splits/test.parquet')
        all_op = pd.concat([train_raw, val_raw, test_raw], axis=0, ignore_index=True)
        t1_op = all_op['anomaly_type'].isin(['communication_dropout', 'data_corruption']) | all_op['temperature_c'].isna()
        ref = all_op[~t1_op].copy()
    else:
        ref = clean
        
    buddy_df = compute_spatial_buddy_features(clean, neighbor_mapping, ref)
    print(f"[{name}] delta_T_buddy max: {buddy_df['delta_T_buddy'].max():.3f}, delta_P_cluster min: {buddy_df['delta_P_cluster'].min():.3f}, max: {buddy_df['delta_P_cluster'].max():.3f}")
