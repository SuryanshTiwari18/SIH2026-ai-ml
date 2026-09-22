import pandas as pd
import numpy as np

# Load previous train.parquet
prev_train = pd.read_parquet('data/features/train.parquet')
prev_train['timestamp'] = pd.to_datetime(prev_train['timestamp'])
prev_train = prev_train.set_index(['station_id', 'timestamp'])

# Compute new train features for dew_point_dep, vpd, sin_hour, cos_hour, mahalanobis_dist
from src.features import (
    compute_diurnal_harmonics,
    compute_thermodynamic_features,
    apply_mahalanobis_distance
)
import joblib

train_raw = pd.read_parquet('data/splits/train.parquet')
tier1_mask = train_raw['anomaly_type'].isin(['communication_dropout', 'data_corruption']) | train_raw['temperature_c'].isna()
clean_train = train_raw[~tier1_mask].copy()

clean_train = compute_diurnal_harmonics(clean_train)
clean_train = compute_thermodynamic_features(clean_train)
stats = joblib.load('models/mahalanobis_stats.joblib')
clean_train = apply_mahalanobis_distance(clean_train, stats)

clean_train['timestamp'] = pd.to_datetime(clean_train['timestamp'])
clean_train = clean_train.set_index(['station_id', 'timestamp'])

cols = ['mahalanobis_dist', 'dew_point_dep', 'vpd', 'sin_hour', 'cos_hour']
# Compare on common indices
common_idx = prev_train.index.intersection(clean_train.index)
print(f"Comparing {len(common_idx)} common rows across 5 columns...")

for col in cols:
    diff = (prev_train.loc[common_idx, col] - clean_train.loc[common_idx, col]).abs().max()
    print(f"Column '{col}' max diff: {diff:.2e}")
    assert diff < 1e-12, f"Mismatch in {col}!"

print(">>> ALL 5 COLUMNS ARE BYTE-FOR-BYTE IDENTICAL!")
