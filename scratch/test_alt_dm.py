import json
import math
import numpy as np
import pandas as pd
import joblib
from src.stations import get_station_catalog
from src.physics import calculate_baseline_pressure_hpa

catalog = get_station_catalog()
with open('models/spatial_neighbors.json') as f:
    neighbors = json.load(f)

stats = joblib.load('models/mahalanobis_stats.joblib')
holdout = pd.read_parquet('data/splits/spatial_holdout_stations.parquet')
holdout_clean = holdout[holdout['is_anomaly'] == False].dropna(subset=['temperature_c', 'pressure_hpa', 'humidity_pct'])

dms = []
for idx, row in holdout_clean.head(1000).iterrows():
    st_id = row['station_id']
    ts = pd.to_datetime(row['timestamp'])
    h = ts.hour
    peer_st = neighbors[st_id][0]
    bucket = stats['station_hour'][(peer_st, h)]
    
    mu = bucket['mu'].copy()
    # adjust pressure mu by altitude difference
    st_alt = catalog[st_id].altitude_m
    peer_alt = catalog[peer_st].altitude_m
    dp = calculate_baseline_pressure_hpa(st_alt) - calculate_baseline_pressure_hpa(peer_alt)
    mu[1] += dp
    
    x = np.array([row['temperature_c'], row['pressure_hpa'], row['humidity_pct']])
    diff = x - mu
    dm = math.sqrt(float(diff @ bucket['inv_cov'] @ diff))
    dms.append(dm)

dms = np.array(dms)
print(f"Altitude-adjusted Mahalanobis on holdout normal rows (N=1000):")
print(f"Min: {dms.min():.3f}, Mean: {dms.mean():.3f}, Max: {dms.max():.3f}, Std: {dms.std():.3f}")
