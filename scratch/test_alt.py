import json
import pandas as pd
from src.stations import get_station_catalog
from src.physics import calculate_baseline_pressure_hpa

catalog = get_station_catalog()
with open('models/spatial_neighbors.json') as f:
    neighbors = json.load(f)

holdout = pd.read_parquet('data/features/spatial_holdout.parquet')
for st_id in holdout['station_id'].unique():
    peer_id = neighbors[st_id][0]
    st_alt = catalog[st_id].altitude_m
    peer_alt = catalog[peer_id].altitude_m
    dp = calculate_baseline_pressure_hpa(st_alt) - calculate_baseline_pressure_hpa(peer_alt)
    print(f"{st_id} ({st_alt}m) -> peer {peer_id} ({peer_alt}m): delta_P_baseline = {dp:.1f} hPa")
