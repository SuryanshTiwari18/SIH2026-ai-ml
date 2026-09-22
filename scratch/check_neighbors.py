import pandas as pd
import numpy as np
from src.stations import INDIAN_AWS_STATIONS, get_station_catalog

catalog = get_station_catalog()
train_df = pd.read_parquet('data/splits/train.parquet')
train_stations = sorted(train_df['station_id'].unique().tolist())
holdout_df = pd.read_parquet('data/splits/spatial_holdout_stations.parquet')
holdout_stations = sorted(holdout_df['station_id'].unique().tolist())

print("Train stations:", train_stations)
print("Holdout stations:", holdout_stations)

def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return r * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

all_st = {st.station_id: (st.latitude, st.longitude) for st in INDIAN_AWS_STATIONS}

print("\n--- If neighbors are computed within train set (12 stations) ---")
for s1 in train_stations:
    dists = []
    lat1, lon1 = all_st[s1]
    for s2 in train_stations:
        if s1 == s2: continue
        lat2, lon2 = all_st[s2]
        dists.append((haversine(lat1, lon1, lat2, lon2), s2))
    dists.sort()
    print(f"{s1}: 3 nearest in train = {[x[1] for x in dists[:3]]}, dists={[round(x[0], 1) for x in dists[:3]]}")

print("\n--- If neighbors are computed within holdout set (4 stations) ---")
for s1 in holdout_stations:
    dists = []
    lat1, lon1 = all_st[s1]
    for s2 in holdout_stations:
        if s1 == s2: continue
        lat2, lon2 = all_st[s2]
        dists.append((haversine(lat1, lon1, lat2, lon2), s2))
    dists.sort()
    print(f"{s1}: 3 other holdout stations = {[x[1] for x in dists[:3]]}, dists={[round(x[0], 1) for x in dists[:3]]}")

print("\n--- If neighbors are computed across ALL 16 stations ---")
for s1 in holdout_stations:
    dists = []
    lat1, lon1 = all_st[s1]
    for s2 in all_st:
        if s1 == s2: continue
        lat2, lon2 = all_st[s2]
        dists.append((haversine(lat1, lon1, lat2, lon2), s2))
    dists.sort()
    print(f"{s1}: 3 nearest in ALL 16 = {[x[1] for x in dists[:3]]}, dists={[round(x[0], 1) for x in dists[:3]]}")
