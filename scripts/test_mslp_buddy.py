import sys
sys.path.insert(0, ".")
import json
import numpy as np
import pandas as pd
from src.physics import reduce_pressure_to_msl
from src.stations import INDIAN_AWS_STATIONS, get_station_catalog

catalog = get_station_catalog()
with open("models/spatial_neighbors.json") as f:
    neighbors_map = json.load(f)

for split in ["test", "spatial_holdout"]:
    df = pd.read_parquet(f"data/tier4_results/{split}.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Compute P_msl for each row
    p_msl_arr = np.zeros(len(df))
    for i in range(len(df)):
        st = df["station_id"].iloc[i]
        alt = catalog[st].altitude_m if st in catalog else 100.0
        p_msl_arr[i] = reduce_pressure_to_msl(
            df["pressure_hpa"].iloc[i],
            df["temperature_c"].iloc[i],
            df["humidity_pct"].iloc[i],
            alt,
        )
    df["pressure_msl"] = p_msl_arr

    # Pivot P_msl by timestamp and station
    piv = df.pivot_table(index="timestamp", columns="station_id", values="pressure_msl")
    
    delta_p_msl = np.zeros(len(df))
    for i in range(len(df)):
        ts = df["timestamp"].iloc[i]
        st = df["station_id"].iloc[i]
        nbrs = neighbors_map.get(st, [])
        nbr_vals = [piv.at[ts, n] for n in nbrs if ts in piv.index and n in piv.columns and not np.isnan(piv.at[ts, n])]
        if nbr_vals:
            delta_p_msl[i] = p_msl_arr[i] - np.median(nbr_vals)
        else:
            delta_p_msl[i] = 0.0
            
    df["delta_P_cluster_msl"] = delta_p_msl
    
    norm = df[~df["is_anomaly"]]
    print(f"=== {split.upper()} MSLP BUDDY CHECK ===")
    print(f"Raw delta_P_cluster normal: mean = {norm['delta_P_cluster'].mean():.2f}, std = {norm['delta_P_cluster'].std():.2f}, max = {norm['delta_P_cluster'].abs().max():.2f}")
    print(f"MSLP delta_P_cluster normal: mean = {norm['delta_P_cluster_msl'].mean():.2f}, std = {norm['delta_P_cluster_msl'].std():.2f}, max = {norm['delta_P_cluster_msl'].abs().max():.2f}")
    print()
