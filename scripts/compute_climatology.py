"""
Compute Station-Month Climatology (Tier 1B) from real 20-year data (2006-2025).
Saves:
1. data/real_climatology/station_month_climatology.json
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

REAL_DIR = Path("data/real_climatology")
OUT_JSON = REAL_DIR / "station_month_climatology.json"

CITIES = ["delhi", "leh", "jaisalmer", "mumbai", "chennai", "guwahati", "bengaluru"]

CITY_ZONE_MAP = {
    "delhi": "plains",
    "leh": "hill",
    "jaisalmer": "arid",
    "mumbai": "coastal",
    "chennai": "coastal",
    "guwahati": "plains",
    "bengaluru": "plains",  # plateau / moderate control
}

STATION_TO_CITY = {
    # 16 standard stations in Indian AWS network
    "AWS_IND_C01": "mumbai",
    "AWS_IND_C02": "chennai",
    "AWS_IND_C03": "mumbai",     # Kochi -> coastal peer
    "AWS_IND_C04": "chennai",    # Puri -> coastal east peer
    "AWS_IND_A01": "jaisalmer",  # Jodhpur -> arid peer
    "AWS_IND_A02": "jaisalmer",
    "AWS_IND_A03": "jaisalmer",  # Bikaner -> arid peer
    "AWS_IND_A04": "jaisalmer",  # Rajkot -> arid peer
    "AWS_IND_H01": "leh",        # Shimla -> hill peer
    "AWS_IND_H02": "leh",        # Srinagar -> hill peer
    "AWS_IND_H03": "guwahati",   # Shillong -> northeast humid peer
    "AWS_IND_H04": "bengaluru",  # Ooty -> high elevation moderate peer
    "AWS_IND_P01": "delhi",
    "AWS_IND_P02": "delhi",      # Lucknow -> plains peer
    "AWS_IND_P03": "guwahati",   # Patna -> eastern plains peer
    "AWS_IND_P04": "delhi",      # Nagpur -> central plains peer
}


def compute_climatology():
    climatology = {}
    city_dfs = {}

    for city in CITIES:
        pq_path = REAL_DIR / f"{city}.parquet"
        if not pq_path.exists():
            raise FileNotFoundError(f"Missing {pq_path}")
        df = pd.read_parquet(pq_path)
        df["time"] = pd.to_datetime(df["time"])
        df["month"] = df["time"].dt.month
        city_dfs[city] = df

        climatology[city] = {}
        for m in range(1, 13):
            sub = df[df["month"] == m]

            # Temperatures
            t_mean = float(sub["temperature_2m_mean"].mean())
            t_std = float(sub["temperature_2m_mean"].std(ddof=1))
            # Use true envelope bounds: min of daily min, max of daily max
            t_min = float(sub["temperature_2m_min"].min())
            t_max = float(sub["temperature_2m_max"].max())

            # Relative Humidity
            rh_mean = float(sub["relative_humidity_2m_mean"].mean())
            rh_std = float(sub["relative_humidity_2m_mean"].std(ddof=1))
            rh_min = float(max(0.0, sub["relative_humidity_2m_min"].min()))
            rh_max = float(min(100.0, sub["relative_humidity_2m_max"].max()))

            climatology[city][str(m)] = {
                "temp_mean": round(t_mean, 2),
                "temp_std": round(t_std, 2),
                "temp_min": round(t_min, 2),
                "temp_max": round(t_max, 2),
                "rh_mean": round(rh_mean, 2),
                "rh_std": round(rh_std, 2),
                "rh_min": round(rh_min, 2),
                "rh_max": round(rh_max, 2),
            }

    # Also compute aggregate climate-zone fallbacks (coastal, arid, hill, plains)
    zones = ["coastal", "arid", "hill", "plains"]
    zone_climatology = {}
    for z in zones:
        zone_cities = [c for c, cz in CITY_ZONE_MAP.items() if cz == z]
        zone_dfs = [city_dfs[c] for c in zone_cities]
        combined = pd.concat(zone_dfs, ignore_index=True)
        zone_climatology[z] = {}
        for m in range(1, 13):
            sub = combined[combined["month"] == m]
            zone_climatology[z][str(m)] = {
                "temp_mean": round(float(sub["temperature_2m_mean"].mean()), 2),
                "temp_std": round(float(sub["temperature_2m_mean"].std(ddof=1)), 2),
                "temp_min": round(float(sub["temperature_2m_min"].min()), 2),
                "temp_max": round(float(sub["temperature_2m_max"].max()), 2),
                "rh_mean": round(float(sub["relative_humidity_2m_mean"].mean()), 2),
                "rh_std": round(float(sub["relative_humidity_2m_mean"].std(ddof=1)), 2),
                "rh_min": round(float(max(0.0, sub["relative_humidity_2m_min"].min())), 2),
                "rh_max": round(float(min(100.0, sub["relative_humidity_2m_max"].max())), 2),
            }

    # Also store zone_fallbacks and station_map in the json structure or auxiliary keys
    # The prompt specified structure: { "<city>": { "<month>": { ... } } }
    # To keep exact root compatibility while supporting zone fallbacks,
    # we include the 7 cities as root keys, plus zone keys or a dedicated "_metadata" key.
    output_data = {}
    for city in CITIES:
        output_data[city] = climatology[city]
    
    # Store climate zone fallbacks
    for z in zones:
        output_data[f"zone_{z}"] = zone_climatology[z]

    output_data["_metadata"] = {
        "cities": CITIES,
        "zones": zones,
        "station_to_city": STATION_TO_CITY,
        "city_zone_map": CITY_ZONE_MAP,
    }

    with open(OUT_JSON, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Successfully computed and saved {OUT_JSON}")
    return output_data


if __name__ == "__main__":
    compute_climatology()
