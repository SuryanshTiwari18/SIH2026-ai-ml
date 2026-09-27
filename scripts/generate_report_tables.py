"""
Formats all tables for docs/SEASON_SAFETY_REPORT.md directly from real computation.
"""

import json
from pathlib import Path
import pandas as pd

with open("data/real_climatology/station_month_climatology.json") as f:
    clim = json.load(f)

with open("data/real_climatology/elevations.json") as f:
    elevations = json.load(f)

with open("data/real_climatology/real_cities_eval.json") as f:
    real_eval = json.load(f)

cities = ["delhi", "leh", "jaisalmer", "mumbai", "chennai", "guwahati", "bengaluru"]
city_titles = {
    "delhi": "Delhi (Plains / Continental)",
    "leh": "Leh (Himalayan / Alpine)",
    "jaisalmer": "Jaisalmer (Arid / Thar Desert)",
    "mumbai": "Mumbai (West Coast / Maritime)",
    "chennai": "Chennai (East Coast / Maritime)",
    "guwahati": "Guwahati (Northeast / Humid)",
    "bengaluru": "Bengaluru (Deccan Plateau / Moderate Control)",
}

print("### Station-Month Climatology Summary Table\n")
for city in cities:
    elev = elevations.get(city, 0.0)
    print(f"#### {city_titles[city]} (Elevation: {elev}m)\n")
    print("| Month | T_mean (°C) | T_std (°C) | T_min (°C) | T_max (°C) | RH_mean (%) | RH_std (%) | RH_min (%) | RH_max (%) |")
    print("|-------|-------------|------------|------------|------------|-------------|------------|------------|------------|")
    for m in range(1, 13):
        row = clim[city][str(m)]
        print(f"| {m:02d} | {row['temp_mean']:11.2f} | {row['temp_std']:10.2f} | {row['temp_min']:10.2f} | {row['temp_max']:10.2f} | {row['rh_mean']:11.2f} | {row['rh_std']:10.2f} | {row['rh_min']:10.2f} | {row['rh_max']:10.2f} |")
    print()
