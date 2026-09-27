"""
Fetch real historical weather data from Open-Meteo Archive API for 7 Indian cities.
Saved under data/real_climatology/<city>.parquet
"""

import json
import logging
from pathlib import Path
import time
import urllib.request
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fetch_climatology")

OUT_DIR = Path("data/real_climatology")
OUT_DIR.mkdir(parents=True, exist_ok=True)

STATIONS = [
    {"city": "delhi", "name": "Delhi", "zone": "plains", "lat": 28.6139, "lon": 77.2090, "station_code": "AWS_IND_P01"},
    {"city": "leh", "name": "Leh", "zone": "hill", "lat": 34.1526, "lon": 77.5771, "station_code": "AWS_IND_H_LEH"},
    {"city": "jaisalmer", "name": "Jaisalmer", "zone": "arid", "lat": 26.9157, "lon": 70.9083, "station_code": "AWS_IND_A02"},
    {"city": "mumbai", "name": "Mumbai", "zone": "coastal", "lat": 19.0760, "lon": 72.8777, "station_code": "AWS_IND_C01"},
    {"city": "chennai", "name": "Chennai", "zone": "coastal", "lat": 13.0827, "lon": 80.2707, "station_code": "AWS_IND_C02"},
    {"city": "guwahati", "name": "Guwahati", "zone": "plains", "lat": 26.1445, "lon": 91.7362, "station_code": "AWS_IND_P_GUW"},
    {"city": "bengaluru", "name": "Bengaluru", "zone": "plains", "lat": 12.9716, "lon": 77.5946, "station_code": "AWS_IND_P_BLR"},
]


def fetch_all():
    elevations = {}
    if (OUT_DIR / "elevations.json").exists():
        try:
            with open(OUT_DIR / "elevations.json", "r") as f:
                elevations = json.load(f)
        except Exception:
            elevations = {}

    for st in STATIONS:
        city = st["city"]
        lat = st["lat"]
        lon = st["lon"]
        pq_path = OUT_DIR / f"{city}.parquet"

        if pq_path.exists():
            df_exist = pd.read_parquet(pq_path)
            if len(df_exist) >= 7000:
                elev = float(df_exist["elevation"].iloc[0])
                elevations[city] = elev
                logger.info("Found existing %s (%d rows, elevation: %s m). Skipping fetch.", pq_path, len(df_exist), elev)
                continue

        logger.info("Fetching 20-year history (2006-2025) for %s (lat=%s, lon=%s)...", city, lat, lon)

        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}&"
            f"start_date=2006-01-01&end_date=2025-12-31&"
            f"daily=temperature_2m_mean,temperature_2m_max,temperature_2m_min,"
            f"relative_humidity_2m_mean,relative_humidity_2m_max,relative_humidity_2m_min,"
            f"surface_pressure_mean,pressure_msl_mean&"
            f"timezone=Asia%2FKolkata"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SkyGuardAI/1.0"})
        
        data = None
        for attempt in range(6):
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as e:
                wait_time = (attempt + 1) * 15
                logger.warning("HTTP %s on %s (attempt %d). Waiting %d seconds before retry...", e.code, city, attempt + 1, wait_time)
                time.sleep(wait_time)
            except Exception as e:
                wait_time = (attempt + 1) * 10
                logger.warning("Error %s on %s. Waiting %d seconds...", e, city, wait_time)
                time.sleep(wait_time)

        if data is None:
            raise RuntimeError(f"Failed to fetch data for {city} after retries.")

        elev = data.get("elevation")
        elevations[city] = elev

        daily = data.get("daily", {})
        df = pd.DataFrame(daily)
        df["time"] = pd.to_datetime(df["time"])
        df["city"] = city
        df["station_name"] = st["name"]
        df["station_id"] = st["station_code"]
        df["climate_zone"] = st["zone"]
        df["latitude"] = lat
        df["longitude"] = lon
        df["elevation"] = elev

        # Standardized column aliases
        df["temperature_2m"] = df["temperature_2m_mean"]
        df["relative_humidity_2m"] = df["relative_humidity_2m_mean"]
        df["surface_pressure"] = df["surface_pressure_mean"]
        df["pressure_msl"] = df["pressure_msl_mean"]

        # Telemetry columns for pipeline compatibility
        df["temperature_c"] = df["temperature_2m_mean"]
        df["humidity_pct"] = df["relative_humidity_2m_mean"]
        df["pressure_hpa"] = df["surface_pressure_mean"]
        df["timestamp"] = df["time"]

        pq_path = OUT_DIR / f"{city}.parquet"
        df.to_parquet(pq_path, index=False)
        logger.info("Saved %s (%d rows, elevation: %s m)", pq_path, len(df), elev)
        time.sleep(3.5)

    with open(OUT_DIR / "elevations.json", "w") as f:
        json.dump(elevations, f, indent=2)
    logger.info("Saved elevations: %s", elevations)
    return elevations


if __name__ == "__main__":
    fetch_all()
