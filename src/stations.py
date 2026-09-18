"""Station catalog for Indian Automatic Weather Station (AWS) network.

Grounded in IMD climatological normals across 4 major Indian climate zones:
1. Coastal / Maritime (moderated temperature, high humidity, sea-level pressure)
2. Arid / Desert (high diurnal temperature range, low humidity, intense solar heating)
3. Hill / Montane (high elevation, substantially reduced surface pressure, cool temperatures)
4. Gangetic Plains / Continental (broad seasonal swings, strong diurnal cycle, monsoon transitions)
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class StationMetadata:
    station_id: str
    name: str
    state: str
    latitude: float
    longitude: float
    altitude_m: float
    climate_zone: str
    # IMD Climatological baselines for reference season (early summer / pre-monsoon / monsoon)
    baseline_temp_mean_c: float
    diurnal_temp_range_c: float
    baseline_rh_mean_pct: float
    diurnal_rh_range_pct: float
    # Sensor Gaussian noise standard deviation
    noise_temp_std: float = 0.12
    noise_press_std: float = 0.18
    noise_rh_std: float = 0.50


# 16 authentic Indian AWS observatories
INDIAN_AWS_STATIONS: List[StationMetadata] = [
    # --- 1. COASTAL / MARITIME STATIONS ---
    StationMetadata(
        station_id="AWS_IND_C01",
        name="Mumbai (Colaba)",
        state="Maharashtra",
        latitude=18.9067,
        longitude=72.8147,
        altitude_m=11.0,
        climate_zone="coastal",
        baseline_temp_mean_c=29.5,
        diurnal_temp_range_c=6.0,
        baseline_rh_mean_pct=78.0,
        diurnal_rh_range_pct=18.0,
    ),
    StationMetadata(
        station_id="AWS_IND_C02",
        name="Chennai (Meenambakkam)",
        state="Tamil Nadu",
        latitude=12.9941,
        longitude=80.1809,
        altitude_m=16.0,
        climate_zone="coastal",
        baseline_temp_mean_c=31.5,
        diurnal_temp_range_c=7.5,
        baseline_rh_mean_pct=72.0,
        diurnal_rh_range_pct=22.0,
    ),
    StationMetadata(
        station_id="AWS_IND_C03",
        name="Kochi (Willingdon Island)",
        state="Kerala",
        latitude=9.9625,
        longitude=76.2707,
        altitude_m=3.0,
        climate_zone="coastal",
        baseline_temp_mean_c=28.2,
        diurnal_temp_range_c=5.5,
        baseline_rh_mean_pct=84.0,
        diurnal_rh_range_pct=16.0,
    ),
    StationMetadata(
        station_id="AWS_IND_C04",
        name="Puri (Seafront Observatory)",
        state="Odisha",
        latitude=19.8135,
        longitude=85.8312,
        altitude_m=9.0,
        climate_zone="coastal",
        baseline_temp_mean_c=30.0,
        diurnal_temp_range_c=6.2,
        baseline_rh_mean_pct=80.0,
        diurnal_rh_range_pct=19.0,
    ),

    # --- 2. ARID / DESERT STATIONS ---
    StationMetadata(
        station_id="AWS_IND_A01",
        name="Jodhpur (Thar Gateway)",
        state="Rajasthan",
        latitude=26.2638,
        longitude=73.0543,
        altitude_m=224.0,
        climate_zone="arid",
        baseline_temp_mean_c=34.5,
        diurnal_temp_range_c=16.5,
        baseline_rh_mean_pct=34.0,
        diurnal_rh_range_pct=32.0,
    ),
    StationMetadata(
        station_id="AWS_IND_A02",
        name="Jaisalmer (Desert Core)",
        state="Rajasthan",
        latitude=26.9157,
        longitude=70.9083,
        altitude_m=225.0,
        climate_zone="arid",
        baseline_temp_mean_c=36.0,
        diurnal_temp_range_c=18.0,
        baseline_rh_mean_pct=26.0,
        diurnal_rh_range_pct=28.0,
    ),
    StationMetadata(
        station_id="AWS_IND_A03",
        name="Bikaner (Northern Thar)",
        state="Rajasthan",
        latitude=28.0229,
        longitude=73.3119,
        altitude_m=242.0,
        climate_zone="arid",
        baseline_temp_mean_c=35.2,
        diurnal_temp_range_c=17.0,
        baseline_rh_mean_pct=30.0,
        diurnal_rh_range_pct=30.0,
    ),
    StationMetadata(
        station_id="AWS_IND_A04",
        name="Rajkot (Saurashtra Semi-Arid)",
        state="Gujarat",
        latitude=22.3039,
        longitude=70.8022,
        altitude_m=138.0,
        climate_zone="arid",
        baseline_temp_mean_c=32.8,
        diurnal_temp_range_c=13.5,
        baseline_rh_mean_pct=48.0,
        diurnal_rh_range_pct=35.0,
    ),

    # --- 3. HILL / MONTANE STATIONS ---
    StationMetadata(
        station_id="AWS_IND_H01",
        name="Shimla (Ridge)",
        state="Himachal Pradesh",
        latitude=31.1048,
        longitude=77.1734,
        altitude_m=2205.0,
        climate_zone="hill",
        baseline_temp_mean_c=18.5,
        diurnal_temp_range_c=9.2,
        baseline_rh_mean_pct=65.0,
        diurnal_rh_range_pct=28.0,
    ),
    StationMetadata(
        station_id="AWS_IND_H02",
        name="Srinagar (Kashmir Valley)",
        state="Jammu and Kashmir",
        latitude=34.0837,
        longitude=74.7973,
        altitude_m=1587.0,
        climate_zone="hill",
        baseline_temp_mean_c=21.0,
        diurnal_temp_range_c=12.0,
        baseline_rh_mean_pct=58.0,
        diurnal_rh_range_pct=32.0,
    ),
    StationMetadata(
        station_id="AWS_IND_H03",
        name="Shillong (Barapani)",
        state="Meghalaya",
        latitude=25.5788,
        longitude=91.8933,
        altitude_m=1496.0,
        climate_zone="hill",
        baseline_temp_mean_c=20.5,
        diurnal_temp_range_c=8.0,
        baseline_rh_mean_pct=76.0,
        diurnal_rh_range_pct=24.0,
    ),
    StationMetadata(
        station_id="AWS_IND_H04",
        name="Ooty (Udhagamandalam)",
        state="Tamil Nadu",
        latitude=11.4102,
        longitude=76.6950,
        altitude_m=2240.0,
        climate_zone="hill",
        baseline_temp_mean_c=16.8,
        diurnal_temp_range_c=8.5,
        baseline_rh_mean_pct=72.0,
        diurnal_rh_range_pct=26.0,
    ),

    # --- 4. GANGETIC PLAINS / CONTINENTAL STATIONS ---
    StationMetadata(
        station_id="AWS_IND_P01",
        name="New Delhi (Safdarjung)",
        state="Delhi",
        latitude=28.5847,
        longitude=77.2065,
        altitude_m=211.0,
        climate_zone="plains",
        baseline_temp_mean_c=33.0,
        diurnal_temp_range_c=13.0,
        baseline_rh_mean_pct=52.0,
        diurnal_rh_range_pct=36.0,
    ),
    StationMetadata(
        station_id="AWS_IND_P02",
        name="Lucknow (Amausi)",
        state="Uttar Pradesh",
        latitude=26.7606,
        longitude=80.8893,
        altitude_m=128.0,
        climate_zone="plains",
        baseline_temp_mean_c=32.5,
        diurnal_temp_range_c=12.2,
        baseline_rh_mean_pct=58.0,
        diurnal_rh_range_pct=34.0,
    ),
    StationMetadata(
        station_id="AWS_IND_P03",
        name="Patna (Airport)",
        state="Bihar",
        latitude=25.5941,
        longitude=85.0877,
        altitude_m=53.0,
        climate_zone="plains",
        baseline_temp_mean_c=31.8,
        diurnal_temp_range_c=11.0,
        baseline_rh_mean_pct=64.0,
        diurnal_rh_range_pct=30.0,
    ),
    StationMetadata(
        station_id="AWS_IND_P04",
        name="Nagpur (Sonegaon)",
        state="Maharashtra",
        latitude=21.0922,
        longitude=79.0573,
        altitude_m=310.0,
        climate_zone="plains",
        baseline_temp_mean_c=33.5,
        diurnal_temp_range_c=13.8,
        baseline_rh_mean_pct=46.0,
        diurnal_rh_range_pct=38.0,
    ),
]


def get_station_catalog(climate_zones: List[str] | None = None) -> Dict[str, StationMetadata]:
    """Retrieve filtered station dictionary keyed by station_id."""
    catalog = {}
    for st in INDIAN_AWS_STATIONS:
        if climate_zones is None or st.climate_zone in climate_zones:
            catalog[st.station_id] = st
    return catalog
