"""Genuine Extreme Weather Modeling for SkyGuard AI.

Critical Requirement:
These events represent authentic severe meteorological phenomena (heatwaves, squalls, fog).
They MUST remain strictly physically consistent across all 3 variables (T, P, RH).
Their ground-truth labels MUST BE `is_anomaly = False`.
This teaches candidate ML models (Physical QC, Isolation Forest, GRU-Autoencoder) to 
distinguish real extreme atmospheric dynamics from hardware sensor faults.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from src.stations import StationMetadata
from src.physics import calculate_dew_point_c


def inject_heatwave_event(
    t_series: pd.Series,
    p_series: pd.Series,
    rh_series: pd.Series,
    start_idx: int,
    duration_steps: int,
    intensity_deg_c: float = 6.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Inject a realistic atmospheric heatwave (synoptic heat dome).
    
    Characteristics:
    - Temperature baseline elevated by +4°C to +8°C with smooth onset and decay.
    - Relative humidity depressed due to high saturation vapor pressure (-15% to -30%).
    - Slight thermal surface pressure drop (-1.5 to -3.5 hPa).
    - Fully physically consistent: T_dew <= T_air.
    """
    n = duration_steps
    half_ramp = max(int(n * 0.15), 1)
    envelope = np.ones(n, dtype=np.float64)
    ramp_up = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, half_ramp)))
    envelope[:half_ramp] = ramp_up
    envelope[-half_ramp:] = ramp_up[::-1]

    end_idx = min(start_idx + duration_steps, len(t_series))
    eff_n = end_idx - start_idx
    eff_envelope = envelope[:eff_n]

    t_mod = t_series.iloc[start_idx:end_idx].to_numpy() + intensity_deg_c * eff_envelope
    p_mod = p_series.iloc[start_idx:end_idx].to_numpy() - (0.4 * intensity_deg_c) * eff_envelope
    
    rh_drop = 3.5 * intensity_deg_c * eff_envelope
    rh_mod = np.maximum(t_series.iloc[start_idx:end_idx].to_numpy() * 0.0 + 8.0, 
                        rh_series.iloc[start_idx:end_idx].to_numpy() - rh_drop)

    td = calculate_dew_point_c(t_mod, rh_mod)
    rh_mod = np.where(td > t_mod, 98.0, rh_mod)

    t_series.iloc[start_idx:end_idx] = t_mod
    p_series.iloc[start_idx:end_idx] = p_mod
    rh_series.iloc[start_idx:end_idx] = rh_mod

    return t_series, p_series, rh_series


def inject_convective_storm_squall(
    t_series: pd.Series,
    p_series: pd.Series,
    rh_series: pd.Series,
    start_idx: int,
    duration_steps: int,
    p_drop_hpa: float = 11.0,
    cooling_deg_c: float = 7.5,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Inject a convective storm / monsoon depression / squall event.
    
    Characteristics:
    - Pre-storm barometric drop of -6 to -15 hPa over 3-6 hours.
    - Cold downdraft / evaporative rain cooling (-5°C to -10°C drop within 30-60 mins).
    - RH surges rapidly to 96-100% (saturated cloud/precipitation environment).
    - Post-storm gradual recovery.
    """
    end_idx = min(start_idx + duration_steps, len(t_series))
    eff_n = end_idx - start_idx

    t_rel = np.linspace(0, 1, eff_n)
    storm_profile = np.exp(-3.0 * t_rel) * np.sin(np.pi * (t_rel ** 0.5))
    if storm_profile.max() > 0:
        storm_profile = storm_profile / storm_profile.max()

    p_mod = p_series.iloc[start_idx:end_idx].to_numpy() - p_drop_hpa * storm_profile
    t_mod = t_series.iloc[start_idx:end_idx].to_numpy() - cooling_deg_c * storm_profile
    
    orig_rh = rh_series.iloc[start_idx:end_idx].to_numpy()
    rh_mod = orig_rh + (99.0 - orig_rh) * storm_profile
    rh_mod = np.clip(rh_mod, 10.0, 99.8)

    td = calculate_dew_point_c(t_mod, rh_mod)
    rh_mod = np.where(td > t_mod, 98.5, rh_mod)

    t_series.iloc[start_idx:end_idx] = t_mod
    p_series.iloc[start_idx:end_idx] = p_mod
    rh_series.iloc[start_idx:end_idx] = rh_mod

    return t_series, p_series, rh_series


def inject_temperature_inversion_fog(
    t_series: pd.Series,
    p_series: pd.Series,
    rh_series: pd.Series,
    start_idx: int,
    duration_steps: int,
    t_depression_c: float = 5.0,
    p_ridge_hpa: float = 4.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Inject a radiation fog / winter temperature inversion event.
    
    Characteristics:
    - Depressed surface temperature and suppressed diurnal range.
    - Saturated relative humidity near 98-99.9%.
    - Barometric ridge (+3 to +6 hPa stable anticyclonic pressure).
    """
    end_idx = min(start_idx + duration_steps, len(t_series))
    eff_n = end_idx - start_idx

    envelope = np.sin(np.linspace(0, np.pi, eff_n)) ** 0.5

    t_mod = t_series.iloc[start_idx:end_idx].to_numpy() - t_depression_c * envelope
    p_mod = p_series.iloc[start_idx:end_idx].to_numpy() + p_ridge_hpa * envelope
    
    orig_rh = rh_series.iloc[start_idx:end_idx].to_numpy()
    rh_mod = orig_rh + (99.5 - orig_rh) * envelope
    rh_mod = np.clip(rh_mod, 15.0, 99.8)

    td = calculate_dew_point_c(t_mod, rh_mod)
    rh_mod = np.where(td > t_mod, 99.0, rh_mod)

    t_series.iloc[start_idx:end_idx] = t_mod
    p_series.iloc[start_idx:end_idx] = p_mod
    rh_series.iloc[start_idx:end_idx] = rh_mod

    return t_series, p_series, rh_series


def schedule_network_extreme_events(
    stations: Dict[str, StationMetadata],
    total_steps: int,
    event_prob_per_station: float = 0.35,
    min_network_events: int = 5,
    event_types: List[str] | None = None,
    rng: np.random.Generator | None = None,
) -> Dict[str, List[dict]]:
    """Plan extreme weather events across the AWS station network.
    
    Guarantees:
    - Each station gets an independent Bernoulli draw against event_prob_per_station.
    - If total drawn < min_network_events, additional stations are selected so at least min_network_events fire.
    - Events are distributed across multiple stations and cover at least 2 (usually all 3) distinct event types.
    """
    if rng is None:
        rng = np.random.default_rng()

    if event_types is None:
        event_types = ["heatwave", "convective_storm_squall", "temperature_inversion_fog"]

    schedule: Dict[str, List[dict]] = {st_id: [] for st_id in stations}
    station_ids = list(stations.keys())

    # 1. Independent Bernoulli draw for each station
    selected_stations = [st_id for st_id in station_ids if rng.random() <= event_prob_per_station]

    # 2. Guarantee target minimum of stations
    target_count = max(min_network_events, 5)
    if len(selected_stations) < target_count:
        unselected = [s for s in station_ids if s not in selected_stations]
        rng.shuffle(unselected)
        needed = target_count - len(selected_stations)
        selected_stations.extend(unselected[:needed])

    # 3. Schedule event(s) for each selected station
    used_event_types = set()
    margin = min(288, max(12, int(total_steps * 0.08)))

    for i, st_id in enumerate(selected_stations):
        station = stations[st_id]
        if station.climate_zone == "arid":
            weights = [0.65, 0.25, 0.10]
        elif station.climate_zone == "coastal":
            weights = [0.10, 0.80, 0.10]
        elif station.climate_zone == "hill":
            weights = [0.10, 0.55, 0.35]
        else:  # plains
            weights = [0.40, 0.40, 0.20]

        chosen_event = rng.choice(event_types, p=weights)

        # Ensure all event types appear across network
        if i >= len(selected_stations) - len(event_types):
            missing_types = [t for t in event_types if t not in used_event_types]
            if missing_types:
                chosen_event = missing_types[0]

        used_event_types.add(chosen_event)

        if chosen_event == "heatwave":
            duration = int(rng.integers(288, min(720, max(289, total_steps - 2 * margin))))
            intensity = float(rng.uniform(4.5, 8.0))
            upper_bound = max(margin + 1, total_steps - margin - duration)
            start_idx = int(rng.integers(margin, upper_bound))
            schedule[st_id].append({
                "station_id": st_id,
                "event_type": "heatwave",
                "start_idx": start_idx,
                "duration_steps": duration,
                "intensity": intensity,
                "is_anomaly": False,
            })
        elif chosen_event == "convective_storm_squall":
            duration = int(rng.integers(24, min(96, max(25, total_steps - 2 * margin))))
            p_drop = float(rng.uniform(7.0, 14.0))
            cooling = float(rng.uniform(5.5, 9.5))
            upper_bound = max(margin + 1, total_steps - margin - duration)
            start_idx = int(rng.integers(margin, upper_bound))
            schedule[st_id].append({
                "station_id": st_id,
                "event_type": "convective_storm_squall",
                "start_idx": start_idx,
                "duration_steps": duration,
                "p_drop_hpa": p_drop,
                "cooling_c": cooling,
                "is_anomaly": False,
            })
        else:  # temperature_inversion_fog
            duration = int(rng.integers(48, min(144, max(49, total_steps - 2 * margin))))
            t_dep = float(rng.uniform(3.5, 6.5))
            p_ridge = float(rng.uniform(3.0, 5.5))
            upper_bound = max(margin + 1, total_steps - margin - duration)
            start_idx = int(rng.integers(margin, upper_bound))
            schedule[st_id].append({
                "station_id": st_id,
                "event_type": "temperature_inversion_fog",
                "start_idx": start_idx,
                "duration_steps": duration,
                "t_depression_c": t_dep,
                "p_ridge_hpa": p_ridge,
                "is_anomaly": False,
            })

    return schedule


def apply_scheduled_extreme_events(
    t_series: pd.Series,
    p_series: pd.Series,
    rh_series: pd.Series,
    events: List[dict],
) -> Tuple[pd.Series, pd.Series, pd.Series, List[dict]]:
    """Apply pre-scheduled extreme weather events to a station's time series."""
    event_logs = []
    for evt in events:
        etype = evt["event_type"]
        s_idx = evt["start_idx"]
        dur = evt["duration_steps"]
        if etype == "heatwave":
            t_series, p_series, rh_series = inject_heatwave_event(
                t_series, p_series, rh_series, s_idx, dur, evt["intensity"]
            )
        elif etype == "convective_storm_squall":
            t_series, p_series, rh_series = inject_convective_storm_squall(
                t_series, p_series, rh_series, s_idx, dur, evt["p_drop_hpa"], evt["cooling_c"]
            )
        elif etype == "temperature_inversion_fog":
            t_series, p_series, rh_series = inject_temperature_inversion_fog(
                t_series, p_series, rh_series, s_idx, dur, evt["t_depression_c"], evt["p_ridge_hpa"]
            )
        event_logs.append(evt)

    return t_series, p_series, rh_series, event_logs
