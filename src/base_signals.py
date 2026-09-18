"""Physics-informed base signal generation for AWS parameters.

Models:
1. Asymmetric solar diurnal cycle (temperature minimum at dawn ~05:30, maximum at ~14:30).
2. Inverse diurnal relative humidity cycle (maximum at dawn, minimum in mid-afternoon).
3. Semi-diurnal atmospheric thermal pressure tide (peaks at 10:00 and 22:00 solar time).
4. Coupled synoptic weather systems via Ornstein-Uhlenbeck / AR(1) process (lows bring moisture & pressure drops).
5. Additive Gaussian instrument measurement noise with physical invariant enforcement.
"""

from typing import Tuple
import numpy as np
import pandas as pd

from src.stations import StationMetadata
from src.physics import (
    calculate_baseline_pressure_hpa,
    calculate_atmospheric_thermal_tide_hpa,
    calculate_dew_point_c,
)


def generate_time_index(
    start_date: str, duration_days: int, sampling_interval_minutes: int
) -> pd.DatetimeIndex:
    """Generate pandas DatetimeIndex for the simulation period."""
    periods = int((duration_days * 24 * 60) / sampling_interval_minutes)
    return pd.date_range(
        start=start_date, periods=periods, freq=f"{sampling_interval_minutes}min"
    )


def simulate_ar1_synoptic_process(
    n_steps: int,
    dt_minutes: int = 10,
    correlation_time_days: float = 3.0,
    std_dev: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate continuous-time Ornstein-Uhlenbeck / discrete AR(1) synoptic drift.
    
    Models passing synoptic-scale weather systems (fronts, multi-day highs/lows).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Decay parameter phi = exp(-dt / tau)
    tau_minutes = correlation_time_days * 24.0 * 60.0
    phi = np.exp(-dt_minutes / tau_minutes)
    shock_std = std_dev * np.sqrt(1.0 - phi**2)

    shocks = rng.normal(0.0, shock_std, size=n_steps)
    series = np.zeros(n_steps, dtype=np.float64)
    series[0] = rng.normal(0.0, std_dev)

    for t in range(1, n_steps):
        series[t] = phi * series[t - 1] + shocks[t]

    return series


def generate_station_base_signal(
    station: StationMetadata,
    time_index: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Synthesize physically consistent base (normal) signals for T, P, RH for a single station.
    
    Returns:
        (temp_series, press_series, rh_series)
    """
    n_steps = len(time_index)
    dt_min = int((time_index[1] - time_index[0]).total_seconds() / 60)

    # 1. Solar time and astronomical angles
    # Approximate local solar hour based on longitude (15 deg per hour from UTC)
    # Time index assumed to be IST (UTC+5:30 = 82.5 deg E reference meridian)
    # Longitude offset in hours: (station.longitude - 82.5) / 15.0
    local_solar_hour_offset = (station.longitude - 82.5) / 15.0
    hour_of_day = (
        time_index.hour + time_index.minute / 60.0 + time_index.second / 3600.0
    ).to_numpy()
    solar_hour = (hour_of_day + local_solar_hour_offset) % 24.0

    day_of_year = time_index.dayofyear.to_numpy()

    # 2. Elevation-derived baseline pressure
    p_base = calculate_baseline_pressure_hpa(station.altitude_m)

    # 3. Semi-diurnal thermal pressure tide
    p_tide = calculate_atmospheric_thermal_tide_hpa(solar_hour, station.latitude)

    # 4. Seasonal slow cycle (monsoon arrival warming/cooling depending on latitude)
    # In June-July in India, pre-monsoon heat peaks around late May/June, then cools 2-4°C with monsoon rains
    seasonal_phase = 2.0 * np.pi * (day_of_year - 145) / 365.25
    seasonal_temp_shift = 1.5 * np.cos(seasonal_phase)
    seasonal_rh_shift = -4.0 * np.cos(seasonal_phase)
    seasonal_press_shift = -2.0 * np.sin(seasonal_phase)

    # 5. Diurnal solar heating cycle
    # Asymmetric curve: trough at ~05:30 (sunrise), peak at ~14:30 (9 hours rise, 15 hours fall)
    # Modelled via shifted Fourier harmonics:
    # Fundamental mode peak at 14.5 hr -> phase shift (hour - 14.5) * 2pi/24
    # Second harmonic sharpens peak and broadens night cooling
    h_rad = 2.0 * np.pi * (solar_hour - 14.5) / 24.0
    diurnal_curve_t = np.cos(h_rad) + 0.22 * np.cos(2.0 * h_rad - 0.4)
    # Normalize diurnal curve to range [-0.5, +0.5]
    diurnal_curve_t = (
        (diurnal_curve_t - diurnal_curve_t.min())
        / (diurnal_curve_t.max() - diurnal_curve_t.min())
    ) - 0.5

    # 6. Coupled synoptic-scale weather system
    # Single primary synoptic pressure wave representing troughs / ridges
    synoptic_p_wave = simulate_ar1_synoptic_process(
        n_steps=n_steps,
        dt_minutes=dt_min,
        correlation_time_days=3.5,
        std_dev=3.5,  # +/- 3.5 hPa synoptic anomaly
        rng=rng,
    )

    # Low pressure system coupling:
    # When pressure is low (trough/depression):
    # - Cloud cover suppresses daytime solar heating (dampens diurnal amplitude)
    # - Influx of maritime/monsoon moisture elevates RH
    # - Nighttime cooling is reduced (cloud greenhouse effect)
    synoptic_rh_effect = -2.2 * synoptic_p_wave  # Low P -> High RH
    synoptic_t_effect = 0.4 * synoptic_p_wave    # Low P -> Slight suppression of mean temp

    # Diurnal amplitude modulation by synoptic moisture (cloud cover dampens DTR)
    dtr_modulation = 1.0 - 0.15 * np.tanh(synoptic_rh_effect / 10.0)
    effective_dtr = station.diurnal_temp_range_c * dtr_modulation

    # 7. Synthesize deterministic signals
    # Temperature
    t_signal = (
        station.baseline_temp_mean_c
        + seasonal_temp_shift
        + synoptic_t_effect
        + effective_dtr * diurnal_curve_t
    )

    # Atmospheric Pressure: P_base + S2 tide + synoptic + seasonal
    p_signal = p_base + p_tide + synoptic_p_wave + seasonal_press_shift

    # Relative Humidity:
    # Physical anti-correlation with temperature diurnal cycle (saturation vapor pressure increases with T)
    # At peak temperature (14:30), RH reaches minimum.
    # At minimum temperature (05:30), RH reaches maximum.
    rh_diurnal_curve = -diurnal_curve_t  # Inverted!
    rh_signal = (
        station.baseline_rh_mean_pct
        + seasonal_rh_shift
        + synoptic_rh_effect
        + (station.diurnal_rh_range_pct * dtr_modulation) * rh_diurnal_curve
    )

    # 8. Sensor Gaussian measurement noise
    t_noise = rng.normal(0.0, station.noise_temp_std, size=n_steps)
    p_noise = rng.normal(0.0, station.noise_press_std, size=n_steps)
    rh_noise = rng.normal(0.0, station.noise_rh_std, size=n_steps)

    t_final = t_signal + t_noise
    p_final = p_signal + p_noise
    rh_final = rh_signal + rh_noise

    # 9. Thermodynamic reconciliation and clipping
    # Relative humidity must strictly satisfy 5.0% <= RH <= 99.5% in standard non-condensing normal conditions
    rh_final = np.clip(rh_final, 5.0, 99.8)

    # Ensure Clausius-Clapeyron invariant: T_dew <= T_air
    # Since RH <= 99.8%, T_dew is mathematically guaranteed to be strictly less than T_air.
    # Just to be 100% physically bounded:
    td = calculate_dew_point_c(t_final, rh_final)
    diff = td - t_final
    if np.any(diff > 0):
        # Slightly pull RH down if numerical float margin touched 100%
        rh_final = np.where(diff > 0, 98.0, rh_final)

    temp_series = pd.Series(t_final, index=time_index, name="temperature_c")
    press_series = pd.Series(p_final, index=time_index, name="pressure_hpa")
    rh_series = pd.Series(rh_final, index=time_index, name="humidity_pct")

    return temp_series, press_series, rh_series
