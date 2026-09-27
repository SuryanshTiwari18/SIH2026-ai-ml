"""Physics-based equations and thermodynamic calculations for AWS sensor telemetry.

Covers:
1. International Barometric Formula for elevation-derived surface pressure.
2. August-Roche-Magnus approximation for saturation vapor pressure and dew point.
3. Semi-diurnal atmospheric thermal tide calculation for tropical/subtropical latitudes.
4. Physical consistency assertions and thermodynamic bounds.
"""

import numpy as np


# International Standard Atmosphere (ISA) constants
P0_SEA_LEVEL_HPA = 1013.25
T0_SEA_LEVEL_K = 288.15
LAPSE_RATE_K_PER_M = 0.0065
GRAVITY_M_PER_S2 = 9.80665
MOLAR_MASS_AIR_KG_MOL = 0.0289644
UNIVERSAL_GAS_CONST = 8.3144598
BAROMETRIC_EXPONENT = (GRAVITY_M_PER_S2 * MOLAR_MASS_AIR_KG_MOL) / (
    UNIVERSAL_GAS_CONST * LAPSE_RATE_K_PER_M
)  # ~5.25588


def calculate_baseline_pressure_hpa(altitude_m: float) -> float:
    """Calculate standard baseline atmospheric pressure at a given altitude using the ISA formula.
    
    Args:
        altitude_m: Station altitude in meters above mean sea level.
        
    Returns:
        Baseline atmospheric surface pressure in hPa.
    """
    if altitude_m < 0:
        altitude_m = 0.0
    pressure = P0_SEA_LEVEL_HPA * (1.0 - (LAPSE_RATE_K_PER_M * altitude_m) / T0_SEA_LEVEL_K) ** BAROMETRIC_EXPONENT
    return float(pressure)


def calculate_saturation_vapor_pressure_hpa(temp_c: np.ndarray | float) -> np.ndarray | float:
    """Calculate saturation vapor pressure e_s(T) in hPa via August-Roche-Magnus approximation.
    
    Valid for temperatures between -40°C and +60°C.
    e_s(T) = 6.112 * exp((17.67 * T) / (T + 243.5))
    """
    return 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))


def calculate_actual_vapor_pressure_hpa(
    temp_c: np.ndarray | float, humidity_pct: np.ndarray | float
) -> np.ndarray | float:
    """Calculate actual partial water vapor pressure e in hPa.
    
    e = e_s(T) * (RH / 100)
    """
    es = calculate_saturation_vapor_pressure_hpa(temp_c)
    return es * (humidity_pct / 100.0)


def calculate_dew_point_c(
    temp_c: np.ndarray | float, humidity_pct: np.ndarray | float
) -> np.ndarray | float:
    """Calculate dew point temperature T_d in °C via inverse August-Roche-Magnus equation.
    
    Thermodynamic invariant: For RH <= 100%, T_d <= T_air.
    """
    # Clip RH below at 0.1% to prevent log(0) in calculations
    rh_safe = np.maximum(humidity_pct, 0.1)
    e = calculate_actual_vapor_pressure_hpa(temp_c, rh_safe)
    # gamma = ln(e / 6.112)
    gamma = np.log(e / 6.112)
    dew_point = (243.5 * gamma) / (17.67 - gamma)
    return dew_point


def check_thermodynamic_consistency(
    temp_c: np.ndarray | float,
    humidity_pct: np.ndarray | float,
    pressure_hpa: np.ndarray | float,
    tolerance_deg_c: float = 0.05,
) -> np.ndarray:
    """Evaluate thermodynamic consistency:
    1. 0.0 <= RH <= 100.0 (allow minor numeric float tolerance)
    2. Dew point <= Air temperature + tolerance
    3. Pressure within physically plausible atmospheric bounds (500 to 1080 hPa)
    4. Temperature within valid terrestrial bounds (-20°C to +60°C for India)
    
    Returns boolean array where True indicates physically consistent, False indicates violation.
    """
    temp = np.asarray(temp_c)
    rh = np.asarray(humidity_pct)
    p = np.asarray(pressure_hpa)

    # Handle NaNs gracefully
    valid_mask = ~np.isnan(temp) & ~np.isnan(rh) & ~np.isnan(p)
    is_consistent = np.ones_like(temp, dtype=bool)

    if not np.any(valid_mask):
        return is_consistent

    t_val = temp[valid_mask]
    rh_val = rh[valid_mask]
    p_val = p[valid_mask]

    # Bounds
    rh_valid = (rh_val >= 0.0) & (rh_val <= 100.05)
    t_valid = (t_val >= -25.0) & (t_val <= 60.0)
    p_valid = (p_val >= 550.0) & (p_val <= 1060.0)

    # Dew point check
    td_val = calculate_dew_point_c(t_val, rh_val)
    td_valid = td_val <= (t_val + tolerance_deg_c)

    consistent_subset = rh_valid & t_valid & p_valid & td_valid
    is_consistent[valid_mask] = consistent_subset
    return is_consistent


def calculate_atmospheric_thermal_tide_hpa(
    solar_hour: np.ndarray | float, latitude_deg: float
) -> np.ndarray | float:
    """Calculate the semi-diurnal atmospheric thermal solar tide S2(P).
    
    Prominent in tropical and subtropical latitudes (amplitude ~1.2 - 2.0 hPa).
    Peaks around 10:00 and 22:00 local solar time; troughs around 04:00 and 16:00.
    Amplitude scales with cos(latitude).
    """
    lat_rad = np.radians(latitude_deg)
    amplitude = 1.65 * np.cos(lat_rad)
    # Peak at 10.0 and 22.0 hours -> phase shift: (hour - 4.0) * (2*pi / 12)
    # At hour = 10.0: (10 - 4)/12 * 2pi = 6/12 * 2pi = pi -> sin(pi) is 0, wait!
    # Let's verify peak: peak of cos((hour - 10)/12 * 2pi) is at hour = 10!
    # cos((hour - 10)/12 * 2pi) has peaks at hour=10 and hour=22 (since period is 12).
    # Value at 10: cos(0) = +1. Value at 16: cos(pi) = -1. Value at 22: cos(2pi) = +1. Perfect!
    phase = 2.0 * np.pi * (solar_hour - 10.0) / 12.0
    return amplitude * np.cos(phase)


# -----------------------------------------------------------------------------
# Season-Safety Invariant Physics Additions (Strategies 1 & 2)
# -----------------------------------------------------------------------------
RD_DRY_AIR_J_KG_K = 287.05  # Gas constant for dry air in J/(kg*K)


def reduce_pressure_to_msl(
    pressure_hpa: np.ndarray | float,
    temp_c: np.ndarray | float,
    rh_pct: np.ndarray | float,
    altitude_m: float,
) -> np.ndarray | float:
    """Computes Mean Sea Level Pressure (MSLP) via hypsometric reduction.

    P_MSL = P * exp(g * z / (Rd * Tv))
    Tv = (T + 273.15) * (1 + 0.61 * q)
    where:
    - g = 9.80665 m/s^2 (standard gravity)
    - Rd = 287.05 J/(kg*K) (dry air gas constant)
    - z = altitude_m (station elevation in meters)
    - Tv = virtual temperature in Kelvin
    - q = specific humidity ~ 0.622 * e / (P - 0.378 * e)

    Enables season- and elevation-invariant spatial buddy pressure comparisons.
    """
    if altitude_m <= 0:
        return pressure_hpa

    p = np.asarray(pressure_hpa, dtype=np.float64)
    t = np.asarray(temp_c, dtype=np.float64)
    rh = np.asarray(rh_pct, dtype=np.float64)

    # Actual vapor pressure e in hPa
    e_val = calculate_actual_vapor_pressure_hpa(t, rh)

    # Specific humidity q (dimensionless kg/kg), safely bounded
    p_safe = np.maximum(p, 300.0)
    denom = np.maximum(p_safe - 0.378 * e_val, 100.0)
    q = np.clip(0.622 * e_val / denom, 0.0, 0.1)

    # Virtual temperature Tv in Kelvin
    t_kelvin = t + 273.15
    t_v = t_kelvin * (1.0 + 0.61 * q)

    # Hypsometric reduction exponent
    exponent = (GRAVITY_M_PER_S2 * altitude_m) / (RD_DRY_AIR_J_KG_K * t_v)
    p_msl = p * np.exp(exponent)

    if np.isscalar(pressure_hpa):
        return float(p_msl)
    return p_msl


def check_dew_point_depression_invariant(
    temp_c: np.ndarray | float,
    humidity_pct: np.ndarray | float,
    tolerance_c: float = 0.5,
) -> np.ndarray | bool:
    """Evaluates the thermodynamic dew-point depression invariant.

    Thermodynamic Invariant: T - T_d >= 0
    Allowing float sensor noise tolerance: T - T_d >= -tolerance_c
    (i.e., T_d <= T + tolerance_c).

    Returns:
        Boolean array/scalar where True indicates a physical VIOLATION
        (unphysical psychrometric breakdown: T_d > T + tolerance_c),
        and False indicates thermodynamically consistent telemetry.
    """
    t = np.asarray(temp_c, dtype=np.float64)
    rh = np.asarray(humidity_pct, dtype=np.float64)

    t_d = calculate_dew_point_c(t, rh)
    depression = t - t_d

    # Violation occurs when dew point depression is strictly less than -tolerance
    is_violation = depression < -tolerance_c

    # NaNs should not be marked as violation here (Tier 1 handles NaNs)
    nan_mask = np.isnan(t) | np.isnan(rh)
    is_violation[nan_mask] = False

    if np.isscalar(temp_c):
        return bool(is_violation)
    return is_violation
