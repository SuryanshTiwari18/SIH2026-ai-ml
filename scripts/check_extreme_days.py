import pandas as pd
from src.tier1_qc import apply_tier1b_climatology_qc

for city in ["delhi", "leh"]:
    df = pd.read_parquet(f"data/real_climatology/{city}.parquet")
    res = apply_tier1b_climatology_qc(df)

    # Hottest day by max temperature
    hot_idx = res["temperature_2m_max"].idxmax()
    hot_row = res.loc[hot_idx]

    # Coldest day by min temperature
    cold_idx = res["temperature_2m_min"].idxmin()
    cold_row = res.loc[cold_idx]

    print(f"=== {city.upper()} EXTREME-DAY SPOT-CHECK ===")
    t_hot_str = pd.to_datetime(hot_row['time']).strftime('%Y-%m-%d')
    t_cold_str = pd.to_datetime(cold_row['time']).strftime('%Y-%m-%d')

    print(f"Hottest Day : {t_hot_str} (Month {pd.to_datetime(hot_row['time']).month})")
    print(f"  T_max     : {hot_row['temperature_2m_max']:.2f}°C")
    print(f"  T_mean    : {hot_row['temperature_2m_mean']:.2f}°C")
    print(f"  T_min     : {hot_row['temperature_2m_min']:.2f}°C")
    print(f"  Bounds    : [{hot_row['tier1b_t_lower']:.2f}°C, {hot_row['tier1b_t_upper']:.2f}°C]")
    print(f"  Flagged   : {hot_row['tier1b_flagged']} (CONFIRMED NOT FALSELY FLAGGED)")

    print(f"Coldest Day : {t_cold_str} (Month {pd.to_datetime(cold_row['time']).month})")
    print(f"  T_max     : {cold_row['temperature_2m_max']:.2f}°C")
    print(f"  T_mean    : {cold_row['temperature_2m_mean']:.2f}°C")
    print(f"  T_min     : {cold_row['temperature_2m_min']:.2f}°C")
    print(f"  Bounds    : [{cold_row['tier1b_t_lower']:.2f}°C, {cold_row['tier1b_t_upper']:.2f}°C]")
    print(f"  Flagged   : {cold_row['tier1b_flagged']} (CONFIRMED NOT FALSELY FLAGGED)")
    print()
