import pandas as pd
import numpy as np

csv_path = "data/raw/aws_telemetry_master.csv"
df = pd.read_csv(csv_path)

cs = df[df["anomaly_type"] == "cross_sensor_inconsistency"]
print(f"Total cross_sensor_inconsistency rows: {len(cs)}")
print(f"RH min: {cs['humidity_pct'].min():.4f}%")
print(f"RH max: {cs['humidity_pct'].max():.4f}%")
print(f"RH outside [0, 100] count: {((cs['humidity_pct'] < 0) | (cs['humidity_pct'] > 100)).sum()}")
print(f"Temp min: {cs['temperature_c'].min():.4f} C")
print(f"Temp max: {cs['temperature_c'].max():.4f} C")
print(f"Temp outside [-30, 60] count: {((cs['temperature_c'] < -30) | (cs['temperature_c'] > 60)).sum()}")
print(f"Pressure min: {cs['pressure_hpa'].min():.4f} hPa")
print(f"Pressure max: {cs['pressure_hpa'].max():.4f} hPa")
print(f"Pressure outside [500, 1100] count: {((cs['pressure_hpa'] < 500) | (cs['pressure_hpa'] > 1100)).sum()}")

# Check station percentile bounds
out_of_station_bounds = 0
for station_id, st_df in df.groupby("station_id"):
    st_norm = st_df[st_df["is_anomaly"] == False]
    p01 = st_norm[["temperature_c", "pressure_hpa", "humidity_pct"]].quantile(0.01)
    p99 = st_norm[["temperature_c", "pressure_hpa", "humidity_pct"]].quantile(0.99)
    st_cs = st_df[st_df["anomaly_type"] == "cross_sensor_inconsistency"]
    for col in ["temperature_c", "pressure_hpa", "humidity_pct"]:
        violations = ((st_cs[col] < p01[col] - 1e-4) | (st_cs[col] > p99[col] + 1e-4)).sum()
        out_of_station_bounds += violations

print(f"Total violations of station historical [p01, p99] bounds across all 3 sensors: {out_of_station_bounds}")
