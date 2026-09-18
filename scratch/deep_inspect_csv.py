import os, time
import pandas as pd
import numpy as np

csv_path = "data/raw/aws_telemetry_master.csv"
print("File:", csv_path)
print("Modified time:", time.ctime(os.path.getmtime(csv_path)))
print("Size:", os.path.getsize(csv_path))

df = pd.read_csv(csv_path)
print("Shape:", df.shape)

cs = df[df["anomaly_type"] == "cross_sensor_inconsistency"]
print("\nCross-sensor inconsistency count:", len(cs))
print("RH min:", cs["humidity_pct"].min())
print("RH max:", cs["humidity_pct"].max())
print("RH > 100 count:", (cs["humidity_pct"] > 100).sum())
print("RH < 0 count:", (cs["humidity_pct"] < 0).sum())

print("\nSample 5 rows of cross_sensor_inconsistency:")
print(cs[["station_id", "timestamp", "temperature_c", "pressure_hpa", "humidity_pct", "anomaly_severity"]].head())

# Check all anomaly types for RH > 100:
print("\nAny RH > 100 across entire dataset:")
print(df[df["humidity_pct"] > 100][["station_id", "anomaly_type", "humidity_pct"]].head())
