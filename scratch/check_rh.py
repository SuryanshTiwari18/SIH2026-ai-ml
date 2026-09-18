import pandas as pd
df_csv = pd.read_csv("data/raw/aws_telemetry_master.csv")
cs_csv = df_csv[df_csv["anomaly_type"] == "cross_sensor_inconsistency"]
print("CSV Count:", len(cs_csv))
print("CSV RH min/max:", cs_csv["humidity_pct"].min(), cs_csv["humidity_pct"].max())

df_pq = pd.read_parquet("data/raw/aws_telemetry_master.parquet")
cs_pq = df_pq[df_pq["anomaly_type"] == "cross_sensor_inconsistency"]
print("Parquet Count:", len(cs_pq))
print("Parquet RH min/max:", cs_pq["humidity_pct"].min(), cs_pq["humidity_pct"].max())
