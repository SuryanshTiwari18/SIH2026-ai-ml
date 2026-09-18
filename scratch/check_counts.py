import pandas as pd
df = pd.read_parquet("data/raw/aws_telemetry_master.parquet")
print("Anomaly type value counts:")
print(df["anomaly_type"].value_counts(dropna=False))
print("\nTotal anomalies:", df["is_anomaly"].sum())
