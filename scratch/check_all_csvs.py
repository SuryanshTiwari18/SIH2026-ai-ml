import glob
import pandas as pd

for path in glob.glob("data/**/*.csv", recursive=True):
    df = pd.read_csv(path)
    if "anomaly_type" in df.columns:
        cs = df[df["anomaly_type"] == "cross_sensor_inconsistency"]
        if len(cs) > 0:
            print(f"{path}: count={len(cs)}, RH min={cs['humidity_pct'].min():.2f}, max={cs['humidity_pct'].max():.2f}")
