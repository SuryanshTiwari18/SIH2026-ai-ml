import numpy as np
import pandas as pd

df = pd.read_parquet("data/raw/aws_telemetry_master.parquet")

for st_id in ["AWS_IND_C01", "AWS_IND_A01", "AWS_IND_H01", "AWS_IND_P01"]:
    st = df[(df["station_id"] == st_id) & (~df["is_anomaly"])]
    X = st[["temperature_c", "pressure_hpa", "humidity_pct"]].to_numpy()
    mu = np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)
    inv_cov = np.linalg.inv(cov + 1e-6 * np.eye(3))

    diffs = X - mu
    dists = np.sqrt(np.sum(diffs @ inv_cov * diffs, axis=1))

    p01 = np.percentile(X, 1, axis=0)
    p99 = np.percentile(X, 99, axis=0)
    p85 = np.percentile(X, 85, axis=0)
    p15 = np.percentile(X, 15, axis=0)

    # Candidate 1: High T + High RH + Low P
    pt1 = np.array([p85[0], p15[1], p85[2]])
    dm1 = np.sqrt((pt1 - mu) @ inv_cov @ (pt1 - mu))
    within1 = (pt1 >= p01).all() and (pt1 <= p99).all()

    # Candidate 2: Low T + Low RH + High P
    pt2 = np.array([p15[0], p85[1], p15[2]])
    dm2 = np.sqrt((pt2 - mu) @ inv_cov @ (pt2 - mu))
    within2 = (pt2 >= p01).all() and (pt2 <= p99).all()

    print(f"Station {st_id}:")
    print(f"  Normal DM: mean={dists.mean():.2f}, 95th={np.percentile(dists, 95):.2f}, 99th={np.percentile(dists, 99):.2f}")
    print(f"  Outlier 1: pt={pt1.round(1)}, DM={dm1:.2f}, within [p01, p99]={within1}, RH={pt1[2]:.1f}%")
    print(f"  Outlier 2: pt={pt2.round(1)}, DM={dm2:.2f}, within [p01, p99]={within2}, RH={pt2[2]:.1f}%")
