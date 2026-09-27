"""
Evaluates Tier 1B + Tier 3 checks on 7 Real Cities (2006-2025).
Computes monthly flag rates for Tier 1B and combined checks.
"""

import sys
sys.path.insert(0, ".")
import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.physics import check_dew_point_depression_invariant
from src.tier1_qc import apply_tier1b_climatology_qc
from src.tier3_multivariate_spatial import Rolling14DayMahalanobisBaseline

CITIES = ["delhi", "leh", "jaisalmer", "mumbai", "chennai", "guwahati", "bengaluru"]


def evaluate_real_cities():
    summary_by_city = {}
    month_matrix = {}

    for city in CITIES:
        pq_path = Path(f"data/real_climatology/{city}.parquet")
        df = pd.read_parquet(pq_path)
        df["time"] = pd.to_datetime(df["time"])
        df["month"] = df["time"].dt.month
        df["station_id"] = city

        # 1. Tier 1B Climatology Check
        df = apply_tier1b_climatology_qc(df)

        # 2. Dew Point Invariant Check
        df["dew_point_flagged"] = check_dew_point_depression_invariant(
            df["temperature_c"].values, df["humidity_pct"].values, tolerance_c=0.5
        )

        # 3. Rolling Mahalanobis Check
        # For single-station real history, rolling baseline tracks seasonal diurnal cycle
        rolling_engine = Rolling14DayMahalanobisBaseline(alpha=0.0235)
        dm_vals = np.zeros(len(df))
        for i in range(len(df)):
            x = np.array([df["temperature_c"].iloc[i], df["pressure_hpa"].iloc[i], df["humidity_pct"].iloc[i]])
            dm_vals[i] = rolling_engine.compute_and_update(city, 12, x, update=True)
        df["mahalanobis_dist_rolling"] = dm_vals
        df["mahalanobis_flagged"] = dm_vals > 6.5

        # Combined Tier 1B + Tier 3 flag
        df["any_flagged"] = df["tier1b_flagged"] | df["dew_point_flagged"]

        n_total = len(df)
        t1b_flagged = int(df["tier1b_flagged"].sum())
        dp_flagged = int(df["dew_point_flagged"].sum())
        any_flagged = int(df["any_flagged"].sum())

        monthly_t1b_rates = {}
        for m in range(1, 13):
            sub = df[df["month"] == m]
            m_total = len(sub)
            m_t1b = int(sub["tier1b_flagged"].sum())
            monthly_t1b_rates[str(m)] = {
                "total_days": m_total,
                "t1b_flagged": m_t1b,
                "t1b_rate_pct": round(m_t1b / m_total * 100.0, 2),
            }

        month_matrix[city] = monthly_t1b_rates
        summary_by_city[city] = {
            "total_days": n_total,
            "elevation_m": float(df["elevation"].iloc[0]),
            "tier1b_flagged": t1b_flagged,
            "tier1b_rate_pct": round(t1b_flagged / n_total * 100.0, 4),
            "dew_point_flagged": dp_flagged,
            "dew_point_rate_pct": round(dp_flagged / n_total * 100.0, 4),
            "mahalanobis_rolling_flagged": int(df["mahalanobis_flagged"].sum()),
            "mahalanobis_rolling_rate_pct": round(int(df["mahalanobis_flagged"].sum()) / n_total * 100.0, 4),
        }

    print("=== REAL CITIES EVALUATION SUMMARY (2006-2025, 7,305 Days Per City) ===")
    print(f"{'City':12s} | {'Elevation':9s} | {'Tier 1B Flagged (%)':20s} | {'Dew Point Flagged (%)':22s} | {'Rolling Mahal Flagged (%)':26s}")
    print("-" * 100)
    for c in CITIES:
        s = summary_by_city[c]
        print(f"{c.capitalize():12s} | {s['elevation_m']:7.1f}m | {s['tier1b_flagged']:4d} ({s['tier1b_rate_pct']:5.2f}%)       | {s['dew_point_flagged']:4d} ({s['dew_point_rate_pct']:5.2f}%)         | {s['mahalanobis_rolling_flagged']:4d} ({s['mahalanobis_rolling_rate_pct']:5.2f}%)")

    print("\n=== TIER 1B FLAG RATE BY MONTH (MONTH 1 TO 12) ===")
    header = f"{'City':12s} | " + " | ".join([f"M{m:02d}" for m in range(1, 13)])
    print(header)
    print("-" * 85)
    for c in CITIES:
        row_str = f"{c.capitalize():12s} | "
        vals = [f"{month_matrix[c][str(m)]['t1b_rate_pct']:4.1f}%" for m in range(1, 13)]
        row_str += " | ".join(vals)
        print(row_str)

    with open("data/real_climatology/real_cities_eval.json", "w") as f:
        json.dump({"summary": summary_by_city, "monthly": month_matrix}, f, indent=2)

    return summary_by_city, month_matrix


if __name__ == "__main__":
    evaluate_real_cities()
