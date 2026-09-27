import sys
sys.path.insert(0, ".")
import math
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.physics import reduce_pressure_to_msl, check_dew_point_depression_invariant
from src.stations import INDIAN_AWS_STATIONS, get_station_catalog
from src.tier1_qc import apply_tier1b_climatology_qc

SENSOR_COLS = ["temperature_c", "pressure_hpa", "humidity_pct"]

# Load existing stats
mahal_stats = joblib.load("models/mahalanobis_stats.joblib")
station_hour_stats = mahal_stats.get("station_hour", {})
zone_hour_stats = mahal_stats.get("zone_hour", {})
hour_fallback = mahal_stats.get("hour_fallback", {})
global_fallback = mahal_stats.get("global_fallback", {})

catalog = get_station_catalog()


class Rolling14DayMahalanobisBaseline:
    """Online 14-Day Exponential Moving Average (EMA) Mahalanobis Baseline.

    Replaces static per-(station, hour) lookup with adaptive baseline that tracks
    seasonal progressions while remaining robust to sensor anomalies.
    """

    def __init__(self, alpha: float = 0.0235, clip_threshold: float = 6.5):
        self.alpha = alpha
        self.clip_threshold = clip_threshold
        # State: (station_id, hour) -> {"mu": np.ndarray, "cov": np.ndarray, "inv_cov": np.ndarray}
        self.state = {}

    def _get_prior(self, station_id: str, hour: int) -> dict:
        if (station_id, hour) in station_hour_stats:
            p = station_hour_stats[(station_id, hour)]
            return {"mu": p["mu"].copy(), "cov": p["cov"].copy(), "inv_cov": p["inv_cov"].copy()}

        st_meta = catalog.get(station_id)
        zone = st_meta.climate_zone if st_meta else "plains"
        if (zone, hour) in zone_hour_stats:
            p = zone_hour_stats[(zone, hour)]
            return {"mu": p["mu"].copy(), "cov": p["cov"].copy(), "inv_cov": p["inv_cov"].copy()}

        if hour in hour_fallback:
            p = hour_fallback[hour]
            return {"mu": p["mu"].copy(), "cov": p["cov"].copy(), "inv_cov": p["inv_cov"].copy()}

        p = global_fallback
        return {"mu": p["mu"].copy(), "cov": p["cov"].copy(), "inv_cov": p["inv_cov"].copy()}

    def get_or_init_state(self, station_id: str, hour: int) -> dict:
        key = (station_id, hour)
        if key not in self.state:
            self.state[key] = self._get_prior(station_id, hour)
        return self.state[key]

    def compute_distance_and_update(
        self, station_id: str, hour: int, x: np.ndarray, update: bool = True
    ) -> float:
        st = self.get_or_init_state(station_id, hour)
        diff = x - st["mu"]
        dm_sq = float(diff @ st["inv_cov"] @ diff)
        dm = math.sqrt(max(0.0, dm_sq))

        if update and not np.isnan(x).any():
            # Robust EMA: downweight severe outliers from shifting baseline
            w = self.alpha if dm <= self.clip_threshold else self.alpha * 0.1
            new_mu = (1.0 - w) * st["mu"] + w * x
            v = x - new_mu
            new_cov = (1.0 - w) * st["cov"] + w * np.outer(v, v) + 1e-4 * np.eye(3)
            new_inv_cov = np.linalg.pinv(new_cov)
            self.state[(station_id, hour)] = {
                "mu": new_mu,
                "cov": new_cov,
                "inv_cov": new_inv_cov,
            }

        return dm


def test_splits():
    for split in ["test", "spatial_holdout"]:
        df = pd.read_parquet(f"data/tier4_results/{split}.parquet")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

        # 1. Tier 1B Climatology check
        df = apply_tier1b_climatology_qc(df)

        # 2. Dew point depression invariant check
        dp_violation = check_dew_point_depression_invariant(
            df["temperature_c"].values, df["humidity_pct"].values, tolerance_c=0.5
        )
        df["dew_point_flagged"] = dp_violation

        # 3. Rolling Mahalanobis
        rolling_engine = Rolling14DayMahalanobisBaseline()
        hours = df["timestamp"].dt.hour.values
        stations = df["station_id"].values
        x_mat = df[SENSOR_COLS].values
        n = len(df)
        dm_rolling = np.zeros(n)

        for i in range(n):
            if np.isnan(x_mat[i]).any():
                dm_rolling[i] = 0.0
            else:
                dm_rolling[i] = rolling_engine.compute_distance_and_update(
                    stations[i], int(hours[i]), x_mat[i], update=True
                )

        df["mahalanobis_dist_rolling"] = dm_rolling

        print(f"=== {split.upper()} STATS ===")
        print(f"Tier 1B newly flagged: {df['tier1b_flagged'].sum()} rows")
        print(f"Dew point violations: {df['dew_point_flagged'].sum()} rows")
        print(f"Old DM mean on normal: {df[~df['is_anomaly']]['mahalanobis_dist'].mean():.2f}")
        print(f"Rolling DM mean on normal: {df[~df['is_anomaly']]['mahalanobis_dist_rolling'].mean():.2f}")
        print()


if __name__ == "__main__":
    test_splits()
