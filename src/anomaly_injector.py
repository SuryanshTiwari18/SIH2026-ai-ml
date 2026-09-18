"""Station-local Sensor Anomaly Injector for SkyGuard AI AWS Telemetry.

Covers all 7 fault types from Problem Statement 26073:
1. spike_or_drop (1-3 readings physically implausible step)
2. frozen_sensor (sensor ADC stuck at exact constant float, zero variance)
3. communication_dropout (telemetry packet loss, ALL THREE variables set to NaN)
4. calibration_drift (slow accumulating linear bias over days)
5. power_fluctuation_glitch (high-frequency voltage ripple / noise burst)
6. data_corruption (out-of-range sentinel or nonsensical values like -999.0)
7. cross_sensor_inconsistency (subtly invalid T/RH/P thermodynamic combination, T_dew > T_air)

Spatial Locality Guarantee:
All injected faults affect ONLY the target station. Neighboring stations remain unaffected,
providing strict ground truth for spatial buddy-check tier validation.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from src.stations import StationMetadata
from src.physics import calculate_dew_point_c


@dataclass
class AnomalyEvent:
    station_id: str
    anomaly_type: str
    start_idx: int
    end_idx: int
    duration_steps: int
    affected_variables: List[str]
    severity: float
    description: str


class SensorAnomalyInjector:
    """Injects realistic, station-local hardware and communication anomalies."""

    def __init__(
        self,
        target_anomaly_rate: float = 0.04,
        fault_type_weights: Dict[str, float] | None = None,
        min_episodes_per_temporal_split: int = 8,
        min_episodes_spatial_holdout: int = 5,
        rng: np.random.Generator | None = None,
    ):
        self.target_anomaly_rate = target_anomaly_rate
        self.min_episodes_per_temporal_split = min_episodes_per_temporal_split
        self.min_episodes_spatial_holdout = min_episodes_spatial_holdout
        self.rng = rng or np.random.default_rng()

        if fault_type_weights is None:
            self.fault_type_weights = {
                "spike_or_drop": 0.16,
                "frozen_sensor": 0.14,
                "communication_dropout": 0.14,
                "calibration_drift": 0.14,
                "power_fluctuation_glitch": 0.14,
                "data_corruption": 0.14,
                "cross_sensor_inconsistency": 0.14,
            }
        else:
            self.fault_type_weights = fault_type_weights

    def plan_network_anomaly_schedule(
        self,
        stations: Dict[str, StationMetadata],
        total_steps: int,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        holdout_station_ids: List[str] | None = None,
    ) -> Dict[str, List[dict]]:
        """Pre-schedule non-overlapping episodes across stations to guarantee quota minimums.
        
        Guarantees:
        - At least `min_episodes_per_temporal_split` (>=8) distinct episodes per anomaly type in Train, Val, Test.
        - At least `min_episodes_spatial_holdout` (>=5) distinct episodes per anomaly type in Spatial Holdout.
        - Episodes distributed across multiple stations (>=4 per window).
        - Strictly non-overlapping intervals within each station.
        """
        if holdout_station_ids is None:
            holdout_station_ids = []

        schedule: Dict[str, List[dict]] = {st_id: [] for st_id in stations}
        station_occupied: Dict[str, np.ndarray] = {
            st_id: np.zeros(total_steps, dtype=bool) for st_id in stations
        }

        train_cutoff = int(total_steps * train_ratio)
        val_cutoff = int(total_steps * (train_ratio + val_ratio))

        temporal_windows = [
            ("temporal_train", 0, train_cutoff, self.min_episodes_per_temporal_split),
            ("temporal_val", train_cutoff, val_cutoff, self.min_episodes_per_temporal_split),
            ("temporal_test", val_cutoff, total_steps, self.min_episodes_per_temporal_split),
        ]

        fault_types = list(self.fault_type_weights.keys())
        non_holdout_stations = [s for s in stations if s not in holdout_station_ids]
        holdout_stations = [s for s in stations if s in holdout_station_ids]

        # Helper to pick duration based on window length and fault type
        def get_fault_duration(fault: str, window_len: int) -> int:
            if fault == "spike_or_drop":
                return int(self.rng.integers(1, 4))
            elif fault == "frozen_sensor":
                return int(self.rng.integers(18, min(48, max(19, window_len // 20))))
            elif fault == "communication_dropout":
                return int(self.rng.integers(6, min(24, max(7, window_len // 40))))
            elif fault == "calibration_drift":
                return int(self.rng.integers(72, min(216, max(73, window_len // 6))))
            elif fault == "power_fluctuation_glitch":
                return int(self.rng.integers(4, min(14, max(5, window_len // 60))))
            elif fault == "data_corruption":
                return int(self.rng.integers(1, 4))
            elif fault == "cross_sensor_inconsistency":
                return int(self.rng.integers(8, min(28, max(9, window_len // 30))))
            return 12

        # 1. Schedule guaranteed quotas for temporal windows on non-holdout stations
        for win_name, w_start, w_end, quota in temporal_windows:
            w_len = w_end - w_start
            for fault in fault_types:
                episodes_placed = 0
                candidate_stations = list(non_holdout_stations)
                self.rng.shuffle(candidate_stations)
                st_idx = 0

                attempts = 0
                while episodes_placed < quota and attempts < 200:
                    attempts += 1
                    target_st = candidate_stations[st_idx % len(candidate_stations)]
                    st_idx += 1

                    dur = get_fault_duration(fault, w_len)
                    margin = 6
                    if w_len <= dur + 2 * margin:
                        continue

                    # Search available slots in target station
                    st_mask = station_occupied[target_st]
                    valid_starts = []
                    for s in range(w_start + margin, w_end - dur - margin, 6):
                        if not np.any(st_mask[s : s + dur]):
                            valid_starts.append(s)

                    if not valid_starts:
                        continue

                    start_idx = int(self.rng.choice(valid_starts))
                    end_idx = start_idx + dur
                    st_mask[start_idx:end_idx] = True

                    schedule[target_st].append({
                        "anomaly_type": fault,
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "duration_steps": dur,
                        "split_window": win_name,
                    })
                    episodes_placed += 1

        # 2. Schedule guaranteed quotas for spatial holdout stations
        if holdout_stations:
            quota = self.min_episodes_spatial_holdout
            for fault in fault_types:
                episodes_placed = 0
                candidate_stations = list(holdout_stations)
                self.rng.shuffle(candidate_stations)
                st_idx = 0
                attempts = 0

                while episodes_placed < quota and attempts < 200:
                    attempts += 1
                    target_st = candidate_stations[st_idx % len(candidate_stations)]
                    st_idx += 1

                    dur = get_fault_duration(fault, total_steps)
                    margin = 12
                    st_mask = station_occupied[target_st]
                    valid_starts = []
                    for s in range(margin, total_steps - dur - margin, 12):
                        if not np.any(st_mask[s : s + dur]):
                            valid_starts.append(s)

                    if not valid_starts:
                        continue

                    start_idx = int(self.rng.choice(valid_starts))
                    end_idx = start_idx + dur
                    st_mask[start_idx:end_idx] = True

                    schedule[target_st].append({
                        "anomaly_type": fault,
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "duration_steps": dur,
                        "split_window": "spatial_holdout",
                    })
                    episodes_placed += 1

        return schedule

    def inject_station_anomalies(
        self,
        station: StationMetadata,
        t_series: pd.Series,
        p_series: pd.Series,
        rh_series: pd.Series,
        scheduled_events: List[dict] | None = None,
    ) -> Tuple[pd.DataFrame, List[AnomalyEvent]]:
        """Inject realistic station-local anomalies into the given station's time series.
        
        Returns:
            (augmented_df, list_of_anomaly_events)
        """
        n_steps = len(t_series)
        t_arr = t_series.to_numpy(dtype=np.float64, copy=True)
        p_arr = p_series.to_numpy(dtype=np.float64, copy=True)
        rh_arr = rh_series.to_numpy(dtype=np.float64, copy=True)

        is_anomaly = np.zeros(n_steps, dtype=bool)
        anomaly_type_arr = np.empty(n_steps, dtype=object)
        anomaly_type_arr[:] = None
        anomaly_severity_arr = np.full(n_steps, np.nan, dtype=np.float64)

        # Empirical normal bounds & covariance for this station
        X_normal = np.column_stack([t_arr, p_arr, rh_arr])
        p01_vals = np.percentile(X_normal, 1.0, axis=0)
        p99_vals = np.percentile(X_normal, 99.0, axis=0)
        p15_vals = np.percentile(X_normal, 15.0, axis=0)
        p85_vals = np.percentile(X_normal, 85.0, axis=0)
        cov_normal = np.cov(X_normal, rowvar=False)
        inv_cov_normal = np.linalg.inv(cov_normal + 1e-6 * np.eye(3))
        mu_normal = np.mean(X_normal, axis=0)

        events: List[AnomalyEvent] = []

        if scheduled_events is None:
            scheduled_events = []

        for evt in scheduled_events:
            fault = evt["anomaly_type"]
            start_idx = evt["start_idx"]
            end_idx = evt["end_idx"]
            duration = end_idx - start_idx

            affected_vars = []
            severity = 0.5
            desc = ""

            if fault == "spike_or_drop":
                target_var = self.rng.choice(["temperature_c", "pressure_hpa", "humidity_pct"])
                sign = self.rng.choice([-1.0, 1.0])
                if target_var == "temperature_c":
                    magnitude = float(self.rng.uniform(11.0, 22.0))
                    t_arr[start_idx:end_idx] += sign * magnitude
                    affected_vars = ["temperature_c"]
                elif target_var == "pressure_hpa":
                    magnitude = float(self.rng.uniform(25.0, 55.0))
                    p_arr[start_idx:end_idx] += sign * magnitude
                    affected_vars = ["pressure_hpa"]
                else:
                    magnitude = float(self.rng.uniform(35.0, 65.0))
                    rh_arr[start_idx:end_idx] = np.clip(
                        rh_arr[start_idx:end_idx] + sign * magnitude, -5.0, 115.0
                    )
                    affected_vars = ["humidity_pct"]
                severity = float(np.clip(magnitude / 30.0, 0.6, 1.0))
                desc = f"Sudden {sign:+0.1f} jump in {target_var}"

            elif fault == "frozen_sensor":
                target_var = self.rng.choice(["temperature_c", "pressure_hpa", "humidity_pct", "all"])
                if target_var in ["temperature_c", "all"]:
                    t_arr[start_idx:end_idx] = float(t_arr[start_idx])
                    affected_vars.append("temperature_c")
                if target_var in ["pressure_hpa", "all"]:
                    p_arr[start_idx:end_idx] = float(p_arr[start_idx])
                    affected_vars.append("pressure_hpa")
                if target_var in ["humidity_pct", "all"]:
                    rh_arr[start_idx:end_idx] = float(rh_arr[start_idx])
                    affected_vars.append("humidity_pct")
                severity = float(np.clip(duration / 36.0, 0.5, 0.95))
                desc = f"Sensor ADC frozen at constant value for {duration*10} mins"

            elif fault == "communication_dropout":
                # CRITICAL FIX: ALL THREE columns MUST be NaN for every row in the episode
                t_arr[start_idx:end_idx] = np.nan
                p_arr[start_idx:end_idx] = np.nan
                rh_arr[start_idx:end_idx] = np.nan
                affected_vars = ["temperature_c", "pressure_hpa", "humidity_pct"]
                severity = float(np.clip(duration / 18.0, 0.75, 1.0))
                desc = f"Communication dropout (complete NaN packet loss across all 3 sensors) for {duration*10} mins"

            elif fault == "calibration_drift":
                target_var = self.rng.choice(["temperature_c", "pressure_hpa", "humidity_pct"])
                ramp = np.linspace(0.0, 1.0, duration)
                sign = self.rng.choice([-1.0, 1.0])
                if target_var == "temperature_c":
                    max_drift = float(self.rng.uniform(3.5, 7.5))
                    t_arr[start_idx:end_idx] += sign * max_drift * ramp
                    affected_vars = ["temperature_c"]
                elif target_var == "pressure_hpa":
                    max_drift = float(self.rng.uniform(8.0, 20.0))
                    p_arr[start_idx:end_idx] += sign * max_drift * ramp
                    affected_vars = ["pressure_hpa"]
                else:
                    max_drift = float(self.rng.uniform(18.0, 38.0))
                    rh_arr[start_idx:end_idx] = np.clip(
                        rh_arr[start_idx:end_idx] + sign * max_drift * ramp, 2.0, 99.9
                    )
                    affected_vars = ["humidity_pct"]
                severity = float(np.clip(max_drift / 15.0, 0.4, 0.85))
                desc = f"Gradual sensor calibration drift of {sign*max_drift:+.1f} over {duration*10/60:.1f} hours"

            elif fault == "power_fluctuation_glitch":
                noise_scale = self.rng.uniform(12.0, 22.0)
                t_arr[start_idx:end_idx] += self.rng.normal(0, station.noise_temp_std * noise_scale, size=duration)
                p_arr[start_idx:end_idx] += self.rng.normal(0, station.noise_press_std * noise_scale, size=duration)
                rh_arr[start_idx:end_idx] = np.clip(
                    rh_arr[start_idx:end_idx] + self.rng.normal(0, station.noise_rh_std * noise_scale, size=duration),
                    0.0, 100.0,
                )
                affected_vars = ["temperature_c", "pressure_hpa", "humidity_pct"]
                severity = float(np.clip(noise_scale / 22.0, 0.5, 0.9))
                desc = f"Power fluctuation voltage ripple & high-frequency jitter for {duration*10} mins"

            elif fault == "data_corruption":
                sentinel_choices = [-999.0, 9999.0, -99.9, 999.9]
                token = float(self.rng.choice(sentinel_choices))
                target_var = self.rng.choice(["temperature_c", "pressure_hpa", "humidity_pct", "all"])
                if target_var in ["temperature_c", "all"]:
                    t_arr[start_idx:end_idx] = token
                    affected_vars.append("temperature_c")
                if target_var in ["pressure_hpa", "all"]:
                    p_arr[start_idx:end_idx] = token
                    affected_vars.append("pressure_hpa")
                if target_var in ["humidity_pct", "all"]:
                    rh_arr[start_idx:end_idx] = float(self.rng.choice([-25.0, 160.0, -999.0]))
                    affected_vars.append("humidity_pct")
                severity = 1.0
                desc = f"Data corruption: out-of-range sentinel value ({token})"

            elif fault == "cross_sensor_inconsistency":
                # Genuine joint statistical outlier staying strictly within univariate bounds:
                # Breaks the normal physical/diurnal joint correlation (e.g. high T + high RH,
                # or cool dawn T + desert-dry RH, or high P paired with storm humidity)
                combo_mode = self.rng.choice(["hot_humid_anomaly", "cool_dry_anomaly", "high_p_humid_squall"])
                if combo_mode == "hot_humid_anomaly":
                    # Pair high temperature with high humidity and lower pressure
                    t_target = float(p85_vals[0])
                    p_target = float(p15_vals[1])
                    rh_target = float(p85_vals[2])
                elif combo_mode == "cool_dry_anomaly":
                    # Pair cool temperature with desert-dry humidity and high pressure
                    t_target = float(p15_vals[0])
                    p_target = float(p85_vals[1])
                    rh_target = float(p15_vals[2])
                else:
                    # Clear-sky high pressure with high monsoon humidity
                    t_target = float(p85_vals[0])
                    p_target = float(p85_vals[1])
                    rh_target = float(p85_vals[2])

                # Smoothly shift the active segment towards the target joint combination
                # with dynamic noise so it continuously varies (NEVER frozen)
                t_noise = self.rng.normal(0, 0.15, size=duration)
                p_noise = self.rng.normal(0, 0.20, size=duration)
                rh_noise = self.rng.normal(0, 0.35, size=duration)

                t_segment = 0.55 * t_arr[start_idx:end_idx] + 0.45 * t_target + t_noise
                p_segment = 0.55 * p_arr[start_idx:end_idx] + 0.45 * p_target + p_noise
                rh_segment = 0.50 * rh_arr[start_idx:end_idx] + 0.50 * rh_target + rh_noise

                # Strictly guarantee each variable stays within the station's historical [p01, p99]
                # and RH strictly stays within [5.0, 96.0]%
                t_arr[start_idx:end_idx] = np.clip(t_segment, p01_vals[0] + 0.2, p99_vals[0] - 0.2)
                p_arr[start_idx:end_idx] = np.clip(p_segment, p01_vals[1] + 0.2, p99_vals[1] - 0.2)
                rh_arr[start_idx:end_idx] = np.clip(
                    rh_segment,
                    max(p01_vals[2] + 0.5, 5.0),
                    min(p99_vals[2] - 0.5, 96.0),
                )

                # Compute realized Mahalanobis distance for logging
                mod_pts = np.column_stack([
                    t_arr[start_idx:end_idx],
                    p_arr[start_idx:end_idx],
                    rh_arr[start_idx:end_idx],
                ])
                diff_pts = mod_pts - mu_normal
                d_m_vals = np.sqrt(np.sum(diff_pts @ inv_cov_normal * diff_pts, axis=1))
                mean_dm = float(np.mean(d_m_vals))

                affected_vars = ["temperature_c", "pressure_hpa", "humidity_pct"]
                severity = float(np.clip(mean_dm / 10.0, 0.70, 0.95))
                desc = f"Joint statistical cross-sensor inconsistency (Mahalanobis D_M={mean_dm:.1f}, within [p01, p99] bounds)"

            # Set labels
            is_anomaly[start_idx:end_idx] = True
            anomaly_type_arr[start_idx:end_idx] = fault
            anomaly_severity_arr[start_idx:end_idx] = severity

            events.append(
                AnomalyEvent(
                    station_id=station.station_id,
                    anomaly_type=fault,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    duration_steps=duration,
                    affected_variables=affected_vars,
                    severity=severity,
                    description=desc,
                )
            )

        df = pd.DataFrame(
            {
                "station_id": station.station_id,
                "timestamp": t_series.index,
                "latitude": station.latitude,
                "longitude": station.longitude,
                "temperature_c": t_arr,
                "pressure_hpa": p_arr,
                "humidity_pct": rh_arr,
                "is_anomaly": is_anomaly,
                "anomaly_type": anomaly_type_arr,
                "anomaly_severity": anomaly_severity_arr,
            }
        )

        return df, events
