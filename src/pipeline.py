"""Master dataset synthesis pipeline for SkyGuard AI.

Orchestrates:
1. Multi-station physics-based signal generation across Indian climate zones.
2. Injection of physically self-consistent extreme weather events (labeled normal).
3. Injection of station-local sensor faults across all 7 fault types (labeled anomaly).
4. Chronological train/val/test temporal splitting without leakage.
5. Spatial holdout station splitting for zero-shot generalization testing.
6. Export to Parquet, CSV, and metadata documentation.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from src.config import GeneratorConfig
from src.stations import get_station_catalog, StationMetadata
from src.base_signals import generate_time_index, generate_station_base_signal
from src.extreme_weather import schedule_network_extreme_events, apply_scheduled_extreme_events
from src.anomaly_injector import SensorAnomalyInjector, AnomalyEvent


class DatasetPipeline:
    """Master orchestrator for synthetic AWS dataset generation."""

    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.rng = np.random.default_rng(config.generation.random_seed)
        self.stations = get_station_catalog(config.stations.include_climate_zones)
        self.time_index = generate_time_index(
            start_date=config.generation.start_date,
            duration_days=config.generation.duration_days,
            sampling_interval_minutes=config.generation.sampling_interval_minutes,
        )
        self.injector = SensorAnomalyInjector(
            target_anomaly_rate=config.anomalies.target_anomaly_rate,
            fault_type_weights=config.anomalies.fault_type_weights,
            min_episodes_per_temporal_split=config.anomalies.min_episodes_per_temporal_split,
            min_episodes_spatial_holdout=config.anomalies.min_episodes_spatial_holdout,
            rng=self.rng,
        )

    def generate(self) -> Tuple[pd.DataFrame, Dict, List[AnomalyEvent]]:
        """Run end-to-end multi-station generation."""
        station_dfs = []
        all_anomaly_events: List[AnomalyEvent] = []
        extreme_event_logs = []

        total_steps = len(self.time_index)
        print(f"Generating synthetic telemetry for {len(self.stations)} stations over {self.config.generation.duration_days} days ({total_steps} steps)...")

        # 1. Pre-schedule network-wide extreme weather events
        extreme_weather_schedule = {}
        if self.config.extreme_weather.enabled:
            extreme_weather_schedule = schedule_network_extreme_events(
                stations=self.stations,
                total_steps=total_steps,
                event_prob_per_station=self.config.extreme_weather.event_probability_per_station,
                min_network_events=self.config.extreme_weather.min_network_events,
                event_types=self.config.extreme_weather.event_types,
                rng=self.rng,
            )

        # 2. Pre-schedule quota-guaranteed anomalies across splits and stations
        anomaly_schedule = {}
        if self.config.anomalies.enabled:
            anomaly_schedule = self.injector.plan_network_anomaly_schedule(
                stations=self.stations,
                total_steps=total_steps,
                train_ratio=self.config.splits.train_ratio,
                val_ratio=self.config.splits.val_ratio,
                holdout_station_ids=self.config.stations.spatial_holdout_station_ids,
            )

        # 3. Synthesize telemetry station-by-station
        for station_id, station in self.stations.items():
            # Base normal signals
            t_ser, p_ser, rh_ser = generate_station_base_signal(
                station=station,
                time_index=self.time_index,
                rng=self.rng,
            )

            # Genuine extreme weather (labeled is_anomaly=False)
            if self.config.extreme_weather.enabled:
                st_ext_events = extreme_weather_schedule.get(station_id, [])
                t_ser, p_ser, rh_ser, ext_logs = apply_scheduled_extreme_events(
                    t_series=t_ser,
                    p_series=p_ser,
                    rh_series=rh_ser,
                    events=st_ext_events,
                )
                extreme_event_logs.extend(ext_logs)

            # Anomaly injection (labeled is_anomaly=True, station-local)
            if self.config.anomalies.enabled:
                st_anom_events = anomaly_schedule.get(station_id, [])
                st_df, events = self.injector.inject_station_anomalies(
                    station=station,
                    t_series=t_ser,
                    p_series=p_ser,
                    rh_series=rh_ser,
                    scheduled_events=st_anom_events,
                )
                all_anomaly_events.extend(events)
            else:
                st_df = pd.DataFrame(
                    {
                        "station_id": station.station_id,
                        "timestamp": self.time_index,
                        "latitude": station.latitude,
                        "longitude": station.longitude,
                        "temperature_c": t_ser.to_numpy(),
                        "pressure_hpa": p_ser.to_numpy(),
                        "humidity_pct": rh_ser.to_numpy(),
                        "is_anomaly": False,
                        "anomaly_type": None,
                        "anomaly_severity": np.nan,
                    }
                )

            station_dfs.append(st_df)

        # Combine and order chronologically
        master_df = pd.concat(station_dfs, ignore_index=True)
        master_df = master_df.sort_values(by=["timestamp", "station_id"]).reset_index(drop=True)

        metadata = {
            "project": "SkyGuard AI (SIH 2026 Problem Statement 26073)",
            "description": "Physics-informed Automatic Weather Station (AWS) synthetic sensor dataset",
            "generation_parameters": {
                "start_date": self.config.generation.start_date,
                "duration_days": self.config.generation.duration_days,
                "sampling_interval_minutes": self.config.generation.sampling_interval_minutes,
                "random_seed": self.config.generation.random_seed,
                "total_records": len(master_df),
                "total_stations": len(self.stations),
                "total_timesteps": len(self.time_index),
            },
            "stations": {
                st_id: {
                    "name": st.name,
                    "state": st.state,
                    "latitude": st.latitude,
                    "longitude": st.longitude,
                    "altitude_m": st.altitude_m,
                    "climate_zone": st.climate_zone,
                }
                for st_id, st in self.stations.items()
            },
            "anomaly_summary": {
                "total_anomalies": int(master_df["is_anomaly"].sum()),
                "anomaly_percentage": float(master_df["is_anomaly"].mean() * 100.0),
                "breakdown_by_type": master_df["anomaly_type"].value_counts(dropna=True).to_dict(),
            },
            "extreme_weather_summary": {
                "total_events": len(extreme_event_logs),
                "stations_affected": list(set(e["station_id"] for e in extreme_event_logs)),
                "event_types_present": list(set(e["event_type"] for e in extreme_event_logs)),
                "events": extreme_event_logs,
            },
        }

        return master_df, metadata, all_anomaly_events

    def split_and_save(
        self,
        master_df: pd.DataFrame,
        metadata: Dict,
        output_dir: str | Path | None = None,
    ) -> Dict[str, Path]:
        """Split dataset into temporal train/val/test and spatial holdout, saving all artifacts."""
        out_path = Path(output_dir or self.config.output.output_dir)
        raw_dir = out_path / "raw"
        splits_dir = out_path / "splits"

        raw_dir.mkdir(parents=True, exist_ok=True)
        splits_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # 1. Save Master Dataset
        parquet_path = raw_dir / "aws_telemetry_master.parquet"
        csv_path = raw_dir / "aws_telemetry_master.csv"
        meta_path = raw_dir / "generation_metadata.json"

        if self.config.output.save_parquet:
            master_df.to_parquet(parquet_path, index=False, engine="pyarrow")
            saved_files["master_parquet"] = parquet_path

        if self.config.output.save_csv:
            master_df.to_csv(csv_path, index=False)
            saved_files["master_csv"] = csv_path

        # 2. Spatial Holdout Split (4 stations: 1 coastal, 1 arid, 1 hill, 1 plains)
        holdout_ids = self.config.stations.spatial_holdout_station_ids
        is_holdout = master_df["station_id"].isin(holdout_ids)
        spatial_holdout_df = master_df[is_holdout].reset_index(drop=True)
        train_pool_df = master_df[~is_holdout].reset_index(drop=True)

        if len(spatial_holdout_df) > 0:
            holdout_pqt = splits_dir / "spatial_holdout_stations.parquet"
            spatial_holdout_df.to_parquet(holdout_pqt, index=False)
            saved_files["spatial_holdout_parquet"] = holdout_pqt

        # 3. Chronological Temporal Splits on remaining stations
        unique_timestamps = np.sort(train_pool_df["timestamp"].unique())
        n_times = len(unique_timestamps)

        train_cutoff_idx = int(n_times * self.config.splits.train_ratio)
        val_cutoff_idx = int(n_times * (self.config.splits.train_ratio + self.config.splits.val_ratio))

        train_cutoff_time = unique_timestamps[train_cutoff_idx]
        val_cutoff_time = unique_timestamps[val_cutoff_idx]

        train_df = train_pool_df[train_pool_df["timestamp"] < train_cutoff_time].reset_index(drop=True)
        val_df = train_pool_df[
            (train_pool_df["timestamp"] >= train_cutoff_time) & (train_pool_df["timestamp"] < val_cutoff_time)
        ].reset_index(drop=True)
        test_df = train_pool_df[train_pool_df["timestamp"] >= val_cutoff_time].reset_index(drop=True)

        # Save temporal splits
        train_pqt = splits_dir / "train.parquet"
        val_pqt = splits_dir / "val.parquet"
        test_pqt = splits_dir / "test.parquet"

        train_df.to_parquet(train_pqt, index=False)
        val_df.to_parquet(val_pqt, index=False)
        test_df.to_parquet(test_pqt, index=False)

        saved_files["train_parquet"] = train_pqt
        saved_files["val_parquet"] = val_pqt
        saved_files["test_parquet"] = test_pqt

        # Also save CSV for test split to ease manual inspection
        test_csv = splits_dir / "test.csv"
        test_df.to_csv(test_csv, index=False)
        saved_files["test_csv"] = test_csv

        # Record split boundaries into metadata
        metadata["splits"] = {
            "spatial_holdout_stations": holdout_ids,
            "spatial_holdout_records": len(spatial_holdout_df),
            "temporal_train": {
                "records": len(train_df),
                "start_time": str(train_df["timestamp"].min()),
                "end_time": str(train_df["timestamp"].max()),
                "anomaly_percentage": float(train_df["is_anomaly"].mean() * 100.0),
            },
            "temporal_val": {
                "records": len(val_df),
                "start_time": str(val_df["timestamp"].min()),
                "end_time": str(val_df["timestamp"].max()),
                "anomaly_percentage": float(val_df["is_anomaly"].mean() * 100.0),
            },
            "temporal_test": {
                "records": len(test_df),
                "start_time": str(test_df["timestamp"].min()),
                "end_time": str(test_df["timestamp"].max()),
                "anomaly_percentage": float(test_df["is_anomaly"].mean() * 100.0),
            },
        }

        if self.config.output.save_metadata_json:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            saved_files["metadata_json"] = meta_path

        print(f"Data generation complete! Saved master and split files to {out_path.absolute()}")
        return saved_files
