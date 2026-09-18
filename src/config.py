"""Configuration models and loader for SkyGuard AI synthetic generator."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import yaml
import json


@dataclass
class GenerationSettings:
    start_date: str = "2026-06-01 00:00:00"
    duration_days: int = 60
    sampling_interval_minutes: int = 10
    random_seed: int = 42


@dataclass
class StationsSettings:
    include_climate_zones: List[str] = field(
        default_factory=lambda: ["coastal", "arid", "hill", "plains"]
    )
    spatial_holdout_station_ids: List[str] = field(
        default_factory=lambda: [
            "AWS_IND_C04",  # Coastal (Puri)
            "AWS_IND_A03",  # Arid (Bikaner)
            "AWS_IND_H03",  # Hill (Shillong)
            "AWS_IND_P03",  # Plains (Patna)
        ]
    )


@dataclass
class ExtremeWeatherSettings:
    enabled: bool = True
    event_probability_per_station: float = 0.35
    min_network_events: int = 5
    event_types: List[str] = field(
        default_factory=lambda: [
            "heatwave",
            "convective_storm_squall",
            "temperature_inversion_fog",
        ]
    )


@dataclass
class AnomalySettings:
    enabled: bool = True
    target_anomaly_rate: float = 0.04
    station_local: bool = True
    min_episodes_per_temporal_split: int = 8
    min_episodes_spatial_holdout: int = 5
    fault_type_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "spike_or_drop": 0.16,
            "frozen_sensor": 0.14,
            "communication_dropout": 0.14,
            "calibration_drift": 0.14,
            "power_fluctuation_glitch": 0.14,
            "data_corruption": 0.14,
            "cross_sensor_inconsistency": 0.14,
        }
    )


@dataclass
class SplitSettings:
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15


@dataclass
class OutputSettings:
    output_dir: str = "data"
    save_parquet: bool = True
    save_csv: bool = True
    save_metadata_json: bool = True
    generate_plots: bool = True


@dataclass
class GeneratorConfig:
    generation: GenerationSettings = field(default_factory=GenerationSettings)
    stations: StationsSettings = field(default_factory=StationsSettings)
    extreme_weather: ExtremeWeatherSettings = field(default_factory=ExtremeWeatherSettings)
    anomalies: AnomalySettings = field(default_factory=AnomalySettings)
    splits: SplitSettings = field(default_factory=SplitSettings)
    output: OutputSettings = field(default_factory=OutputSettings)

    @classmethod
    def from_file(cls, config_path: str | Path) -> "GeneratorConfig":
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found at: {path}")

        with open(path, "r", encoding="utf-8") as f:
            if path.suffix.lower() in [".yaml", ".yml"]:
                data = yaml.safe_load(f) or {}
            elif path.suffix.lower() == ".json":
                data = json.load(f) or {}
            else:
                raise ValueError(f"Unsupported config format: {path.suffix}")

        generation = GenerationSettings(**data.get("generation", {}))
        stations = StationsSettings(**data.get("stations", {}))
        extreme_weather = ExtremeWeatherSettings(**data.get("extreme_weather", {}))
        anomalies = AnomalySettings(**data.get("anomalies", {}))
        
        split_data = data.get("splits", {})
        temporal_data = split_data.get("temporal", split_data)
        splits = SplitSettings(**temporal_data)
        
        output = OutputSettings(**data.get("output", {}))

        return cls(
            generation=generation,
            stations=stations,
            extreme_weather=extreme_weather,
            anomalies=anomalies,
            splits=splits,
            output=output,
        )
