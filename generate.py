"""CLI entry point for SkyGuard AI synthetic AWS sensor data generator.

Usage:
    python generate.py --config configs/default_config.yaml
    python generate.py --days 30 --anomaly-rate 0.04 --seed 42 --plot
"""

import argparse
from pathlib import Path
import sys

from src.config import GeneratorConfig
from src.pipeline import DatasetPipeline


def main():
    parser = argparse.ArgumentParser(
        description="SkyGuard AI - Physics-Informed Synthetic AWS Sensor Data Generator"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to YAML or JSON generator configuration file",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Override simulation duration in days",
    )
    parser.add_argument(
        "--sampling-min",
        type=int,
        default=None,
        help="Override sampling interval in minutes (default: 10)",
    )
    parser.add_argument(
        "--anomaly-rate",
        type=float,
        default=None,
        help="Override target anomaly rate (e.g. 0.035 for 3.5%%)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed for reproducibility",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory (default: data)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate visual sanity-check plots after data synthesis",
    )

    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        config = GeneratorConfig.from_file(config_path)
    else:
        print(f"Warning: Config file '{args.config}' not found. Using default configuration.")
        config = GeneratorConfig()

    # Apply CLI overrides
    if args.days is not None:
        config.generation.duration_days = args.days
    if args.sampling_min is not None:
        config.generation.sampling_interval_minutes = args.sampling_min
    if args.anomaly_rate is not None:
        config.anomalies.target_anomaly_rate = args.anomaly_rate
    if args.seed is not None:
        config.generation.random_seed = args.seed
    if args.output_dir is not None:
        config.output.output_dir = args.output_dir
    if args.plot:
        config.output.generate_plots = True

    print("================================================================================")
    print("                    SkyGuard AI - AWS Synthetic Data Generator                  ")
    print("================================================================================")
    print(f"Configuration:           {config_path}")
    print(f"Duration:                {config.generation.duration_days} days")
    print(f"Sampling Cadence:        {config.generation.sampling_interval_minutes} minutes")
    print(f"Target Anomaly Rate:     {config.anomalies.target_anomaly_rate * 100:.2f}%")
    print(f"Random Seed:             {config.generation.random_seed}")
    print(f"Output Directory:        {config.output.output_dir}")
    print("--------------------------------------------------------------------------------")

    pipeline = DatasetPipeline(config)
    master_df, metadata, events = pipeline.generate()

    saved_files = pipeline.split_and_save(master_df, metadata)

    print("\nGenerated Artifacts:")
    for key, path in saved_files.items():
        print(f"  - {key:25s}: {path}")

    # Optionally trigger visualizer
    if config.output.generate_plots:
        from visualize import generate_sanity_check_plots
        plots_dir = Path(config.output.output_dir) / "plots"
        generate_sanity_check_plots(master_df, events, plots_dir)

    print("\nGeneration process finished successfully!")


if __name__ == "__main__":
    main()
