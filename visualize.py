"""Visual sanity-check generator for SkyGuard AI synthetic AWS sensor data.

Generates:
1. climate_zones_comparison.png: 7-day multi-variable trace across the 4 Indian climate zones.
2. extreme_weather_cases.png: Authentic weather phenomena (squall, heatwave, fog) labeled is_anomaly=False.
3. sensor_fault_taxonomy.png: Zoomed-in panels demonstrating all 7 sensor fault types with ground-truth highlights.
"""

from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from src.anomaly_injector import AnomalyEvent


# Styling configuration
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.35
plt.rcParams["grid.linestyle"] = "--"


def plot_climate_zones_comparison(master_df: pd.DataFrame, output_path: Path):
    """Plot multi-variable time series across 4 representative climate zone stations."""
    target_stations = {
        "AWS_IND_C01": ("Mumbai (Colaba) - Coastal", "#0284c7"),
        "AWS_IND_A01": ("Jodhpur - Arid/Desert", "#ea580c"),
        "AWS_IND_H01": ("Shimla - Hill/Montane", "#16a34a"),
        "AWS_IND_P01": ("New Delhi - Gangetic Plains", "#9333ea"),
    }

    # Filter to first 5 days for clear diurnal inspection
    min_time = master_df["timestamp"].min()
    max_time = min_time + pd.Timedelta(days=5)
    sub_df = master_df[(master_df["timestamp"] >= min_time) & (master_df["timestamp"] <= max_time)]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("SkyGuard AI: Multi-Station Diurnal Signals Across Indian Climate Zones (5-Day Baseline)", fontsize=14, weight="bold")

    for st_id, (label, color) in target_stations.items():
        st_data = sub_df[sub_df["station_id"] == st_id].sort_values("timestamp")
        if len(st_data) == 0:
            continue
        axes[0].plot(st_data["timestamp"], st_data["temperature_c"], label=label, color=color, lw=1.6)
        axes[1].plot(st_data["timestamp"], st_data["pressure_hpa"], label=label, color=color, lw=1.6)
        axes[2].plot(st_data["timestamp"], st_data["humidity_pct"], label=label, color=color, lw=1.6)

    axes[0].set_ylabel("Air Temperature (°C)", weight="bold")
    axes[0].legend(loc="upper right", framealpha=0.9)
    axes[0].set_title("Diurnal Thermal Cycle (Solar Noon Lag & Climatological DTR)")

    axes[1].set_ylabel("Surface Pressure (hPa)", weight="bold")
    axes[1].set_title("Barometric Altitude Offset & Semi-Diurnal Solar Tides")

    axes[2].set_ylabel("Relative Humidity (%)", weight="bold")
    axes[2].set_title("Diurnal Humidity Dynamics (Inverse Correlation with Temperature)")
    axes[2].set_xlabel("Time (IST)", weight="bold")

    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))
    fig.autofmt_xdate()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_extreme_weather_cases(master_df: pd.DataFrame, output_path: Path):
    """Plot genuine extreme weather events to verify physical consistency and normal labeling."""
    # Find stations that experienced extreme weather (e.g. sharp pressure drop or heatwave)
    # Search for coastal station with convective squall or arid with heatwave
    normal_df = master_df[~master_df["is_anomaly"]].copy()

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=False)
    fig.suptitle("SkyGuard AI: Genuine Severe Weather Dynamics (Ground Truth: is_anomaly = FALSE)", fontsize=13, weight="bold")

    # Example 1: Convective storm / Monsoon squall in Coastal station
    coastal_df = normal_df[normal_df["station_id"] == "AWS_IND_C01"].sort_values("timestamp").reset_index(drop=True)
    if len(coastal_df) > 500:
        # Find minimum pressure period
        min_p_idx = coastal_df["pressure_hpa"].idxmin()
        window_start = max(0, min_p_idx - 144)
        window_end = min(len(coastal_df), min_p_idx + 144)
        storm_sample = coastal_df.iloc[window_start:window_end]

        ax0_t = axes[0].twinx()
        l1 = axes[0].plot(storm_sample["timestamp"], storm_sample["pressure_hpa"], color="#dc2626", lw=1.8, label="Pressure (hPa)")
        l2 = ax0_t.plot(storm_sample["timestamp"], storm_sample["temperature_c"], color="#0284c7", lw=1.5, label="Temperature (°C)")
        l3 = ax0_t.plot(storm_sample["timestamp"], storm_sample["humidity_pct"], color="#16a34a", lw=1.2, ls="--", label="Humidity (%)")
        
        axes[0].set_ylabel("Pressure (hPa)", color="#dc2626", weight="bold")
        ax0_t.set_ylabel("Temp (°C) / RH (%)", color="#0284c7", weight="bold")
        axes[0].set_title("Monsoon Depression / Squall (Rapid Pressure Drop + Rain Cooling Downdraft + 98% RH) [NORMAL]", fontsize=11)
        axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))
        lines = l1 + l2 + l3
        labels = [l.get_label() for l in lines]
        axes[0].legend(lines, labels, loc="upper right")

    # Example 2: Heatwave in Arid zone
    arid_df = normal_df[normal_df["station_id"] == "AWS_IND_A01"].sort_values("timestamp").reset_index(drop=True)
    if len(arid_df) > 1000:
        max_t_idx = arid_df["temperature_c"].idxmax()
        w_start = max(0, max_t_idx - 288)
        w_end = min(len(arid_df), max_t_idx + 288)
        heat_sample = arid_df.iloc[w_start:w_end]

        ax1_t = axes[1].twinx()
        l1 = axes[1].plot(heat_sample["timestamp"], heat_sample["temperature_c"], color="#b91c1c", lw=1.8, label="Temperature (°C)")
        l2 = ax1_t.plot(heat_sample["timestamp"], heat_sample["humidity_pct"], color="#0891b2", lw=1.5, label="Humidity (%)")
        axes[1].set_ylabel("Temperature (°C)", color="#b91c1c", weight="bold")
        ax1_t.set_ylabel("Humidity (%)", color="#0891b2", weight="bold")
        axes[1].set_title("Arid Heatwave Episode (Sustained High T + Severely Depressed RH) [NORMAL]", fontsize=11)
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))
        lines = l1 + l2
        axes[1].legend(lines, [l.get_label() for l in lines], loc="upper right")

    # Example 3: Fog / Inversion in Plains
    plains_df = normal_df[normal_df["station_id"] == "AWS_IND_P01"].sort_values("timestamp").reset_index(drop=True)
    if len(plains_df) > 500:
        sample_fog = plains_df.iloc[144:432]
        axes[2].plot(sample_fog["timestamp"], sample_fog["humidity_pct"], color="#059669", lw=1.8, label="Humidity (%)")
        ax2_t = axes[2].twinx()
        ax2_t.plot(sample_fog["timestamp"], sample_fog["temperature_c"], color="#475569", lw=1.5, label="Temperature (°C)")
        axes[2].set_ylabel("Humidity (%)", color="#059669", weight="bold")
        ax2_t.set_ylabel("Temperature (°C)", color="#475569", weight="bold")
        axes[2].set_title("Temperature Inversion / Radiation Fog (RH Saturated >98%, DTR Collapsed) [NORMAL]", fontsize=11)
        axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))
        axes[2].legend(loc="lower left")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_sensor_fault_taxonomy(master_df: pd.DataFrame, events: List[AnomalyEvent], output_path: Path):
    """Plot high-resolution panels illustrating every fault type in the dataset."""
    fault_types = [
        "spike_or_drop",
        "frozen_sensor",
        "communication_dropout",
        "calibration_drift",
        "power_fluctuation_glitch",
        "data_corruption",
        "cross_sensor_inconsistency",
    ]

    fig, axes = plt.subplots(len(fault_types), 1, figsize=(14, 18), sharex=False)
    fig.suptitle("SkyGuard AI: Ground-Truth Sensor Fault Taxonomy (PS 26073)", fontsize=14, weight="bold", y=0.995)

    for i, fault in enumerate(fault_types):
        ax = axes[i]
        # Find an event of this fault type
        matching_events = [e for e in events if e.anomaly_type == fault]
        if not matching_events:
            ax.text(0.5, 0.5, f"No event found for {fault}", ha="center", va="center")
            continue

        # Choose the first clear event
        event = matching_events[0]
        st_data = master_df[master_df["station_id"] == event.station_id].sort_values("timestamp").reset_index(drop=True)

        # Get window with padding
        pad = max(int(event.duration_steps * 1.5), 18)
        w_start = max(0, event.start_idx - pad)
        w_end = min(len(st_data), event.end_idx + pad)
        window_df = st_data.iloc[w_start:w_end]

        # Plot variables
        t_line, = ax.plot(window_df["timestamp"], window_df["temperature_c"], label="Temperature (°C)", color="#ef4444", lw=1.5)
        
        # Second axis for pressure
        ax_p = ax.twinx()
        p_line, = ax_p.plot(window_df["timestamp"], window_df["pressure_hpa"], label="Pressure (hPa)", color="#3b82f6", lw=1.5, ls="--")

        # Highlight anomaly span
        anom_mask = window_df["is_anomaly"]
        if np.any(anom_mask):
            anom_times = window_df[anom_mask]["timestamp"]
            ax.axvspan(anom_times.min(), anom_times.max(), color="#fee2e2", alpha=0.6, label="Anomaly Window")

        title_str = f"Fault Type: [{fault}] | Station: {event.station_id} | Severity: {event.severity:.2f} | {event.description}"
        ax.set_title(title_str, fontsize=10.5, weight="bold", loc="left", color="#991b1b")
        ax.set_ylabel("Temp (°C)", color="#ef4444", fontsize=9)
        ax_p.set_ylabel("Pressure (hPa)", color="#3b82f6", fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))

        if i == 0:
            lines = [t_line, p_line]
            ax.legend(lines, [l.get_label() for l in lines], loc="upper right", fontsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {output_path}")


def generate_sanity_check_plots(
    master_df: pd.DataFrame, events: List[AnomalyEvent], plots_dir: Path
):
    """Orchestrate saving all visual sanity checks."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    print("\nRendering visual sanity-check plots...")
    plot_climate_zones_comparison(master_df, plots_dir / "climate_zones_comparison.png")
    plot_extreme_weather_cases(master_df, plots_dir / "extreme_weather_cases.png")
    plot_sensor_fault_taxonomy(master_df, events, plots_dir / "sensor_fault_taxonomy.png")
    print(f"All sanity check plots successfully saved to: {plots_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/raw/aws_telemetry_master.parquet")
    parser.add_argument("--plots-dir", default="data/plots")
    args = parser.parse_args()

    df = pd.read_parquet(args.data_path)
    # Reconstruct basic events from labels
    anom_rows = df[df["is_anomaly"]].copy()
    dummy_events = []
    for (st_id, anom_t), group in anom_rows.groupby(["station_id", "anomaly_type"]):
        dummy_events.append(
            AnomalyEvent(
                station_id=st_id,
                anomaly_type=anom_t,
                start_idx=group.index[0],
                end_idx=group.index[-1] + 1,
                duration_steps=len(group),
                affected_variables=["all"],
                severity=float(group["anomaly_severity"].iloc[0] or 0.7),
                description=f"Recorded {anom_t}",
            )
        )
    generate_sanity_check_plots(df, dummy_events, Path(args.plots_dir))
