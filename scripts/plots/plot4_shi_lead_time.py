"""
SkyGuard AI (SIH 2026, Problem Statement 26073)
Deliverable 4: Sensor Health Index (SHI) Time Series with Predictive Lead Time

Outputs: outputs/plots/shi_lead_time.png
Recomputed against: data/tier5_results/test.parquet (AWS_IND_A02 calibration drift episode)
"""

from pathlib import Path
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TIER5_PARQUET = PROJECT_ROOT / "data" / "tier5_results" / "test.parquet"
OUTPUT_PNG = PROJECT_ROOT / "outputs" / "plots" / "shi_lead_time.png"

# Documented benchmarks in docs/TIER5_EVALUATION.md and docs/FINAL_EVALUATION.md
DOCUMENTED_DRIFT = {
    "station_id": "AWS_IND_A02",
    "split": "test",
    "drift_start": "2026-07-23 12:00:00",
    "drift_end": "2026-07-24 14:20:00",
    "pre_drift_mean_shi": 99.89,
    "first_fusion_flag": "2026-07-23 15:00:00",
    "fusion_delay_hours": 3.0,
    "first_tier3_flag": "2026-07-23 20:30:00",
    "tier3_delay_hours": 8.5,
    "predictive_lead_time_hours": 5.5,  # 8.5h - 3.0h
    "end_drift_shi": 90.39,
}

# Style configuration
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8


def main():
    print(f"Loading Tier 5 evaluation data from {TIER5_PARQUET}...")
    df_t5 = pd.read_parquet(TIER5_PARQUET)
    df_t5["timestamp"] = pd.to_datetime(df_t5["timestamp"])

    # Extract target station AWS_IND_A02
    target_station = "AWS_IND_A02"
    st_df = df_t5[df_t5["station_id"] == target_station].sort_values("timestamp").reset_index(drop=True)

    # Ground truth drift window
    drift_mask = st_df["anomaly_type"] == "calibration_drift"
    drift_df = st_df[drift_mask]

    t_drift_start = drift_df["timestamp"].iloc[0]
    t_drift_end = drift_df["timestamp"].iloc[-1]

    # First detection by Track 2 (fusion_flagged)
    fus_flagged = drift_df[drift_df["fusion_flagged"] == True]
    t_detected = fus_flagged["timestamp"].iloc[0]

    # First formal Tier 3 static flag (mahalanobis_flagged)
    t3_flagged = drift_df[drift_df["mahalanobis_flagged"] == True]
    t_formal_alert = t3_flagged["timestamp"].iloc[0]

    # Compute exact timings
    detection_delay_hrs = (t_detected - t_drift_start).total_seconds() / 3600.0
    formal_alert_delay_hrs = (t_formal_alert - t_drift_start).total_seconds() / 3600.0
    predictive_lead_time_hrs = (t_formal_alert - t_detected).total_seconds() / 3600.0

    shi_start = float(st_df[st_df["timestamp"] == t_drift_start]["sensor_health_index"].iloc[0])
    shi_end = float(st_df[st_df["timestamp"] == t_drift_end]["sensor_health_index"].iloc[0])
    shi_pre_mean = float(st_df[st_df["timestamp"] < t_drift_start]["sensor_health_index"].mean())

    print("\n--- Cross-Verification with TIER5_EVALUATION.md ---")
    print(f"Station:                    {target_station}")
    print(f"Drift Start:                {t_drift_start} (Doc: {DOCUMENTED_DRIFT['drift_start']})")
    print(f"Track-2 Detected:           {t_detected} (+{detection_delay_hrs:.1f}h | Doc: +{DOCUMENTED_DRIFT['fusion_delay_hours']}h)")
    print(f"Tier 3 Formal Alert:        {t_formal_alert} (+{formal_alert_delay_hrs:.1f}h | Doc: +{DOCUMENTED_DRIFT['tier3_delay_hours']}h)")
    print(f"Predictive Lead Time:       {predictive_lead_time_hrs:.1f} hours (Doc: {DOCUMENTED_DRIFT['predictive_lead_time_hours']} hours)")
    print(f"Pre-Drift Mean SHI:         {shi_pre_mean:.2f}% (Doc: {DOCUMENTED_DRIFT['pre_drift_mean_shi']}%)")
    print(f"End-of-Episode SHI:         {shi_end:.2f}% (Doc: {DOCUMENTED_DRIFT['end_drift_shi']}%)")

    assert abs(predictive_lead_time_hrs - DOCUMENTED_DRIFT["predictive_lead_time_hours"]) < 0.05
    assert abs(shi_end - DOCUMENTED_DRIFT["end_drift_shi"]) < 0.05

    # Filter plotting window: 24h baseline before drift to end of drift
    plot_start = t_drift_start - pd.Timedelta(hours=20)
    plot_end = t_drift_end + pd.Timedelta(hours=4)
    win_df = st_df[(st_df["timestamp"] >= plot_start) & (st_df["timestamp"] <= plot_end)].copy()

    # Plot generation
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Plot SHI trajectory
    ax.plot(
        win_df["timestamp"],
        win_df["sensor_health_index"],
        color="#1E487C",
        linewidth=2.4,
        label="Sensor Health Index (EMA SHI)",
        zorder=4,
    )

    # Mark "Drift begins" vertical dashed line
    ax.axvline(x=t_drift_start, color="#F79646", linestyle="--", linewidth=1.8, zorder=5)
    ax.text(
        t_drift_start - pd.Timedelta(minutes=50),
        91.0,
        "Drift begins\n(Ground Truth)\n12:00:00",
        color="#C0504D",
        fontsize=9.5,
        fontweight="bold",
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#FDEDEC", edgecolor="#F79646", linewidth=0.8),
    )

    # Mark "Detected" vertical dashed line (Track-2 detection)
    ax.axvline(x=t_detected, color="#4BACC6", linestyle="--", linewidth=1.8, zorder=5)
    ax.text(
        t_detected + pd.Timedelta(minutes=40),
        97.8,
        "Detected\n(Track 2 Flag)\n15:00:00",
        color="#1B4F72",
        fontsize=9.5,
        fontweight="bold",
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#EBF5FB", edgecolor="#4BACC6", linewidth=0.8),
    )

    # Mark formal Tier 3 static alert line
    ax.axvline(x=t_formal_alert, color="#C00000", linestyle=":", linewidth=1.8, zorder=5)
    ax.text(
        t_formal_alert + pd.Timedelta(minutes=40),
        93.5,
        "Formal Alert\n(Tier 3 Static Flag)\n20:30:00",
        color="#900C3F",
        fontsize=9.5,
        fontweight="bold",
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#FADBD8", edgecolor="#C00000", linewidth=0.8),
    )

    # Shade the predictive lead time region between Track 2 detection and Formal Tier 3 alert
    ax.axvspan(t_detected, t_formal_alert, color="#9BBB59", alpha=0.25, zorder=2)
    mid_lead = t_detected + (t_formal_alert - t_detected) / 2
    ax.text(
        mid_lead,
        95.0,
        f"+{predictive_lead_time_hrs:.1f} Hours\nPredictive Lead Time\nAhead of Static Alert",
        color="#274E13",
        fontsize=10.5,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#EAF2D3", edgecolor="#9BBB59", linewidth=1.0),
    )

    # Also highlight early detection window from drift start (3.0h)
    ax.axvspan(t_drift_start, t_detected, color="#F79646", alpha=0.15, zorder=2)

    # Horizontal guide lines for health levels
    ax.axhline(y=100.0, color="#A6B9D0", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axhline(y=90.0, color="#C0504D", linestyle=":", linewidth=0.8, alpha=0.6)

    ax.set_ylabel("Sensor Health Index (0 – 100)", fontsize=12.5, fontweight="bold", labelpad=8)
    ax.set_title(
        f"AWS_IND_A02 Calibration Drift: Continuous EMA Health Decay & +{predictive_lead_time_hrs:.1f}h Lead Time",
        fontsize=14.5,
        fontweight="bold",
        pad=16,
        color="#1E487C",
    )
    ax.set_ylim(88.0, 102.5)

    # Date formatting on X-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b\n%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=8))
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10.5)

    ax.yaxis.grid(True, linestyle="--", alpha=0.35, color="#A6B9D0", zorder=0)
    ax.xaxis.grid(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")

    # Informational subtitle
    fig.text(
        0.5,
        0.015,
        f"Episode: AWS_IND_A02 Jaisalmer Arid Drift (159 steps / 26.5h) | Pre-drift SHI: {shi_pre_mean:.2f}% | End-of-drift SHI: {shi_end:.2f}%",
        ha="center",
        fontsize=9.5,
        color="#555555",
        style="italic",
    )

    legend = ax.legend(frameon=True, facecolor="white", edgecolor="#D0D7DE", fontsize=10, loc="upper right")
    legend.get_frame().set_linewidth(0.8)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PNG, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"\nSuccessfully generated {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
