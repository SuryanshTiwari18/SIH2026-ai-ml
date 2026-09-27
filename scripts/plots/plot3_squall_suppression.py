"""
SkyGuard AI (SIH 2026, Problem Statement 26073)
Deliverable 3: H01 Squall Suppression (Before/After Bar Pair)

Outputs: outputs/plots/squall_suppression.png
Recomputed against: data/tier4_results/train.parquet (AWS_IND_H01 convective squall episode)
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_PARQUET = PROJECT_ROOT / "data" / "tier4_results" / "train.parquet"
OUTPUT_PNG = PROJECT_ROOT / "outputs" / "plots" / "squall_suppression.png"

# Documented benchmarks in docs/FINAL_EVALUATION.md
DOCUMENTED_H01 = {
    "total_steps": 77,
    "standalone_gru_flagged": 38,
    "standalone_gru_fpr": 49.35,  # 38 / 77
    "gated_pipeline_flagged": 3,
    "gated_pipeline_fpr": 3.90,   # 3 / 77
    "suppression_rate": 92.11,    # (38 - 3) / 38
}

# Style configuration
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8


def main():
    print(f"Loading train split from {TRAIN_PARQUET}...")
    df_train = pd.read_parquet(TRAIN_PARQUET)
    df_train["timestamp"] = pd.to_datetime(df_train["timestamp"])

    # Locate canonical AWS_IND_H01 convective squall episode (start_idx = 5743, duration = 77)
    t_start = pd.Timestamp("2026-06-01 00:00:00") + pd.Timedelta(minutes=5743 * 10)
    t_end = t_start + pd.Timedelta(minutes=(77 - 1) * 10)

    h01_squall = df_train[
        (df_train["station_id"] == "AWS_IND_H01")
        & (df_train["timestamp"] >= t_start)
        & (df_train["timestamp"] <= t_end)
    ].sort_values("timestamp")

    total_steps = len(h01_squall)
    assert total_steps == 77, f"Expected 77 steps, got {total_steps}"

    # Standalone Tier 2 GRU-AE alarms
    n_gru_alarms = int((h01_squall["gru_flagged"] == True).sum())
    gru_fpr = (n_gru_alarms / total_steps) * 100.0

    # Final Two-Track / Gated Hard Rule alarms (Track 1 operational alerts)
    n_final_alarms = int((h01_squall["hard_rule_flagged"] == True).sum())
    final_fpr = (n_final_alarms / total_steps) * 100.0

    # Suppression rate
    n_suppressed = n_gru_alarms - n_final_alarms
    suppression_pct = (n_suppressed / n_gru_alarms) * 100.0
    relative_reduction = ((gru_fpr - final_fpr) / gru_fpr) * 100.0

    print("\n--- Cross-Verification with FINAL_EVALUATION.md ---")
    print(f"Total Squall Steps: {total_steps} (Doc: {DOCUMENTED_H01['total_steps']})")
    print(f"Standalone GRU-AE: {n_gru_alarms}/{total_steps} ({gru_fpr:.2f}% | Doc: {DOCUMENTED_H01['standalone_gru_fpr']}%)")
    print(f"Gated Pipeline:    {n_final_alarms}/{total_steps} ({final_fpr:.2f}% | Doc: {DOCUMENTED_H01['gated_pipeline_fpr']}%)")
    print(f"Suppression Rate:  {n_suppressed}/{n_gru_alarms} ({suppression_pct:.2f}% | Doc: {DOCUMENTED_H01['suppression_rate']}%)")

    assert abs(gru_fpr - DOCUMENTED_H01["standalone_gru_fpr"]) < 0.05
    assert abs(final_fpr - DOCUMENTED_H01["gated_pipeline_fpr"]) < 0.05
    assert abs(suppression_pct - DOCUMENTED_H01["suppression_rate"]) < 0.05

    # Plot generation
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bars_x = [
        "Standalone Tier 2 (GRU-AE)\nSingle-Station Temporal Model",
        "Two-Track Architecture\nWith Spatial-Consensus Gate",
    ]
    bars_y = [gru_fpr, final_fpr]
    # Colors: orange #F79646 (before/problem), green #9BBB59 (after/resolved)
    colors = ["#F79646", "#9BBB59"]

    bar_width = 0.42
    rects = ax.bar(bars_x, bars_y, width=bar_width, color=colors, edgecolor="none", zorder=3)

    # Annotate bars
    counts = [f"{n_gru_alarms}/{total_steps} flagged", f"{n_final_alarms}/{total_steps} flagged"]
    for i, (rect, val, cnt) in enumerate(zip(rects, bars_y, counts)):
        height = rect.get_height()
        ax.annotate(
            f"{val:.2f}%\n({cnt})",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="#C0504D" if i == 0 else "#4F6228",
        )

    # Callout banner showing 92.1% false alarm reduction
    ax.annotate(
        f"−{suppression_pct:.1f}% False Alarm Suppression\n(35 of 38 false alarms cleared via 3-NN consensus)",
        xy=(1.0, final_fpr + 2.0),
        xytext=(0.5, 34.0),
        ha="center",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        color="#1E487C",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#EBF5FB", edgecolor="#4BACC6", linewidth=1.5),
        arrowprops=dict(arrowstyle="->", color="#1E487C", lw=1.8, connectionstyle="arc3,rad=-0.15"),
    )

    ax.set_ylabel("False Positive Rate (%)", fontsize=13, fontweight="bold", labelpad=8)
    ax.set_title("AWS_IND_H01 Convective Squall: False Alarm Suppression", fontsize=15.5, fontweight="bold", pad=16, color="#1E487C")
    ax.set_ylim(0, 60)
    ax.set_yticks(np.arange(0, 61, 10))
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)

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
        f"Episode: AWS_IND_H01 Shimla (77 steps / 12.8 hrs, 2026-07-10 21:10 to 07-11 09:50) | Ground Truth: Severe Weather (Normal Sensor)",
        ha="center",
        fontsize=9.5,
        color="#555555",
        style="italic",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PNG, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"\nSuccessfully generated {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
