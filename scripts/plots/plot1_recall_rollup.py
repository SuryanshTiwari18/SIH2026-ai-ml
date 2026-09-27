"""
SkyGuard AI (SIH 2026, Problem Statement 26073)
Deliverable 1: Two-Track Recall Rollup (Grouped Bar Chart)

Outputs: outputs/plots/recall_rollup.png
Recomputed against: data/tier4_results/test.parquet and data/tier4_results/spatial_holdout.parquet
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TEST_PARQUET = PROJECT_ROOT / "data" / "tier4_results" / "test.parquet"
HOLDOUT_PARQUET = PROJECT_ROOT / "data" / "tier4_results" / "spatial_holdout.parquet"
OUTPUT_PNG = PROJECT_ROOT / "outputs" / "plots" / "recall_rollup.png"

# Style configuration
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8

CATEGORIES = [
    "data_corruption",
    "communication_dropout",
    "spike_or_drop",
    "power_fluctuation_glitch",
    "frozen_sensor",
    "calibration_drift",
    "cross_sensor_inconsistency",
]

CATEGORY_LABELS = [
    "Data\nCorruption",
    "Communication\nDropout",
    "Spike / Drop\nTransient",
    "Power Glitch\nJitter",
    "Frozen Sensor\nFlatline",
    "Calibration\nDrift",
    "Cross-Sensor\nInconsistency",
]

# Documented benchmarks in docs/FINAL_EVALUATION.md for cross-verification
DOCUMENTED_TEST_RECALLS = {
    "data_corruption": 100.00,
    "communication_dropout": 100.00,
    "spike_or_drop": 76.47,
    "power_fluctuation_glitch": 59.65,
    "frozen_sensor": 41.55,
    "calibration_drift": 80.96,
    "cross_sensor_inconsistency": 97.95,
}

DOCUMENTED_HOLDOUT_RECALLS = {
    "data_corruption": 100.00,
    "communication_dropout": 100.00,
    "spike_or_drop": 100.00,
    "power_fluctuation_glitch": 53.33,
    "frozen_sensor": 40.00,
    "calibration_drift": 51.97,
    "cross_sensor_inconsistency": 88.00,
}


def compute_recalls(df: pd.DataFrame) -> dict:
    """Computes recall for each fault category using production two-track decision logic."""
    recalls = {}
    for cat in CATEGORIES:
        sub = df[df["anomaly_type"] == cat]
        total = len(sub)
        if total == 0:
            recalls[cat] = 0.0
            continue
        # In Two-Track architecture:
        # Track 1 physical faults caught at Tier 1 (hard_rule_flagged and fusion_flagged)
        # Track 2 faults caught by Learned Meta-Classifier (fusion_flagged)
        # Combined evaluation matches fusion_flagged (which routes to Track 1 or Track 2)
        flagged = int((sub["fusion_flagged"] == True).sum())
        rec = (flagged / total) * 100.0
        recalls[cat] = {
            "total": total,
            "flagged": flagged,
            "recall": rec,
        }
    return recalls


def main():
    print(f"Loading test split from {TEST_PARQUET}...")
    df_test = pd.read_parquet(TEST_PARQUET)
    print(f"Loading spatial holdout split from {HOLDOUT_PARQUET}...")
    df_holdout = pd.read_parquet(HOLDOUT_PARQUET)

    test_metrics = compute_recalls(df_test)
    holdout_metrics = compute_recalls(df_holdout)

    print("\n--- Cross-Verification with FINAL_EVALUATION.md ---")
    test_recalls = []
    holdout_recalls = []
    for cat in CATEGORIES:
        t_rec = test_metrics[cat]["recall"]
        h_rec = holdout_metrics[cat]["recall"]
        t_doc = DOCUMENTED_TEST_RECALLS[cat]
        h_doc = DOCUMENTED_HOLDOUT_RECALLS[cat]

        test_diff = abs(t_rec - t_doc)
        holdout_diff = abs(h_rec - h_doc)
        print(
            f"{cat:28s} | Test: {t_rec:6.2f}% (doc {t_doc:6.2f}%, diff={test_diff:.4f}%) | "
            f"Holdout: {h_rec:6.2f}% (doc {h_doc:6.2f}%, diff={holdout_diff:.4f}%)"
        )
        assert test_diff < 0.05, f"Discrepancy in test recall for {cat}: {t_rec} vs {t_doc}"
        assert holdout_diff < 0.05, f"Discrepancy in holdout recall for {cat}: {h_rec} vs {h_doc}"

        test_recalls.append(t_rec)
        holdout_recalls.append(h_rec)

    # Plot generation
    fig, ax = plt.subplots(figsize=(10.5, 5.5), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = np.arange(len(CATEGORIES))
    width = 0.36

    # Colors as mandated by deck guidelines:
    # test = blue #4F81BD, spatial_holdout = navy #1E487C
    color_test = "#4F81BD"
    color_holdout = "#1E487C"

    rects1 = ax.bar(x - width / 2, test_recalls, width, label="Test Split (12 Stations, Unseen Time)", color=color_test, edgecolor="none", zorder=3)
    rects2 = ax.bar(x + width / 2, holdout_recalls, width, label="Spatial Holdout (4 Unseen Stations)", color=color_holdout, edgecolor="none", zorder=3)

    # Annotate bars with exact percentages
    def autolabel(rects, is_holdout=False):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
                color="#1E487C" if is_holdout else "#2B547E",
            )

    autolabel(rects1, is_holdout=False)
    autolabel(rects2, is_holdout=True)

    ax.set_ylabel("Recall (%)", fontsize=13, fontweight="bold", labelpad=8)
    ax.set_title("SkyGuard AI: End-to-End Recall Rollup across Fault Categories", fontsize=16, fontweight="bold", pad=15, color="#1E487C")
    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORY_LABELS, fontsize=10.5, fontweight="medium")
    ax.set_ylim(0, 115)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.tick_params(axis="both", which="major", labelsize=11)

    # Subtle horizontal gridlines only to aid value reading
    ax.yaxis.grid(True, linestyle="--", alpha=0.35, color="#A6B9D0", zorder=0)
    ax.xaxis.grid(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")

    # Legend
    legend = ax.legend(frameon=True, facecolor="white", edgecolor="#D0D7DE", fontsize=11, loc="upper right")
    legend.get_frame().set_linewidth(0.8)

    plt.tight_layout()
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PNG, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"\nSuccessfully generated {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
