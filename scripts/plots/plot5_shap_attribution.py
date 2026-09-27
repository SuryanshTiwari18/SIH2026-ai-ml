"""
SkyGuard AI (SIH 2026, Problem Statement 26073)
Deliverable 5: TreeSHAP Feature Attribution (Horizontal Bar Chart)

Outputs: outputs/plots/shap_attribution.png
Recomputed by running the LightGBM booster with pred_contrib=True against
a representative Track 2 anomaly instance from data/tier4_results/test.parquet.
"""

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TEST_PARQUET = PROJECT_ROOT / "data" / "tier4_results" / "test.parquet"
MODEL_PATH = PROJECT_ROOT / "models" / "tier4_fusion_classifier.joblib"
OUTPUT_PNG = PROJECT_ROOT / "outputs" / "plots" / "shap_attribution.png"

# Style configuration
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8

# Documented benchmark in docs/TIER5_EVALUATION.md Section 2 (Row 29)
# cross_sensor_inconsistency | AWS_IND_H01 | 2026-07-27 13:50:00
DOCUMENTED_INSTANCE = {
    "station_id": "AWS_IND_H01",
    "timestamp": "2026-07-27 13:50:00",
    "ground_truth_type": "cross_sensor_inconsistency",
    "expected_top_shap": "mahalanobis_dist",
    "documented_contribs": {
        "mahalanobis_dist": 4.14,
        "gru_score": -0.85,
        "buddy_flagged": -0.35,
    },
}

FEATURE_DISPLAY_NAMES = {
    "mahalanobis_dist": "Mahalanobis Covariance Distance (Tier 3)",
    "gru_score": "GRU-Autoencoder Reconstruction Error (Tier 2)",
    "buddy_flagged": "Spatial Peer Consensus Flag (Tier 3)",
    "if_score": "Isolation Forest Outlier Score (Tier 2)",
    "isolated_deviation": "Isolated Spatial Deviation Gate (Tier 3)",
    "tier1_flagged": "Physical QC Violation Flag (Tier 1)",
}


def main():
    print(f"Loading trained LightGBM meta-classifier from {MODEL_PATH}...")
    t4_dict = joblib.load(MODEL_PATH)
    clf = t4_dict["model"]
    classes = t4_dict["classes"]
    features = t4_dict["features"]

    print(f"Loading test split from {TEST_PARQUET}...")
    df_test = pd.read_parquet(TEST_PARQUET)

    # Locate canonical sample instance
    st_id = DOCUMENTED_INSTANCE["station_id"]
    ts_str = DOCUMENTED_INSTANCE["timestamp"]
    sample_df = df_test[(df_test["station_id"] == st_id) & (df_test["timestamp"] == ts_str)]

    assert len(sample_df) == 1, f"Expected 1 matching row for {st_id} at {ts_str}, found {len(sample_df)}"

    row = sample_df.iloc[0]
    X = sample_df[features].copy()
    for col in ["tier1_flagged", "buddy_flagged", "isolated_deviation"]:
        X[col] = X[col].astype(float)

    # Compute exact TreeSHAP feature contributions
    contribs = clf.booster_.predict(X, pred_contrib=True).reshape(len(classes), len(features) + 1)

    # Attribution for target fault class
    target_class = DOCUMENTED_INSTANCE["ground_truth_type"]
    class_idx = classes.index(target_class)
    feat_contribs = contribs[class_idx, :-1]

    # Map features to contributions
    shap_dict = dict(zip(features, feat_contribs))

    print(f"\n--- TreeSHAP Contributions for {target_class} at {st_id} ({ts_str}) ---")
    for f, val in shap_dict.items():
        doc_val = DOCUMENTED_INSTANCE["documented_contribs"].get(f)
        doc_str = f"(Documented: {doc_val:+.2f})" if doc_val is not None else ""
        print(f"  {f:22s}: {val:+.4f} {doc_str}")

    # Verify against documented values in TIER5_EVALUATION.md
    for f, doc_val in DOCUMENTED_INSTANCE["documented_contribs"].items():
        assert abs(shap_dict[f] - doc_val) < 0.05, f"Discrepancy for {f}: {shap_dict[f]} vs {doc_val}"

    # Sort features by absolute SHAP value (ascending for bottom-to-top horizontal bar chart)
    sorted_feats = sorted(features, key=lambda f: abs(shap_dict[f]), reverse=False)
    sorted_names = [FEATURE_DISPLAY_NAMES[f] for f in sorted_feats]
    sorted_vals = [shap_dict[f] for f in sorted_feats]

    # Colors: positive = orange #F79646 (pushes toward anomaly), negative = teal #4BACC6 (pushes toward normal)
    bar_colors = ["#F79646" if v >= 0 else "#4BACC6" for v in sorted_vals]

    # Plot generation
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y_pos = np.arange(len(sorted_feats))
    rects = ax.barh(y_pos, sorted_vals, height=0.55, color=bar_colors, edgecolor="none", zorder=3)

    # Vertical reference line at SHAP = 0
    ax.axvline(x=0.0, color="#666666", linestyle="-", linewidth=1.0, zorder=4)

    # Annotate bars with signed values
    for rect, val in zip(rects, sorted_vals):
        width = rect.get_width()
        ha = "left" if width >= 0 else "right"
        offset = 0.08 if width >= 0 else -0.08
        ax.annotate(
            f"{val:+.2f}",
            xy=(width + offset, rect.get_y() + rect.get_height() / 2),
            va="center",
            ha=ha,
            fontsize=11,
            fontweight="bold",
            color="#C0504D" if width >= 0 else "#1B4F72",
        )

    # Labels and title
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names, fontsize=10.5, fontweight="medium")
    ax.set_xlabel("TreeSHAP Value (Log-Odds Impact on Anomaly Attribution)", fontsize=11.5, fontweight="bold", labelpad=8)

    title_text = (
        f"TreeSHAP Attribution: {st_id} ({ts_str})\n"
        f"Predicted Fault: Cross-Sensor Inconsistency (Track 2 Maintenance Queue)"
    )
    ax.set_title(title_text, fontsize=13.5, fontweight="bold", pad=16, color="#1E487C")

    ax.set_xlim(-1.6, 5.0)
    ax.tick_params(axis="x", labelsize=10.5)

    ax.xaxis.grid(True, linestyle="--", alpha=0.35, color="#A6B9D0", zorder=0)
    ax.yaxis.grid(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")

    # Legend proxy handles
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#F79646", edgecolor="none", label="Positive: Pushes toward anomaly attribution"),
        Patch(facecolor="#4BACC6", edgecolor="none", label="Negative: Pushes toward normal / non-fault"),
    ]
    legend = ax.legend(handles=legend_elements, loc="lower right", frameon=True, facecolor="white", edgecolor="#D0D7DE", fontsize=9.5)
    legend.get_frame().set_linewidth(0.8)

    plt.tight_layout()
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PNG, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"\nSuccessfully generated {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
