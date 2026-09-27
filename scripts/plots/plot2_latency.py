"""
SkyGuard AI (SIH 2026, Problem Statement 26073)
Deliverable 2: Latency Benchmark (Bar Chart + SLA Line)

Outputs: outputs/plots/latency_benchmark.png
Recomputed by running SkyGuardPipeline.benchmark_latency() against data/tier4_results/test.parquet
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TEST_PARQUET = PROJECT_ROOT / "data" / "tier4_results" / "test.parquet"
OUTPUT_PNG = PROJECT_ROOT / "outputs" / "plots" / "latency_benchmark.png"

# Import SkyGuardPipeline from src
import sys
sys.path.insert(0, str(PROJECT_ROOT))
from src.skyguard_pipeline import SkyGuardPipeline

# Documented benchmarks in docs/FINAL_EVALUATION.md for cross-verification
DOCUMENTED_LATENCY = {
    "mean_ms": 8.99,
    "p50_ms": 0.09,
    "p95_ms": 28.88,
    "p99_ms": 31.74,
    "sla_ms": 500.0,
    "sla_multiple": 15.7,  # 500.0 / 31.74
}

# Style configuration
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8


def main():
    print(f"Initializing SkyGuardPipeline...")
    pipeline = SkyGuardPipeline(models_dir=PROJECT_ROOT / "models")

    print(f"Loading test telemetry from {TEST_PARQUET}...")
    df_test = pd.read_parquet(TEST_PARQUET)

    print("Executing benchmark_latency() across 500 individual cold-telemetry rows...")
    metrics = pipeline.benchmark_latency(df_test, n_samples=500)

    p50 = metrics["p50_ms"]
    p95 = metrics["p95_ms"]
    p99 = metrics["p99_ms"]
    mean_val = metrics["mean_ms"]
    sla_multiple = round(500.0 / p99, 1)

    print("\n--- Latency Benchmark Results ---")
    print(f"Mean Latency: {mean_val:.2f} ms (Documented: {DOCUMENTED_LATENCY['mean_ms']} ms)")
    print(f"P50 (Median): {p50:.2f} ms (Documented: {DOCUMENTED_LATENCY['p50_ms']} ms)")
    print(f"P95 Latency:  {p95:.2f} ms (Documented: {DOCUMENTED_LATENCY['p95_ms']} ms)")
    print(f"P99 Latency:  {p99:.2f} ms (Documented: {DOCUMENTED_LATENCY['p99_ms']} ms)")
    print(f"Computed SLA Compliance: {sla_multiple}x under SLA target (Documented: {DOCUMENTED_LATENCY['sla_multiple']}x)")

    # Plot generation
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bars_x = ["Median (P50)", "95th Percentile (P95)", "99th Percentile (P99)"]
    bars_y = [p50, p95, p99]
    colors = ["#4BACC6", "#4F81BD", "#1E487C"]

    bar_width = 0.45
    rects = ax.bar(bars_x, bars_y, width=bar_width, color=colors, edgecolor="none", zorder=3)

    # Horizontal SLA Target Line at y=500 ms
    ax.axhline(y=500.0, color="#C00000", linestyle="--", linewidth=2.0, zorder=4)
    ax.text(
        0.5,
        508.0,
        "500 ms Operational SLA Target (IMD / AWS Real-Time Constraint)",
        color="#C00000",
        fontsize=11.5,
        fontweight="bold",
        ha="center",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#FDEDEC", edgecolor="#E6B0AA", linewidth=0.8),
    )

    # Shaded compliance envelope
    ax.axhspan(0, 500, color="#F4F8FA", alpha=0.6, zorder=1)

    # Annotate bars with ms values
    for rect, val in zip(rects, bars_y):
        height = rect.get_height()
        ax.annotate(
            f"{val:.2f} ms",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="#1E487C",
        )

    # Highlight P99 multiple under SLA
    p99_rect = rects[2]
    ax.annotate(
        f"{sla_multiple:.1f}× Under SLA\n(Measured: {p99:.2f} ms | Baseline: {DOCUMENTED_LATENCY['p99_ms']:.2f} ms)",
        xy=(p99_rect.get_x() + p99_rect.get_width() / 2, p99),
        xytext=(0, 48),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color="#1E487C",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#EAF2F8", edgecolor="#4F81BD", linewidth=1.2),
        arrowprops=dict(arrowstyle="->", color="#1E487C", lw=1.5, connectionstyle="arc3,rad=0.0"),
    )

    ax.set_ylabel("End-to-End Latency (ms)", fontsize=13, fontweight="bold", labelpad=8)
    ax.set_title("SkyGuard AI: End-to-End Latency Benchmark vs. 500ms Operational SLA", fontsize=16, fontweight="bold", pad=20, color="#1E487C")
    ax.set_ylim(0, 560)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=11)

    ax.yaxis.grid(True, linestyle="--", alpha=0.35, color="#A6B9D0", zorder=0)
    ax.xaxis.grid(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")

    # Add subtitle caption
    fig.text(
        0.5,
        0.02,
        f"Benchmark: 500 cold rows sampled from test.parquet | Mean Latency: {mean_val:.2f} ms | Throughput: {metrics['throughput_rows_per_sec']:.1f} rows/sec",
        ha="center",
        fontsize=10,
        color="#555555",
        style="italic",
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PNG, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"\nSuccessfully generated {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
