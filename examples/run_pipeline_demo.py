"""
SkyGuard AI (SIH 2026, Problem Statement 26073)
Production Pipeline Demonstration Script: examples/run_pipeline_demo.py

Demonstrates:
1. One-time initialization of the unified 5-tier SkyGuardPipeline.
2. Ingestion of live telemetry rows covering:
   - Normal Quiet-Period Baseline
   - Tier 1 Deterministic Hardware Faults: Data Corruption (-999.0) & Communication Dropout (Nulls)
   - Tier 2 Temporal ML Faults: Spike/Drop, Frozen Sensor, Power Glitch
   - Tier 3 Multivariate/Spatial Faults: Calibration Drift, Cross-Sensor Inconsistency
   - Genuine Extreme Weather: Regional Convective Storm Squall (AWS_IND_H01)
3. Formatted display of Two-Track routing, TreeSHAP plain-language rationales,
   Sensor Health Index (SHI), and 3-NN spatial buddy median value corrections.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.skyguard_pipeline import SkyGuardPipeline


def main():
    print("=" * 80)
    print("   SKYGUARD AI — PS 26073 END-TO-END TELEMETRY QC & ANOMALY PIPELINE DEMO   ")
    print("=" * 80)

    # 1. Initialize Pipeline (Loads all 5 tiers exactly once)
    print("\n[Step 1/3] Initializing SkyGuardPipeline from persisted model artifacts...")
    pipeline = SkyGuardPipeline(models_dir=PROJECT_ROOT / "models")
    print("  -> Models loaded: Tier 1 QC, Scaler, Isolation Forest, GRU-AE, Mahalanobis, LightGBM, TreeSHAP.")

    # 2. Load Telemetry Data for Demonstration
    print("\n[Step 2/3] Loading representative telemetry slices from test and train...")
    df_test = pd.read_parquet(PROJECT_ROOT / "data/tier4_results/test.parquet")
    df_train = pd.read_parquet(PROJECT_ROOT / "data/tier4_results/train.parquet")

    # Select representative events
    demo_cases = [
        {
            "category": "1. Normal Baseline Telemetry",
            "station": "AWS_IND_C02",
            "timestamp": "2026-07-22 01:20:00",
            "row": df_test[df_test["anomaly_type"].isna()].iloc[100].to_dict(),
            "expected_behavior": "Declared Normal (Track 1 Operational Pass, SHI remains 100%)",
        },
        {
            "category": "2. Tier 1 Hardware Corruption (Sentinel Value)",
            "station": "AWS_IND_H01",
            "timestamp": "2026-07-22 09:00:00",
            "row": df_test[df_test["anomaly_type"] == "data_corruption"].iloc[0].to_dict(),
            "expected_behavior": "Caught deterministically at Tier 1 (Track 1 Operational Alert)",
        },
        {
            "category": "3. Tier 1 Communication Dropout (Null Telemetry)",
            "station": "AWS_IND_P04",
            "timestamp": "2026-07-23 04:00:00",
            "row": df_test[df_test["anomaly_type"] == "communication_dropout"].iloc[0].to_dict(),
            "expected_behavior": "Caught deterministically at Tier 1 (Track 1 Operational Alert)",
        },
        {
            "category": "4. Tier 2 Temporal Fault: Sudden Thermal Spike",
            "station": "AWS_IND_A02",
            "timestamp": "2026-07-22 05:00:00",
            "row": df_test[df_test["anomaly_type"] == "spike_or_drop"].iloc[0].to_dict(),
            "expected_behavior": "Caught by Temporal/Multivariate models (Track 1 Operational Alert)",
        },
        {
            "category": "5. Tier 2 Temporal Fault: Frozen Sensor Flatline",
            "station": "AWS_IND_H01",
            "timestamp": "2026-07-23 02:00:00",
            "row": df_test[df_test["anomaly_type"] == "frozen_sensor"].iloc[50].to_dict(),
            "expected_behavior": "Routed to Track 2 (Secondary Maintenance Queue)",
        },
        {
            "category": "6. Tier 2 Temporal Fault: Power Glitch Jitter",
            "station": "AWS_IND_H04",
            "timestamp": "2026-07-26 02:20:00",
            "row": df_test[df_test["anomaly_type"] == "power_fluctuation_glitch"].iloc[10].to_dict(),
            "expected_behavior": "Caught by Isolation Forest / GRU (Track 1 or Track 2 Queue)",
        },
        {
            "category": "7. Tier 3 Multivariate Fault: Calibration Drift",
            "station": "AWS_IND_C03",
            "timestamp": "2026-07-24 02:10:00",
            "row": df_test[df_test["anomaly_type"] == "calibration_drift"].iloc[100].to_dict(),
            "expected_behavior": "Routed to Track 2 (Maintenance Queue, Mahalanobis attribution)",
        },
        {
            "category": "8. Tier 3 Multivariate Fault: Cross-Sensor Inconsistency",
            "station": "AWS_IND_H01",
            "timestamp": "2026-07-27 11:10:00",
            "row": df_test[df_test["anomaly_type"] == "cross_sensor_inconsistency"].iloc[50].to_dict(),
            "expected_behavior": "Routed to Track 2 (Maintenance Queue, Thermodynamic violation)",
        },
        {
            "category": "9. Genuine Extreme Weather: Convective Storm Squall",
            "station": "AWS_IND_H01",
            "timestamp": "2026-07-10 20:00:00",
            "row": df_train[(df_train["station_id"] == "AWS_IND_H01") & (df_train["timestamp"] == "2026-07-10 20:00:00")].iloc[0].to_dict(),
            "expected_behavior": "Suppressed by Spatial Consensus Gate (0 false alarm, regional agreement)",
        },
    ]

    # Pre-populate history for stations so higher tiers are fully active
    print("  -> Pre-warming rolling station history buffers from test/train...")
    for case in demo_cases:
        st = case["station"]
        st_history = df_test[df_test["station_id"] == st].head(30).to_dict(orient="records")
        for r in st_history:
            pipeline.process_row(r)

    # 3. Execute Demo Slices
    print("\n[Step 3/3] Ingesting Live Telemetry Rows through SkyGuard Pipeline...")
    print("-" * 80)

    for case in demo_cases:
        row_data = case["row"]
        concurrent_nbrs = None
        if "Extreme Weather" in case["category"]:
            ts_str = case["timestamp"]
            st_id = case["station"]
            nbr_ids = pipeline.spatial_neighbors.get(st_id, [])
            nbr_df = df_train[(df_train["station_id"].isin(nbr_ids)) & (df_train["timestamp"] == ts_str)]
            if len(nbr_df) > 0:
                concurrent_nbrs = {}
                for _, nr in nbr_df.iterrows():
                    concurrent_nbrs[nr["station_id"]] = {
                        "t": nr["temperature_c"],
                        "p": nr["pressure_hpa"],
                        "rh": nr["humidity_pct"],
                        "timestamp": pd.to_datetime(ts_str),
                    }

        res = pipeline.process_row(row_data, concurrent_neighbors=concurrent_nbrs)

        print(f"\n>>> CASE: {case['category']}")
        print(f"    Expected:   {case['expected_behavior']}")
        print(f"    Station:    {res['station_id']} | Timestamp: {res['timestamp']}")
        print(f"    Raw Input:  Temp={row_data.get('temperature_c')} °C, Press={row_data.get('pressure_hpa')} hPa, RH={row_data.get('humidity_pct')} %")
        print(f"    Decision:   is_anomaly={res['is_anomaly']} | Type={res['predicted_type']} (Confidence: {res['confidence']:.2f})")
        print(f"    Routing:    which_track={res['which_track']} (Track 1 = Operational Alert, Track 2 = Maintenance Queue)")
        print(f"    Rationale:  {res['shap_rationale_text']}")
        print(f"    Health SHI: {res['sensor_health_index']:.1f} / 100.0")

        if res["corrected_value_suggestion"]:
            corr = res["corrected_value_suggestion"]
            print(f"    Correction: 3-NN Buddy Median Suggestion: Temp={corr['temperature_c']} °C, Press={corr['pressure_hpa']} hPa, RH={corr['humidity_pct']} %")
        else:
            print("    Correction: None (Reading within valid bounds or unflagged)")

        diag = res["diagnostics"]
        print(f"    Diagnostics: Latency = {diag['latency_ms']:.2f} ms | History Length = {diag['history_length']} | Neighbor Fallback = {diag['neighbor_fallback_used']}")

    print("\n" + "=" * 80)
    print("DEMO COMPLETE: All 9 representative meteorological scenarios successfully processed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
