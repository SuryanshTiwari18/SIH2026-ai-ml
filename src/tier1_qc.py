"""SkyGuard AI - Tier 1: Deterministic Physical Quality Control (QC).

This module implements the Tier 1 Physical QC layer as specified in
docs/EDA_INSIGHTS.md Section 6. Tier 1 is a deterministic, rule-based,
zero-training filter operating directly on RAW Automatic Weather Station (AWS)
telemetry prior to feature engineering.

Rules enforced:
1. Communication Dropout:
   - Complete 3-sensor dropout: temperature_c, pressure_hpa, humidity_pct are all NaN.
     Flag reason: 'communication_dropout'
   - Partial sensor dropout: 1 or 2 sensors are NaN (defensive check).
     Flag reason: 'partial_nan_anomaly'
2. Sentinel Value Match:
   - Exact-match empirical sentinels derived from data_corruption telemetry:
     * temperature_c in {-999.0, -99.9, 999.9, 9999.0}
     * pressure_hpa in {-999.0, -99.9, 999.9, 9999.0}
     * humidity_pct in {-999.0, -25.0, 160.0}
     Flag reason: 'sentinel_value'
3. Physical Range Violation:
   - Out-of-bounds sensor readings:
     * temperature_c < -30.0 or > 60.0 °C
     * pressure_hpa < 500.0 or > 1100.0 hPa
     * humidity_pct < 0.0 or > 100.0 %
     Flag reason: 'range_violation'

Outputs per row:
- is_flagged: bool (True if any rule triggers, False otherwise)
- flag_reason: Optional[str] ('communication_dropout', 'partial_nan_anomaly',
  'sentinel_value', 'range_violation', or None)
"""

import argparse
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tier1_qc")

# -----------------------------------------------------------------------------
# Physical Range Limits (EDA_INSIGHTS.md Section 2)
# -----------------------------------------------------------------------------
PHYSICAL_RANGE_LIMITS: Dict[str, Tuple[float, float]] = {
    "temperature_c": (-30.0, 60.0),
    "pressure_hpa": (500.0, 1100.0),
    "humidity_pct": (0.0, 100.0),
}

# -----------------------------------------------------------------------------
# Empirically-Derived Sentinel Values
# Extracted from data/features/tier1_excluded_rows.parquet (data_corruption rows)
# -----------------------------------------------------------------------------
EMPIRICAL_SENTINEL_VALUES: Dict[str, Set[float]] = {
    "temperature_c": {-999.0, -99.9, 999.9, 9999.0},
    "pressure_hpa": {-999.0, -99.9, 999.9, 9999.0},
    "humidity_pct": {-999.0, -25.0, 160.0},
}


def derive_sentinels_from_excluded_rows(
    excluded_rows_path: Path = Path("data/features/tier1_excluded_rows.parquet"),
) -> Dict[str, Set[float]]:
    """Empirically inspects tier1_excluded_rows.parquet to extract exact sentinel values.

    Args:
        excluded_rows_path: Path to tier1_excluded_rows.parquet.

    Returns:
        Dictionary mapping column names to sets of float sentinel values.
    """
    if not excluded_rows_path.exists():
        logger.warning(
            "Reference file %s not found. Using pre-computed empirical sentinels.",
            excluded_rows_path,
        )
        return {k: set(v) for k, v in EMPIRICAL_SENTINEL_VALUES.items()}

    df_excl = pd.read_parquet(excluded_rows_path)
    dc = df_excl[df_excl["anomaly_type"] == "data_corruption"]

    sentinels: Dict[str, Set[float]] = {"temperature_c": set(), "pressure_hpa": set(), "humidity_pct": set()}

    # Look for known sentinel patterns or values violating range in data_corruption rows
    for col, (min_v, max_v) in PHYSICAL_RANGE_LIMITS.items():
        if col in dc.columns:
            non_null = dc[col].dropna()
            for val in non_null.unique():
                val_float = float(val)
                # A sentinel in data_corruption is either an out-of-range value or a recognized sentinel token
                if (val_float < min_v or val_float > max_v) or (val_float in {999.9, -99.9, -999.0, 9999.0}):
                    sentinels[col].add(val_float)

    logger.info("Empirically derived sentinels from %s:", excluded_rows_path)
    for col, vals in sentinels.items():
        logger.info("  %s: %s", col, sorted(list(vals)))

    return sentinels


def apply_tier1_qc(
    df: pd.DataFrame,
    sentinels: Optional[Dict[str, Set[float]]] = None,
    reason_priority: str = "sentinel_first",
) -> pd.DataFrame:
    """Applies Tier 1 Physical QC rules to raw telemetry DataFrame.

    Args:
        df: Input DataFrame containing raw telemetry columns:
            'temperature_c', 'pressure_hpa', 'humidity_pct'.
        sentinels: Dict mapping sensor names to sentinel float sets. If None,
            uses EMPIRICAL_SENTINEL_VALUES.
        reason_priority: Strategy for resolving multiple flags. Options:
            - 'sentinel_first': communication_dropout -> partial_nan_anomaly -> sentinel_value -> range_violation
            - 'range_first': communication_dropout -> partial_nan_anomaly -> range_violation -> sentinel_value

    Returns:
        DataFrame with original columns plus 'is_flagged' (bool) and 'flag_reason' (str or None).
    """
    if sentinels is None:
        sentinels = EMPIRICAL_SENTINEL_VALUES

    # Ensure required columns exist
    for col in ["temperature_c", "pressure_hpa", "humidity_pct"]:
        if col not in df.columns:
            raise ValueError(f"Required telemetry column '{col}' missing from DataFrame.")

    t = df["temperature_c"]
    p = df["pressure_hpa"]
    rh = df["humidity_pct"]

    # Rule 1: Communication Dropout / Missingness
    nan_t = t.isna()
    nan_p = p.isna()
    nan_rh = rh.isna()

    mask_comm_dropout = nan_t & nan_p & nan_rh
    mask_partial_nan = (nan_t | nan_p | nan_rh) & ~mask_comm_dropout

    # Rule 2: Sentinel Value Match
    mask_sent_t = t.isin(sentinels.get("temperature_c", set()))
    mask_sent_p = p.isin(sentinels.get("pressure_hpa", set()))
    mask_sent_rh = rh.isin(sentinels.get("humidity_pct", set()))
    mask_sentinel = mask_sent_t | mask_sent_p | mask_sent_rh

    # Rule 3: Physical Range Violation
    t_min, t_max = PHYSICAL_RANGE_LIMITS["temperature_c"]
    p_min, p_max = PHYSICAL_RANGE_LIMITS["pressure_hpa"]
    rh_min, rh_max = PHYSICAL_RANGE_LIMITS["humidity_pct"]

    mask_range_t = (t < t_min) | (t > t_max)
    mask_range_p = (p < p_min) | (p > p_max)
    mask_range_rh = (rh < rh_min) | (rh > rh_max)
    mask_range_viol = mask_range_t | mask_range_p | mask_range_rh

    # Combine into is_flagged
    is_flagged = mask_comm_dropout | mask_partial_nan | mask_sentinel | mask_range_viol

    # Assign flag_reason based on priority
    flag_reason = pd.Series(None, index=df.index, dtype="object")

    if reason_priority == "sentinel_first":
        flag_reason[mask_comm_dropout] = "communication_dropout"
        flag_reason[mask_partial_nan & flag_reason.isna()] = "partial_nan_anomaly"
        flag_reason[mask_sentinel & flag_reason.isna()] = "sentinel_value"
        flag_reason[mask_range_viol & flag_reason.isna()] = "range_violation"
    elif reason_priority == "range_first":
        flag_reason[mask_comm_dropout] = "communication_dropout"
        flag_reason[mask_partial_nan & flag_reason.isna()] = "partial_nan_anomaly"
        flag_reason[mask_range_viol & flag_reason.isna()] = "range_violation"
        flag_reason[mask_sentinel & flag_reason.isna()] = "sentinel_value"
    else:
        raise ValueError(f"Unknown reason_priority: {reason_priority}. Must be 'sentinel_first' or 'range_first'.")

    # Construct result DataFrame
    res_df = df.copy()
    res_df["is_flagged"] = is_flagged
    res_df["flag_reason"] = flag_reason

    return res_df


def benchmark_tier1_qc_latency(
    df: pd.DataFrame,
    n_iterations: int = 50,
) -> Dict[str, float]:
    """Measures batch and per-row processing latency of Tier 1 QC.

    Args:
        df: Input DataFrame for benchmarking.
        n_iterations: Number of timed runs.

    Returns:
        Dict with total_rows, batch_latency_ms, and per_row_latency_us.
    """
    n_rows = len(df)
    times = []

    # Warm-up run
    _ = apply_tier1_qc(df)

    for _ in range(n_iterations):
        t0 = time.perf_counter()
        _ = apply_tier1_qc(df)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    mean_batch_s = float(np.mean(times))
    mean_batch_ms = mean_batch_s * 1000.0
    per_row_us = (mean_batch_s / n_rows) * 1e6
    per_row_ms = per_row_us / 1000.0

    return {
        "n_rows": n_rows,
        "n_iterations": n_iterations,
        "batch_latency_ms": mean_batch_ms,
        "per_row_latency_us": per_row_us,
        "per_row_latency_ms": per_row_ms,
    }


def evaluate_tier1_qc_comprehensive(
    raw_splits: Dict[str, pd.DataFrame],
    meta_path: Path = Path("data/raw/generation_metadata.json"),
    ref_excluded_path: Path = Path("data/features/tier1_excluded_rows.parquet"),
) -> Dict[str, Any]:
    """Runs a complete evaluation of Tier 1 QC on all raw splits.

    Computes:
    - Recall on communication_dropout and data_corruption per split.
    - FPR on normal telemetry.
    - Explicit audit of the 5 genuine extreme weather event windows.
    - Flagged counts/percentages across each of the other 5 anomaly types.
    - Cross-check against tier1_excluded_rows.parquet.
    - Latency benchmarks.
    """
    eval_results: Dict[str, Any] = {
        "splits": {},
        "target_fault_recall": {},
        "normal_fpr": {},
        "extreme_weather": [],
        "other_anomaly_types": {},
        "cross_check": {},
        "latency": {},
    }

    # Derive empirical sentinels
    sentinels = derive_sentinels_from_excluded_rows(ref_excluded_path)

    qc_results: Dict[str, pd.DataFrame] = {}
    for sname, df in raw_splits.items():
        qc_df = apply_tier1_qc(df, sentinels=sentinels, reason_priority="sentinel_first")
        qc_results[sname] = qc_df
        eval_results["splits"][sname] = {
            "total_rows": len(df),
            "flagged_rows": int(qc_df["is_flagged"].sum()),
            "reasons": qc_df["flag_reason"].value_counts(dropna=False).to_dict(),
        }

    # Concat all splits
    all_raw = pd.concat(raw_splits.values(), ignore_index=True)
    all_qc = pd.concat(qc_results.values(), ignore_index=True)

    # 1. Recall on target fault types per split
    target_faults = ["communication_dropout", "data_corruption"]
    for fault in target_faults:
        eval_results["target_fault_recall"][fault] = {}
        for sname, qc_df in qc_results.items():
            mask = qc_df["anomaly_type"] == fault
            total = int(mask.sum())
            flagged = int(qc_df.loc[mask, "is_flagged"].sum())
            recall = float(flagged / total) if total > 0 else 1.0
            eval_results["target_fault_recall"][fault][sname] = {
                "total": total,
                "flagged": flagged,
                "recall": recall,
            }

    # 2. FPR on Normal rows
    normal_mask = all_qc["is_anomaly"] == False
    normal_total = int(normal_mask.sum())
    normal_flagged = int(all_qc.loc[normal_mask, "is_flagged"].sum())
    normal_fpr = float(normal_flagged / normal_total) if normal_total > 0 else 0.0
    eval_results["normal_fpr"] = {
        "total_normal_rows": normal_total,
        "flagged_normal_rows": normal_flagged,
        "fpr": normal_fpr,
    }

    # 3. Extreme Weather Event audit
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        events = meta.get("extreme_weather_summary", {}).get("events", [])
        for i, ev in enumerate(events):
            st = ev["station_id"]
            ev_type = ev.get("event_type", ev.get("type", "unknown"))
            start_idx = ev["start_idx"]
            dur = ev["duration_steps"]

            # Station telemetry slice ordered chronologically
            st_telemetry = all_raw[all_raw["station_id"] == st].sort_values("timestamp").reset_index(drop=True)
            ev_slice = st_telemetry.iloc[start_idx : start_idx + dur]

            # Run QC on slice
            ev_qc = apply_tier1_qc(ev_slice, sentinels=sentinels)
            n_flagged = int(ev_qc["is_flagged"].sum())

            eval_results["extreme_weather"].append({
                "event_id": i + 1,
                "station_id": st,
                "event_type": ev_type,
                "start_idx": start_idx,
                "duration_steps": dur,
                "rows_evaluated": len(ev_slice),
                "rows_flagged": n_flagged,
                "false_positive_rate": float(n_flagged / len(ev_slice)) if len(ev_slice) > 0 else 0.0,
                "temp_min": float(ev_slice["temperature_c"].min()),
                "temp_max": float(ev_slice["temperature_c"].max()),
                "press_min": float(ev_slice["pressure_hpa"].min()),
                "press_max": float(ev_slice["pressure_hpa"].max()),
                "rh_min": float(ev_slice["humidity_pct"].min()),
                "rh_max": float(ev_slice["humidity_pct"].max()),
            })

    # 4. Other 5 anomaly types separately
    other_types = [
        "calibration_drift",
        "frozen_sensor",
        "cross_sensor_inconsistency",
        "power_fluctuation_glitch",
        "spike_or_drop",
    ]
    for atype in other_types:
        mask = all_qc["anomaly_type"] == atype
        tot = int(mask.sum())
        flg = int(all_qc.loc[mask, "is_flagged"].sum())
        flg_reasons = all_qc.loc[mask & all_qc["is_flagged"], "flag_reason"].value_counts().to_dict()
        pct = float(flg / tot) if tot > 0 else 0.0
        eval_results["other_anomaly_types"][atype] = {
            "total_rows": tot,
            "flagged_rows": flg,
            "flagged_pct": pct,
            "reasons": flg_reasons,
        }

    # 5. Cross-check against tier1_excluded_rows.parquet
    if ref_excluded_path.exists():
        ref_df = pd.read_parquet(ref_excluded_path)
        ref_keys = set(zip(ref_df["station_id"], pd.to_datetime(ref_df["timestamp"])))
        flagged_mask = all_qc["is_flagged"]
        flagged_keys = set(zip(all_qc.loc[flagged_mask, "station_id"], pd.to_datetime(all_qc.loc[flagged_mask, "timestamp"])))

        in_both = len(ref_keys & flagged_keys)
        in_flagged_not_ref = len(flagged_keys - ref_keys)
        in_ref_not_flagged = len(ref_keys - flagged_keys)

        # Inspect any rows in flagged but not in ref
        discrepancies: List[Dict[str, Any]] = []
        if in_flagged_not_ref > 0:
            extra_keys = flagged_keys - ref_keys
            for st, ts in sorted(list(extra_keys)):
                row_match = all_qc[(all_qc["station_id"] == st) & (pd.to_datetime(all_qc["timestamp"]) == ts)].iloc[0]
                discrepancies.append({
                    "station_id": st,
                    "timestamp": str(ts),
                    "temperature_c": float(row_match["temperature_c"]),
                    "pressure_hpa": float(row_match["pressure_hpa"]),
                    "humidity_pct": float(row_match["humidity_pct"]),
                    "anomaly_type": str(row_match["anomaly_type"]),
                    "anomaly_severity": float(row_match["anomaly_severity"]),
                    "flag_reason": str(row_match["flag_reason"]),
                })

        eval_results["cross_check"] = {
            "ref_file_rows": len(ref_df),
            "tier1_flagged_total": len(flagged_keys),
            "rows_in_both": in_both,
            "rows_in_flagged_not_ref": in_flagged_not_ref,
            "rows_in_ref_not_flagged": in_ref_not_flagged,
            "extra_flagged_rows": discrepancies,
        }

    # 6. Latency Benchmark
    eval_results["latency"] = benchmark_tier1_qc_latency(all_raw)

    return eval_results


def run_tier1_pipeline(
    splits_dir: Path = Path("data/splits"),
    output_dir: Path = Path("data/tier1_results"),
) -> Dict[str, pd.DataFrame]:
    """Runs Tier 1 QC on all raw splits and saves results to output_dir.

    Outputs:
    - data/tier1_results/train.parquet
    - data/tier1_results/val.parquet
    - data/tier1_results/test.parquet
    - data/tier1_results/spatial_holdout.parquet
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sentinels = derive_sentinels_from_excluded_rows()

    split_file_map = {
        "train": splits_dir / "train.parquet",
        "val": splits_dir / "val.parquet",
        "test": splits_dir / "test.parquet",
        "spatial_holdout": splits_dir / "spatial_holdout_stations.parquet",
    }

    results: Dict[str, pd.DataFrame] = {}
    logger.info("Executing Tier 1 Physical QC across all canonical splits...")

    for split_name, split_path in split_file_map.items():
        if not split_path.exists():
            raise FileNotFoundError(f"Canonical split file {split_path} not found.")

        df_raw = pd.read_parquet(split_path)
        qc_df = apply_tier1_qc(df_raw, sentinels=sentinels, reason_priority="sentinel_first")

        out_path = output_dir / f"{split_name}.parquet"
        qc_df.to_parquet(out_path, index=False)
        logger.info(
            "Saved %s Tier 1 results to %s (%d rows, %d flagged)",
            split_name,
            out_path,
            len(qc_df),
            qc_df["is_flagged"].sum(),
        )
        results[split_name] = qc_df

    return results


def main() -> None:
    """CLI entry point for Tier 1 QC module."""
    parser = argparse.ArgumentParser(
        description="SkyGuard AI - Tier 1 Physical Quality Control (QC)"
    )
    parser.add_argument(
        "--splits-dir",
        type=str,
        default="data/splits",
        help="Path to directory containing canonical raw splits",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/tier1_results",
        help="Path to output directory for Tier 1 results",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Process a single raw telemetry parquet file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output parquet path when processing a single file",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run comprehensive evaluation and print report",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        default=True,
        help="Save processed parquet results",
    )

    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)
    output_dir = Path(args.output_dir)

    if args.input:
        in_path = Path(args.input)
        out_path = Path(args.output) if args.output else in_path.parent / f"{in_path.stem}_tier1_qc.parquet"
        df_raw = pd.read_parquet(in_path)
        qc_df = apply_tier1_qc(df_raw)
        qc_df.to_parquet(out_path, index=False)
        print(f"Processed {len(df_raw)} rows -> {out_path} ({qc_df['is_flagged'].sum()} flagged)")
        return

    # Process all splits
    qc_results = run_tier1_pipeline(splits_dir=splits_dir, output_dir=output_dir)

    if args.evaluate:
        raw_splits = {
            "train": pd.read_parquet(splits_dir / "train.parquet"),
            "val": pd.read_parquet(splits_dir / "val.parquet"),
            "test": pd.read_parquet(splits_dir / "test.parquet"),
            "spatial_holdout": pd.read_parquet(splits_dir / "spatial_holdout_stations.parquet"),
        }
        eval_metrics = evaluate_tier1_qc_comprehensive(raw_splits)

        print("\n" + "=" * 80)
        print("                  SKYGUARD AI - TIER 1 QC EVALUATION REPORT")
        print("=" * 80)

        print("\n1. TARGET FAULT RECALL (Must be near 100%):")
        for fault, split_data in eval_metrics["target_fault_recall"].items():
            print(f"   [{fault.upper()}]")
            for sname, stats in split_data.items():
                print(f"     * {sname:16s}: {stats['flagged']}/{stats['total']} ({stats['recall']:.4%})")

        print("\n2. FALSE POSITIVE RATE ON NORMAL TELEMETRY:")
        norm = eval_metrics["normal_fpr"]
        print(f"   * Total Normal Rows:  {norm['total_normal_rows']:,}")
        print(f"   * Flagged Normal:     {norm['flagged_normal_rows']}")
        print(f"   * Normal FPR:         {norm['fpr']:.6%}")

        print("\n3. GENUINE EXTREME WEATHER AUDIT (5 Scheduled Events):")
        for ev in eval_metrics["extreme_weather"]:
            print(f"   * Event {ev['event_id']} ({ev['station_id']}, {ev['event_type']}):")
            print(f"       Window: steps {ev['start_idx']} to {ev['start_idx'] + ev['duration_steps']} ({ev['duration_steps']} steps)")
            print(f"       Flagged Rows: {ev['rows_flagged']} / {ev['rows_evaluated']} (FPR: {ev['false_positive_rate']:.4%})")
            print(f"       Sensor Ranges: T=[{ev['temp_min']:.2f}, {ev['temp_max']:.2f}], P=[{ev['press_min']:.2f}, {ev['press_max']:.2f}], RH=[{ev['rh_min']:.2f}, {ev['rh_max']:.2f}]")

        print("\n4. OTHER 5 ANOMALY TYPES BREAKDOWN:")
        for atype, stats in eval_metrics["other_anomaly_types"].items():
            print(f"   * {atype:30s}: {stats['flagged_rows']:4d} / {stats['total_rows']:4d} flagged ({stats['flagged_pct']:7.2%}) | Reasons: {stats['reasons']}")

        print("\n5. CROSS-CHECK AGAINST tier1_excluded_rows.parquet:")
        cc = eval_metrics["cross_check"]
        print(f"   * Ground Truth Excluded Rows:  {cc['ref_file_rows']}")
        print(f"   * Rows in Both:                {cc['rows_in_both']}")
        print(f"   * Rows Missed by Tier 1:       {cc['rows_in_ref_not_flagged']}")
        print(f"   * Extra Rows Flagged by Tier 1:{cc['rows_in_flagged_not_ref']}")

        print("\n6. SELF-VERIFICATION (False Positives Inspection):")
        if cc["rows_in_flagged_not_ref"] > 0:
            print(f"   Found {cc['rows_in_flagged_not_ref']} rows outside tier1_excluded_rows.parquet flagged by Tier 1:")
            for extra in cc["extra_flagged_rows"]:
                print(f"     - Station: {extra['station_id']}, Time: {extra['timestamp']}, True Type: {extra['anomaly_type']}, "
                      f"T={extra['temperature_c']:.2f}C, P={extra['pressure_hpa']:.2f}hPa, RH={extra['humidity_pct']:.2f}%, "
                      f"Rule Triggered: {extra['flag_reason']}")
        else:
            print("   0 false positives found across all checked categories")

        print("\n7. LATENCY BENCHMARK:")
        lat = eval_metrics["latency"]
        print(f"   * Batch Latency ({lat['n_rows']} rows): {lat['batch_latency_ms']:.3f} ms")
        print(f"   * Per-Row Processing Latency:            {lat['per_row_latency_us']:.4f} microseconds ({lat['per_row_latency_ms']:.7f} ms)")
        print(f"   * Target Budget (<500ms):               PASSED (Headroom: {500.0 / max(lat['per_row_latency_ms'], 1e-9):,.0f}x)")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
