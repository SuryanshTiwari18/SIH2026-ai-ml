"""
SkyGuard AI — Season-Safety Regression & Ablation Verification (Part C).
Runs 4 configurations driven strictly by versioned JSON config files:
1. BASELINE: models/config_ablation_baseline.json
2. TIER1B_ONLY: models/config_ablation_tier1b_only.json
3. TIER1B_PLUS_T3_STATIC_BASELINE: models/config_ablation_tier1b_plus_t3_static.json
4. FULL: models/config_ablation_full.json

Evaluates across test.parquet and spatial_holdout.parquet.
Produces 4-column side-by-side comparison tables for Track 1 (Gated Hard Rule)
and Track 2 (Learned Fusion).
"""

import sys
sys.path.insert(0, ".")
import json
import logging
from pathlib import Path
from typing import Any, Dict, List
import joblib
import numpy as np
import pandas as pd

from src.physics import check_dew_point_depression_invariant
from src.tier1_qc import apply_tier1b_climatology_qc
from src.tier3_multivariate_spatial import (
    Rolling14DayMahalanobisBaseline,
    apply_tier3_models,
    compute_mslp_buddy_features,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_regression_verification")

CATEGORIES = [
    "data_corruption",
    "communication_dropout",
    "spike_or_drop",
    "power_fluctuation_glitch",
    "frozen_sensor",
    "calibration_drift",
    "cross_sensor_inconsistency",
]

CONFIG_FILES = [
    Path("models/config_ablation_baseline.json"),
    Path("models/config_ablation_tier1b_only.json"),
    Path("models/config_ablation_tier1b_plus_t3_static.json"),
    Path("models/config_ablation_full.json"),
    Path("models/config_ablation_full_fixed.json"),
]


def load_artifacts():
    clf_artifact = joblib.load("models/tier4_fusion_classifier.joblib")
    clf = clf_artifact["model"]
    with open("models/spatial_neighbors.json", "r") as f:
        neighbors_map = json.load(f)
    return clf, neighbors_map


def evaluate_metrics(df: pd.DataFrame, flag_col: str = "fusion_flagged") -> dict:
    """Computes recall per fault category and normal false-positive rate."""
    recalls = {}
    for cat in CATEGORIES:
        sub = df[df["anomaly_type"] == cat]
        total = len(sub)
        flagged = int((sub[flag_col] == True).sum()) if total > 0 else 0
        rec = (flagged / total * 100.0) if total > 0 else 0.0
        recalls[cat] = {"total": total, "flagged": flagged, "recall": rec}

    norm = df[~df["is_anomaly"]]
    n_norm = len(norm)
    n_flag_norm = int((norm[flag_col] == True).sum())
    fpr = (n_flag_norm / n_norm * 100.0) if n_norm > 0 else 0.0

    return {
        "recalls": recalls,
        "normal_total": n_norm,
        "normal_flagged": n_flag_norm,
        "normal_fpr": fpr,
    }


def run_pipeline_for_config(
    split_name: str,
    config: Dict[str, Any],
    clf: Any,
    neighbors_map: Dict[str, List[str]],
    op_clean_ref: pd.DataFrame,
) -> Dict[str, Any]:
    """Runs pipeline evaluation using parameters loaded from JSON config."""
    df = pd.read_parquet(f"data/tier4_results/{split_name}.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    tier1b_enabled = config.get("tier1b_enabled", False)
    use_rolling_baseline = config.get("use_rolling_baseline", False)
    freeze_on_sustained_deviation = bool(config.get("freeze_on_sustained_deviation", False))
    freeze_threshold = int(config.get("freeze_threshold", 3))
    use_mslp_reduction = config.get("use_mslp_reduction", False)
    dew_point_check_enabled = config.get("dew_point_check_enabled", False)
    dew_point_tolerance_c = float(config.get("dew_point_tolerance_c", 0.5))
    rolling_alpha = float(config.get("rolling_baseline_alpha", 0.0235))

    th_config = {
        "mahalanobis_threshold": float(config.get("mahalanobis_threshold", 6.5)),
        "buddy_delta_threshold": float(config.get("buddy_delta_threshold", 2.0)),
        "buddy_peer_threshold": float(config.get("buddy_peer_threshold", 2.0)),
    }

    df_run = df.copy()

    # Step 1: Tier 1B Climatology
    if tier1b_enabled:
        clim_file = Path("data/real_climatology/station_month_climatology_heldout_fit.json")
        df_run = apply_tier1b_climatology_qc(df_run, climatology_path=clim_file)
        new_t1_flag = df_run["tier1_flagged"] | df_run["tier1b_flagged"]
        df_run["tier1_flagged"] = new_t1_flag
    else:
        df_run["tier1b_flagged"] = False

    # Step 2: Tier 3 MSLP Buddy features
    if use_mslp_reduction:
        ref = op_clean_ref if split_name == "spatial_holdout" else None
        df_run = compute_mslp_buddy_features(
            df_run,
            neighbors_map,
            metadata_path=Path("data/raw/generation_metadata.json"),
            reference_telemetry=ref,
        )

    # Step 3: Tier 3 Rolling Baseline
    if use_rolling_baseline:
        rolling_engine = Rolling14DayMahalanobisBaseline(
            alpha=rolling_alpha,
            metadata_path=Path("data/raw/generation_metadata.json"),
            freeze_on_sustained_deviation=freeze_on_sustained_deviation,
            freeze_threshold=freeze_threshold,
        )
        hours = df_run["timestamp"].dt.hour.values
        stations = df_run["station_id"].values
        x_mat = df_run[["temperature_c", "pressure_hpa", "humidity_pct"]].values
        n = len(df_run)
        dm_rolling = np.zeros(n, dtype=np.float64)

        for i in range(n):
            if np.isnan(x_mat[i]).any():
                dm_rolling[i] = 0.0
            else:
                dm_rolling[i] = rolling_engine.compute_and_update(
                    stations[i], int(hours[i]), x_mat[i], update=True
                )
        df_run["mahalanobis_dist_rolling"] = dm_rolling

    # Step 4: Tier 3 Model Logic
    df_run = apply_tier3_models(
        df_run,
        th_config,
        use_rolling_baseline=use_rolling_baseline,
        use_mslp_reduction=use_mslp_reduction,
        check_dew_point=dew_point_check_enabled,
        dew_point_tolerance_c=dew_point_tolerance_c,
    )

    # Step 5: Tier 4 Decision Fusion
    # 5a. Track 1 Gated Hard Rule
    t1_flag = df_run["tier1_flagged"] == True
    t2_raw = (df_run["if_flagged"] == True) | (df_run["gru_flagged"] == True)
    t3_raw = (df_run["tier3_multivariate_flagged"] == True) | (df_run["buddy_flagged"] == True)
    t23_raw = t2_raw | t3_raw
    t23_confirmed = t23_raw & (df_run["isolated_deviation"] == True)
    df_run["hard_rule_flagged"] = t1_flag | t23_confirmed

    # 5b. Track 2 Learned Meta-Classifier
    mahal_feat = df_run["mahalanobis_dist_rolling"] if (use_rolling_baseline and "mahalanobis_dist_rolling" in df_run.columns) else df_run["mahalanobis_dist"]
    feat_df = pd.DataFrame({
        "tier1_flagged": df_run["tier1_flagged"].astype(int),
        "if_score": df_run["if_score"].fillna(0.0),
        "gru_score": df_run["gru_score"].fillna(0.0),
        "mahalanobis_dist": mahal_feat,
        "buddy_flagged": df_run["buddy_flagged"].astype(int),
        "isolated_deviation": df_run["isolated_deviation"].astype(int),
    })

    full_mask = df_run["tier_coverage"] == "full"
    pred_types = pd.Series("normal", index=df_run.index, dtype="object")
    flagged = pd.Series(False, index=df_run.index, dtype=bool)

    if full_mask.any():
        preds_full = clf.predict(feat_df.loc[full_mask])
        pred_types.loc[full_mask] = preds_full
        flagged.loc[full_mask] = preds_full != "normal"

    # Tier 1 deterministic override
    t1_mask = df_run["tier1_flagged"] == True
    if t1_mask.any():
        flagged.loc[t1_mask] = True
        pred_types.loc[t1_mask] = np.where(
            df_run.loc[t1_mask, "tier1_reason"] == "communication_dropout",
            "communication_dropout",
            "data_corruption",
        )

    # Insufficient context rows
    insuf_mask = (df_run["tier_coverage"] == "tier1_only") & (~t1_mask)
    if insuf_mask.any():
        flagged.loc[insuf_mask] = False
        pred_types.loc[insuf_mask] = "insufficient_context"

    df_run["fusion_predicted_type"] = pred_types
    df_run["fusion_flagged"] = flagged

    hard_rule_metrics = evaluate_metrics(df_run, "hard_rule_flagged")
    fusion_metrics = evaluate_metrics(df_run, "fusion_flagged")

    return {
        "hard_rule": hard_rule_metrics,
        "fusion": fusion_metrics,
        "df": df_run,
    }


def run_ablation():
    clf, neighbors_map = load_artifacts()

    # Load clean operational reference telemetry for holdout spatial buddy check
    train_df = pd.read_parquet("data/tier4_results/train.parquet")
    op_clean_ref = train_df[~train_df["anomaly_type"].isin(["communication_dropout", "data_corruption"]) & (train_df["temperature_c"] > -50.0)].copy()

    configs = []
    for cfg_path in CONFIG_FILES:
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
            configs.append((cfg_path.stem, cfg))

    ablation_results = {}

    for split in ["test", "spatial_holdout"]:
        print(f"\n================================================================================")
        print(f"RUNNING 4-CONFIGURATION ABLATION ON {split.upper()}")
        print(f"================================================================================")
        ablation_results[split] = {}

        for cfg_name, cfg in configs:
            print(f"Running config: {cfg_name} ({cfg['config_name']})...")
            res = run_pipeline_for_config(split, cfg, clf, neighbors_map, op_clean_ref)
            ablation_results[split][cfg_name] = res

    # Print side-by-side tables
    for split in ["test", "spatial_holdout"]:
        print(f"\n\n================================================================================")
        print(f"ABLATION MATRIX: {split.upper()} (Track 2 Learned Fusion)")
        print(f"================================================================================")
        c_names = [c[0] for c in configs]
        c_labels = [c[1].get("config_name", c[0])[:12] for c in configs]
        header = f"{'Fault Category':28s} | " + " | ".join([f"{lbl:13s}" for lbl in c_labels])
        print(header)
        sep_len = len(header)
        print("-" * sep_len)

        for cat in CATEGORIES:
            row_str = f"{cat:28s} | "
            for cn in c_names:
                rec = ablation_results[split][cn]["fusion"]["recalls"][cat]["recall"]
                row_str += f"{rec:11.2f}% | "
            print(row_str)

        print("-" * sep_len)
        fpr_row = f"{'Normal Telemetry (FPR)':28s} | "
        for cn in c_names:
            fpr = ablation_results[split][cn]["fusion"]["normal_fpr"]
            fpr_row += f"{fpr:11.2f}% | "
        print(fpr_row)

        print(f"\n\n================================================================================")
        print(f"ABLATION MATRIX: {split.upper()} (Track 1 Gated Hard Rule)")
        print(f"================================================================================")
        print(header)
        print("-" * sep_len)
        for cat in CATEGORIES:
            row_str = f"{cat:28s} | "
            for cn in c_names:
                rec = ablation_results[split][cn]["hard_rule"]["recalls"][cat]["recall"]
                row_str += f"{rec:11.2f}% | "
            print(row_str)

        print("-" * sep_len)
        fpr_row = f"{'Normal Telemetry (FPR)':28s} | "
        for cn in c_names:
            fpr = ablation_results[split][cn]["hard_rule"]["normal_fpr"]
            fpr_row += f"{fpr:11.2f}% | "
        print(fpr_row)

    # Save ablation results json
    json_summary = {}
    for split in ["test", "spatial_holdout"]:
        json_summary[split] = {}
        for cn in c_names:
            json_summary[split][cn] = {
                "track1_hard_rule": ablation_results[split][cn]["hard_rule"],
                "track2_fusion": ablation_results[split][cn]["fusion"],
            }

    with open("data/real_climatology/ablation_results.json", "w") as f:
        json.dump(json_summary, f, indent=2)
    print("\nSaved data/real_climatology/ablation_results.json")

    return ablation_results


if __name__ == "__main__":
    run_ablation()
