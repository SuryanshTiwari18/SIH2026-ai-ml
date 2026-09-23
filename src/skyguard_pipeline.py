"""
SkyGuard AI (SIH 2026, Problem Statement 26073)
Master Production Pipeline: SkyGuardPipeline

A callable, end-to-end meteorological anomaly detection & sensor health pipeline
integrating all 5 tiers into one unified deployable architecture:
- Tier 1: Deterministic Physical Quality Control (range, step, dropout, sentinel)
- Tier 2: Temporal Anomaly Detection (Isolation Forest + GRU-Autoencoder)
- Tier 3: Multivariate Consistency (Mahalanobis) & Spatial Buddy Consensus
- Tier 4: Two-Track Operational Decision Fusion (Track 1 Operational Alert vs. Track 2 Maintenance Queue)
- Tier 5: TreeSHAP Attribution Rationales, Sensor Health Index (SHI), and 3-NN Value Correction

Edge Cases Handled:
1. Cold Start: Station with <24 history steps runs Tier 1 physical QC; higher tiers return
   'insufficient_context' until history buffer accumulates.
2. Missing Neighbors: Single-station fallback mode when concurrent buddy telemetry is absent.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.physics import (
    calculate_baseline_pressure_hpa,
    calculate_dew_point_c,
    calculate_saturation_vapor_pressure_hpa,
)
from src.stations import INDIAN_AWS_STATIONS, get_station_catalog

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SkyGuard.Pipeline")


# -----------------------------------------------------------------------------
# PyTorch GRU-Autoencoder Architecture
# -----------------------------------------------------------------------------
class GRUAutoencoder(nn.Module):
    """GRU-based Sequence Autoencoder for temporal anomaly detection."""

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 32,
        latent_dim: int = 16,
        seq_len: int = 72,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len

        self.encoder_gru = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.enc_dense = nn.Linear(hidden_dim, latent_dim)
        self.dec_dense = nn.Linear(latent_dim, hidden_dim)
        self.decoder_gru = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.out_dense = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.encoder_gru(x)
        latent = self.enc_dense(h_n.squeeze(0))
        dec_init = self.dec_dense(latent).unsqueeze(1)
        dec_input = dec_init.repeat(1, self.seq_len, 1)
        dec_out, _ = self.decoder_gru(dec_input)
        recon = self.out_dense(dec_out)
        return recon


# -----------------------------------------------------------------------------
# SkyGuard Master Pipeline
# -----------------------------------------------------------------------------
class SkyGuardPipeline:
    """Production callable end-to-end SkyGuard AI anomaly detection pipeline."""

    FEATURE_SCALER_COLS = [
        "delta_T", "delta_P", "delta_RH",
        "var_T_1h", "var_T_6h", "vol_ratio_T",
        "var_P_1h", "var_P_6h", "vol_ratio_P",
        "var_RH_1h", "var_RH_6h", "vol_ratio_RH",
        "sin_hour", "cos_hour", "dew_point_dep", "vpd",
        "mahalanobis_dist", "delta_T_buddy", "delta_P_cluster",
    ]

    TEMPORAL_MODEL_COLS = [
        "delta_T_scaled", "delta_P_scaled", "delta_RH_scaled",
        "var_T_1h_scaled", "var_T_6h_scaled", "vol_ratio_T_scaled",
        "var_P_1h_scaled", "var_P_6h_scaled", "vol_ratio_P_scaled",
        "var_RH_1h_scaled", "var_RH_6h_scaled", "vol_ratio_RH_scaled",
    ]

    TIER4_FEATURES = [
        "tier1_flagged",
        "if_score",
        "gru_score",
        "mahalanobis_dist",
        "buddy_flagged",
        "isolated_deviation",
    ]

    TRACK1_ASSIGNED_TYPES = [
        "data_corruption",
        "communication_dropout",
        "spike_or_drop",
        "power_fluctuation_glitch",
    ]

    TRACK2_ASSIGNED_TYPES = [
        "frozen_sensor",
        "calibration_drift",
        "cross_sensor_inconsistency",
    ]

    def __init__(self, models_dir: Union[str, Path] = "models", window_size: int = 72) -> None:
        self.models_dir = Path(models_dir)
        self.window_size = window_size
        self.station_catalog = get_station_catalog()

        logger.info("Initializing SkyGuardPipeline and loading persistent model artifacts...")

        # 1. Feature Scaler
        scaler_path = self.models_dir / "feature_scaler.joblib"
        self.feature_scaler = joblib.load(scaler_path)

        # 2. Isolation Forest
        if_path = self.models_dir / "isolation_forest.joblib"
        self.isolation_forest = joblib.load(if_path)

        # 3. GRU-Autoencoder
        gru_path = self.models_dir / "gru_autoencoder.pt"
        gru_data = torch.load(gru_path, map_location="cpu")
        self.gru_autoencoder = GRUAutoencoder(input_dim=12, hidden_dim=32, latent_dim=16, seq_len=window_size)
        self.gru_autoencoder.load_state_dict(gru_data["state_dict"])
        self.gru_autoencoder.eval()

        # 4. Mahalanobis Statistics & Fallbacks
        mahal_path = self.models_dir / "mahalanobis_stats.joblib"
        self.mahalanobis_stats = joblib.load(mahal_path)

        # 5. Spatial Neighbors Graph
        nbrs_path = self.models_dir / "spatial_neighbors.json"
        with open(nbrs_path, "r") as f:
            self.spatial_neighbors = json.load(f)

        # 6. Volatility Baselines
        vol_path = self.models_dir / "volatility_baselines.json"
        with open(vol_path, "r") as f:
            self.volatility_baselines = json.load(f)

        # 7. Tier 4 Fusion Classifier (LightGBM)
        t4_path = self.models_dir / "tier4_fusion_classifier.joblib"
        t4_dict = joblib.load(t4_path)
        self.tier4_classifier = t4_dict["model"]
        self.classifier_classes = t4_dict["classes"]

        # 8. Tier Thresholds & Policies
        t2_thresh_path = self.models_dir / "tier2_thresholds.json"
        with open(t2_thresh_path, "r") as f:
            self.tier2_thresholds = json.load(f)

        t3_thresh_path = self.models_dir / "tier3_thresholds.json"
        with open(t3_thresh_path, "r") as f:
            self.tier3_thresholds = json.load(f)

        t4_cfg_path = self.models_dir / "tier4_hard_rule_config.json"
        with open(t4_cfg_path, "r") as f:
            self.tier4_config = json.load(f)

        # State buffers
        self.station_history: Dict[str, List[dict]] = {}
        self.station_shi: Dict[str, float] = {}
        self.network_latest: Dict[str, dict] = {}

        logger.info("SkyGuardPipeline successfully initialized. Ready for telemetry ingestion.")

    def reset_state(self) -> None:
        """Resets all streaming station rolling history buffers and health indices."""
        self.station_history.clear()
        self.station_shi.clear()
        self.network_latest.clear()
        logger.info("Pipeline internal streaming state reset.")

    EMPIRICAL_SENTINELS: Dict[str, Set[float]] = {
        "temperature_c": {-999.0, -99.9, 999.9, 9999.0},
        "pressure_hpa": {-999.0, -99.9, 9999.0},  # Note: 999.0 and 999.9 are excluded (valid sea-level pressures)
        "humidity_pct": {-999.0, -25.0, 160.0},
    }

    def _check_tier1(
        self,
        station_id: str,
        t: float,
        p: float,
        rh: float,
        history: List[dict],
    ) -> Tuple[bool, Optional[str]]:
        """Tier 1: Deterministic Physical Quality Control."""
        # Sentinel checks with exact matching (0.01 tolerance for floating-point safety)
        for val, col in [(t, "temperature_c"), (p, "pressure_hpa"), (rh, "humidity_pct")]:
            if pd.isna(val):
                return True, "sentinel_value"
            if any(abs(val - s) < 0.01 for s in self.EMPIRICAL_SENTINELS[col]):
                return True, "sentinel_value"

        # Physical range checks
        if not (-40.0 <= t <= 60.0) or not (500.0 <= p <= 1100.0) or not (0.0 <= rh <= 100.0):
            return True, "range_violation"

        # Communication flatline check (3 consecutive identical values)
        if len(history) >= 2:
            prev1 = history[-1]
            prev2 = history[-2]
            if (
                t == prev1.get("t") == prev2.get("t")
                and p == prev1.get("p") == prev2.get("p")
                and rh == prev1.get("rh") == prev2.get("rh")
            ):
                return True, "communication_dropout"

        return False, None


    def _compute_mahalanobis(self, station_id: str, hour: int, t: float, p: float, rh: float) -> float:
        """Computes Mahalanobis distance with climate-zone fallback for holdout stations."""
        x = np.array([t, p, rh], dtype=float)
        stats = None

        if (station_id, hour) in self.mahalanobis_stats.get("station_hour", {}):
            stats = self.mahalanobis_stats["station_hour"][(station_id, hour)]
        else:
            # Climate-zone fallback
            st_meta = self.station_catalog.get(station_id)
            czone = st_meta.climate_zone if st_meta else "plains"
            if (czone, hour) in self.mahalanobis_stats.get("zone_hour", {}):
                stats = self.mahalanobis_stats["zone_hour"][(czone, hour)]
            else:
                stats = self.mahalanobis_stats.get("global_fallback")

        if stats is None:
            return 1.5

        diff = x - stats["mu"]
        dm_sq = float(np.dot(np.dot(diff, stats["inv_cov"]), diff))
        return math.sqrt(max(0.0, dm_sq))

    def _extract_features(
        self,
        station_id: str,
        timestamp: pd.Timestamp,
        t: float,
        p: float,
        rh: float,
        history: List[dict],
        concurrent_neighbors: Optional[dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float, float, float, bool]:
        """Extracts engineered features for Tiers 2-4."""
        hour = timestamp.hour
        sin_hour = math.sin(2.0 * math.pi * hour / 24.0)
        cos_hour = math.cos(2.0 * math.pi * hour / 24.0)

        # Physics
        dew_point = calculate_dew_point_c(t, rh)
        dew_point_dep = t - dew_point
        es = calculate_saturation_vapor_pressure_hpa(t)
        vpd = es * (1.0 - rh / 100.0)

        # Derivatives from previous step
        prev = history[-1]
        delta_T = t - prev["t"]
        delta_P = p - prev["p"]
        delta_RH = rh - prev["rh"]

        # Volatilities from history buffer
        t_hist = [h["t"] for h in history[-36:]] + [t]
        p_hist = [h["p"] for h in history[-36:]] + [p]
        rh_hist = [h["rh"] for h in history[-36:]] + [rh]

        var_T_1h = float(np.var(t_hist[-6:])) if len(t_hist) >= 6 else 0.1
        var_P_1h = float(np.var(p_hist[-6:])) if len(p_hist) >= 6 else 0.1
        var_RH_1h = float(np.var(rh_hist[-6:])) if len(rh_hist) >= 6 else 1.0

        var_T_6h = float(np.var(t_hist)) if len(t_hist) >= 12 else var_T_1h * 2.0
        var_P_6h = float(np.var(p_hist)) if len(p_hist) >= 12 else var_P_1h * 2.0
        var_RH_6h = float(np.var(rh_hist)) if len(rh_hist) >= 12 else var_RH_1h * 2.0

        # Baseline VolRatio
        st_base = self.volatility_baselines.get(station_id, {"temperature_c": 4.0, "pressure_hpa": 3.0, "humidity_pct": 10.0})
        vol_ratio_T = math.sqrt(max(0.0, var_T_1h)) / (st_base["temperature_c"] + 1e-4)
        vol_ratio_P = math.sqrt(max(0.0, var_P_1h)) / (st_base["pressure_hpa"] + 1e-4)
        vol_ratio_RH = math.sqrt(max(0.0, var_RH_1h)) / (st_base["humidity_pct"] + 1e-4)

        # Mahalanobis
        dm = self._compute_mahalanobis(station_id, hour, t, p, rh)

        # Spatial buddy features
        nbr_ids = self.spatial_neighbors.get(station_id, [])
        nbr_t_vals = []
        nbr_p_vals = []
        fallback_used = False

        lookup = concurrent_neighbors or self.network_latest
        for nid in nbr_ids:
            if nid in lookup and lookup[nid].get("t") is not None:
                nbr_t_vals.append(lookup[nid]["t"])
                nbr_p_vals.append(lookup[nid]["p"])

        if len(nbr_t_vals) > 0:
            delta_T_buddy = abs(float(t - nbr_t_vals[0]))
            delta_P_cluster = float(p - np.median(nbr_p_vals))
        else:
            delta_T_buddy = 0.0
            delta_P_cluster = 0.0
            fallback_used = True

        raw_19_df = pd.DataFrame([[
            delta_T, delta_P, delta_RH,
            var_T_1h, var_T_6h, vol_ratio_T,
            var_P_1h, var_P_6h, vol_ratio_P,
            var_RH_1h, var_RH_6h, vol_ratio_RH,
            sin_hour, cos_hour, dew_point_dep, vpd,
            dm, delta_T_buddy, delta_P_cluster
        ]], columns=self.FEATURE_SCALER_COLS)

        scaled_19 = self.feature_scaler.transform(raw_19_df)[0]
        scaled_12 = scaled_19[:12]


        return scaled_19, scaled_12, dm, delta_T, delta_P, fallback_used

    def _update_shi(self, station_id: str, penalty: float, decay_rate: float = 0.05, recovery_rate: float = 0.01) -> float:
        """Updates per-station Exponential Moving Average (EMA) Sensor Health Index."""
        cur_shi = self.station_shi.get(station_id, 100.0)
        p = min(100.0, max(0.0, penalty))

        if p > 0.0:
            target = max(0.0, 100.0 - p)
            new_shi = (1.0 - decay_rate) * cur_shi + decay_rate * target
        else:
            new_shi = (1.0 - recovery_rate) * cur_shi + recovery_rate * 100.0

        self.station_shi[station_id] = new_shi
        return new_shi

    def process_row(
        self,
        row: Dict[str, Any],
        concurrent_neighbors: Optional[Dict[str, dict]] = None,
    ) -> Dict[str, Any]:
        """
        Processes a single raw telemetry row through all 5 tiers.

        Args:
            row: dict containing 'station_id', 'timestamp', 'temperature_c', 'pressure_hpa', 'humidity_pct'.
            concurrent_neighbors: optional dict mapping neighbor station_ids to {'t': float, 'p': float, 'rh': float}.

        Returns:
            dict containing comprehensive diagnostic decision, rationales, health, and corrections.
        """
        t_start = time.perf_counter()

        station_id = str(row["station_id"])
        ts = pd.to_datetime(row["timestamp"])
        t = row.get("temperature_c")
        p = row.get("pressure_hpa")
        rh = row.get("humidity_pct")

        t = float(t) if t is not None and not pd.isna(t) else np.nan
        p = float(p) if p is not None and not pd.isna(p) else np.nan
        rh = float(rh) if rh is not None and not pd.isna(rh) else np.nan

        history = self.station_history.setdefault(station_id, [])

        # Stage 1: Tier 1 Physical QC
        t1_flag, t1_reason = self._check_tier1(station_id, t, p, rh, history)

        if t1_flag:
            pred_type = "communication_dropout" if t1_reason == "communication_dropout" else "data_corruption"
            shi = self._update_shi(station_id, penalty=40.0)

            # Suggest buddy correction if neighbors available
            corrected_val = None
            nbr_ids = self.spatial_neighbors.get(station_id, [])
            lookup = concurrent_neighbors or self.network_latest
            valid_t = [lookup[n]["t"] for n in nbr_ids if n in lookup and pd.notna(lookup[n].get("t"))]
            valid_p = [lookup[n]["p"] for n in nbr_ids if n in lookup and pd.notna(lookup[n].get("p"))]
            valid_rh = [lookup[n]["rh"] for n in nbr_ids if n in lookup and pd.notna(lookup[n].get("rh"))]
            if valid_t and valid_p and valid_rh:
                corrected_val = {
                    "temperature_c": round(float(np.median(valid_t)), 2),
                    "pressure_hpa": round(float(np.median(valid_p)), 2),
                    "humidity_pct": round(float(np.median(valid_rh)), 2),
                }

            rationale = f"Flagged deterministically by Tier 1 Physical QC: {t1_reason}."
            if t1_reason == "sentinel_value":
                rationale = "Flagged deterministically by Tier 1 Physical QC: sentinel value detected (hardware failure)."
            elif t1_reason == "communication_dropout":
                rationale = "Flagged deterministically by Tier 1 Physical QC: communication dropout detected (telemetry flatline)."

            elapsed_ms = (time.perf_counter() - t_start) * 1e3

            # Update history with nan or raw
            history.append({"timestamp": ts, "t": t, "p": p, "rh": rh, "scaled_12": np.zeros(12)})
            if len(history) > 144:
                history.pop(0)

            return {
                "station_id": station_id,
                "timestamp": str(ts),
                "is_anomaly": True,
                "predicted_type": pred_type,
                "confidence": 1.0,
                "which_track": "1",
                "shap_rationale_text": rationale,
                "sensor_health_index": round(shi, 2),
                "corrected_value_suggestion": corrected_val,
                "tier_signals": {
                    "tier1_flagged": True,
                    "tier1_reason": t1_reason,
                    "if_score": None,
                    "gru_score": None,
                    "mahalanobis_dist": None,
                    "buddy_flagged": None,
                    "isolated_deviation": None,
                },
                "diagnostics": {
                    "latency_ms": round(elapsed_ms, 3),
                    "history_length": len(history),
                    "neighbor_fallback_used": False,
                },
            }

        # Stage 2: Cold-Start Check (< 24 history steps)
        if len(history) < 24:
            shi = self._update_shi(station_id, penalty=0.0)
            history.append({"timestamp": ts, "t": t, "p": p, "rh": rh, "scaled_12": np.zeros(12)})
            elapsed_ms = (time.perf_counter() - t_start) * 1e3
            return {
                "station_id": station_id,
                "timestamp": str(ts),
                "is_anomaly": False,
                "predicted_type": "insufficient_context",
                "confidence": 0.0,
                "which_track": "insufficient_context",
                "shap_rationale_text": "Warmup window lookback buffer (<24 steps); insufficient context for higher tiers.",
                "sensor_health_index": round(shi, 2),
                "corrected_value_suggestion": None,
                "tier_signals": {
                    "tier1_flagged": False,
                    "tier1_reason": None,
                    "if_score": None,
                    "gru_score": None,
                    "mahalanobis_dist": None,
                    "buddy_flagged": None,
                    "isolated_deviation": None,
                },
                "diagnostics": {
                    "latency_ms": round(elapsed_ms, 3),
                    "history_length": len(history),
                    "neighbor_fallback_used": False,
                },
            }

        # Stage 3: Feature Engineering & Tiers 2-3 Scoring
        scaled_19, scaled_12, dm, delta_T, delta_P, fallback_used = self._extract_features(
            station_id, ts, t, p, rh, history, concurrent_neighbors
        )

        # Tier 2: Isolation Forest
        if_score = float(-self.isolation_forest.score_samples(scaled_12.reshape(1, -1))[0])
        if_flagged = if_score > self.tier2_thresholds["isolation_forest"]["threshold"]

        # Tier 2: GRU-Autoencoder
        hist_scaled = [h["scaled_12"] for h in history[-(self.window_size - 1):]] + [scaled_12]
        if len(hist_scaled) < self.window_size:
            pad = [hist_scaled[0]] * (self.window_size - len(hist_scaled))
            seq_arr = np.array(pad + hist_scaled, dtype=np.float32)
        else:
            seq_arr = np.array(hist_scaled[-self.window_size:], dtype=np.float32)

        seq_tensor = torch.from_numpy(seq_arr).unsqueeze(0)
        with torch.no_grad():
            recon_tensor = self.gru_autoencoder(seq_tensor)
            gru_score = float(torch.mean((seq_tensor - recon_tensor) ** 2).item())
        gru_flagged = gru_score > self.tier2_thresholds["gru_autoencoder"]["threshold"]

        # Tier 3: Mahalanobis & Buddy
        mahal_flagged = dm > self.tier3_thresholds["mahalanobis_threshold"]
        delta_T_buddy_scaled = scaled_19[17]
        delta_P_cluster_scaled = scaled_19[18]
        th_buddy = self.tier3_thresholds.get("buddy_peer_threshold", 2.0)
        th_delta = self.tier3_thresholds.get("buddy_delta_threshold", 2.0)
        buddy_flagged = bool((delta_T_buddy_scaled > th_buddy) or (abs(delta_P_cluster_scaled) > th_buddy))

        delta_T_scaled = scaled_19[0]
        delta_P_scaled = scaled_19[1]
        own_delta_large = bool((abs(delta_T_scaled) > th_delta) or (abs(delta_P_scaled) > th_delta))
        isolated_deviation = bool(own_delta_large and buddy_flagged)

        # Stage 4: Tier 4 Two-Track Decision Engine
        t2_raw_flag = if_flagged or gru_flagged
        t3_raw_flag = mahal_flagged or buddy_flagged
        t23_raw_flag = t2_raw_flag or t3_raw_flag

        hard_rule_flagged = bool(t23_raw_flag and isolated_deviation)
        hard_rule_suppressed = bool(t23_raw_flag and not isolated_deviation)

        # Learned Meta-Classifier inference
        t4_feat_df = pd.DataFrame([{
            "tier1_flagged": 0.0,
            "if_score": if_score,
            "gru_score": gru_score,
            "mahalanobis_dist": dm,
            "buddy_flagged": float(buddy_flagged),
            "isolated_deviation": float(isolated_deviation),
        }])

        lgbm_probs = self.tier4_classifier.predict_proba(t4_feat_df)[0]
        pred_idx = int(np.argmax(lgbm_probs))
        lgbm_class = self.classifier_classes[pred_idx]
        lgbm_conf = float(lgbm_probs[pred_idx])

        # Routing decision
        which_track = "1"
        is_anomaly = False
        predicted_type = "normal"
        confidence = float(lgbm_probs[self.classifier_classes.index("normal")])
        penalty = 0.0

        if hard_rule_flagged:
            # Track 1 Confirmed Operational Alert
            is_anomaly = True
            which_track = "1"
            if lgbm_class in ["spike_or_drop", "power_fluctuation_glitch"]:
                predicted_type = lgbm_class
                confidence = lgbm_conf
            else:
                predicted_type = "spike_or_drop" if gru_score > 5.0 else "power_fluctuation_glitch"
                confidence = 0.95
            penalty = 25.0
        elif lgbm_class in self.TRACK2_ASSIGNED_TYPES and not hard_rule_suppressed:
            # Track 2 Secondary Maintenance Queue Route
            is_anomaly = True
            which_track = "2"
            predicted_type = lgbm_class
            confidence = lgbm_conf
            penalty = 15.0
        elif lgbm_class in ["power_fluctuation_glitch", "spike_or_drop"] and not hard_rule_suppressed:
            # Low-amplitude glitch/spike routed to Track 2
            is_anomaly = True
            which_track = "2"
            predicted_type = lgbm_class
            confidence = lgbm_conf
            penalty = 10.0
        else:
            # Normal or Suppressed Severe Weather
            is_anomaly = False
            which_track = "1"
            predicted_type = "normal"
            confidence = float(lgbm_probs[self.classifier_classes.index("normal")])
            penalty = 0.0

        # Stage 5: Tier 5 TreeSHAP Attribution & Rationales
        if is_anomaly:
            contribs = self.tier4_classifier.booster_.predict(t4_feat_df, pred_contrib=True).reshape(len(self.classifier_classes), 7)
            target_idx = self.classifier_classes.index(predicted_type) if predicted_type in self.classifier_classes else pred_idx
            feat_contribs = contribs[target_idx, :-1]

            ranked = sorted(zip(self.TIER4_FEATURES, feat_contribs), key=lambda x: x[1], reverse=True)
            top_f1, val1 = ranked[0]
            top_f2, val2 = ranked[1]

            if hard_rule_flagged and val1 < 0.10:
                if gru_score > 5.0:
                    rationale = f"Flagged by Track 1 Gated Hard Rule: confirmed isolated spatial deviation with high temporal reconstruction error (GRU={gru_score:.2f}, IF={if_score:.3f})."
                elif if_score > self.tier2_thresholds["isolation_forest"]["threshold"]:
                    rationale = f"Flagged by Track 1 Gated Hard Rule: confirmed isolated spatial deviation with elevated Isolation Forest score (IF={if_score:.3f})."
                else:
                    rationale = f"Flagged by Track 1 Gated Hard Rule: confirmed isolated spatial deviation (D_M={dm:.2f}, IF={if_score:.3f})."
            elif top_f1 == "mahalanobis_dist":
                rationale = f"Flagged primarily due to elevated Mahalanobis distance (D_M={dm:.2f}, SHAP: {val1:+.2f}), indicating multivariate thermodynamic inconsistency."
            elif top_f1 == "if_score":
                rationale = f"Flagged primarily due to elevated Isolation Forest score (IF={if_score:.3f}, SHAP: {val1:+.2f}) and GRU error (GRU={gru_score:.2f}), indicating transient hardware spike/glitch."
            elif top_f1 == "gru_score":
                rationale = f"Flagged primarily due to high temporal reconstruction error (GRU={gru_score:.2f}, SHAP: {val1:+.2f}), indicating sequence discontinuity."
            else:
                rationale = f"Flagged primarily due to {top_f1} (SHAP: {val1:+.2f}) supported by {top_f2}."
        else:
            rationale = "Normal telemetry: sensor signals remain within calibrated physical, temporal, and spatial bounds."

        # Sensor Health Index
        shi = self._update_shi(station_id, penalty=penalty)

        # 3-NN Value Correction
        corrected_val = None
        if is_anomaly:
            nbr_ids = self.spatial_neighbors.get(station_id, [])
            lookup = concurrent_neighbors or self.network_latest
            valid_t = [lookup[n]["t"] for n in nbr_ids if n in lookup and pd.notna(lookup[n].get("t"))]
            valid_p = [lookup[n]["p"] for n in nbr_ids if n in lookup and pd.notna(lookup[n].get("p"))]
            valid_rh = [lookup[n]["rh"] for n in nbr_ids if n in lookup and pd.notna(lookup[n].get("rh"))]
            if valid_t and valid_p and valid_rh:
                corrected_val = {
                    "temperature_c": round(float(np.median(valid_t)), 2),
                    "pressure_hpa": round(float(np.median(valid_p)), 2),
                    "humidity_pct": round(float(np.median(valid_rh)), 2),
                }

        # Update buffers
        history.append({"timestamp": ts, "t": t, "p": p, "rh": rh, "scaled_12": scaled_12})
        if len(history) > 144:
            history.pop(0)

        self.network_latest[station_id] = {"t": t, "p": p, "rh": rh, "timestamp": ts}
        elapsed_ms = (time.perf_counter() - t_start) * 1e3

        return {
            "station_id": station_id,
            "timestamp": str(ts),
            "is_anomaly": is_anomaly,
            "predicted_type": predicted_type,
            "confidence": round(confidence, 4),
            "which_track": which_track,
            "shap_rationale_text": rationale,
            "sensor_health_index": round(shi, 2),
            "corrected_value_suggestion": corrected_val,
            "tier_signals": {
                "tier1_flagged": False,
                "tier1_reason": None,
                "if_score": round(if_score, 4),
                "gru_score": round(gru_score, 4),
                "mahalanobis_dist": round(dm, 2),
                "buddy_flagged": buddy_flagged,
                "isolated_deviation": isolated_deviation,
            },
            "diagnostics": {
                "latency_ms": round(elapsed_ms, 3),
                "history_length": len(history),
                "neighbor_fallback_used": fallback_used,
            },
        }

    def process_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Processes a chronological batch of telemetry rows, maintaining state."""
        df_sorted = df.sort_values(["timestamp", "station_id"]).reset_index(drop=True)
        results = []

        for _, row in df_sorted.iterrows():
            res = self.process_row(row.to_dict())
            results.append({
                "station_id": res["station_id"],
                "timestamp": res["timestamp"],
                "is_anomaly": res["is_anomaly"],
                "predicted_type": res["predicted_type"],
                "confidence": res["confidence"],
                "which_track": res["which_track"],
                "sensor_health_index": res["sensor_health_index"],
                "shap_rationale_text": res["shap_rationale_text"],
                "latency_ms": res["diagnostics"]["latency_ms"],
            })

        return pd.DataFrame(results)

    def benchmark_latency(self, df_sample: pd.DataFrame, n_samples: int = 500) -> Dict[str, Any]:
        """Measures cold, single-row per-stage and end-to-end latency across sample rows."""
        logger.info(f"Benchmarking pipeline latency over {n_samples} individual rows...")
        latencies = []
        rows = df_sample.sample(n=min(n_samples, len(df_sample)), random_state=42).to_dict(orient="records")

        # Pre-populate history for realistic state
        for r in rows[:30]:
            self.process_row(r)

        t_batch_start = time.perf_counter()
        for r in rows:
            t0 = time.perf_counter()
            self.process_row(r)
            latencies.append((time.perf_counter() - t0) * 1e3)
        t_batch_end = time.perf_counter()

        arr = np.array(latencies)
        throughput = len(rows) / (t_batch_end - t_batch_start)

        return {
            "n_samples": len(rows),
            "mean_ms": round(float(np.mean(arr)), 3),
            "p50_ms": round(float(np.percentile(arr, 50)), 3),
            "p95_ms": round(float(np.percentile(arr, 95)), 3),
            "p99_ms": round(float(np.percentile(arr, 99)), 3),
            "min_ms": round(float(np.min(arr)), 3),
            "max_ms": round(float(np.max(arr)), 3),
            "throughput_rows_per_sec": round(float(throughput), 1),
            "meets_sla_500ms": bool(np.percentile(arr, 99) < 500.0),
        }
