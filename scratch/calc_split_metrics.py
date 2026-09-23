import pandas as pd
import numpy as np

def analyze_split(parquet_path, split_name):
    df = pd.read_parquet(parquet_path)
    print(f"\n==================== {split_name.upper()} ====================")
    print(f"Total Rows: {len(df)}")
    
    # Check normal vs anomalous
    normal_mask = df["anomaly_type"] == "normal"
    anom_mask = ~normal_mask
    print(f"Normal Rows: {normal_mask.sum()}")
    print(f"Anomalous Rows: {anom_mask.sum()}")
    assert normal_mask.sum() + anom_mask.sum() == len(df)
    
    # Two-Track routing definition:
    # Track 1 assigned types: data_corruption, communication_dropout, spike_or_drop, power_fluctuation_glitch
    # Track 2 assigned types: frozen_sensor, calibration_drift, cross_sensor_inconsistency
    track1_types = ["data_corruption", "communication_dropout", "spike_or_drop", "power_fluctuation_glitch"]
    track2_types = ["frozen_sensor", "calibration_drift", "cross_sensor_inconsistency"]
    
    all_types = [
        "data_corruption",
        "communication_dropout",
        "spike_or_drop",
        "power_fluctuation_glitch",
        "frozen_sensor",
        "calibration_drift",
        "cross_sensor_inconsistency"
    ]
    
    print("\n--- PER FAULT TYPE ROLLUP ---")
    sum_true_steps = 0
    sum_episodes = 0
    sum_combined_tp = 0
    sum_t1_tp = 0
    sum_t2_tp = 0
    
    rows_summary = []
    
    for atype in all_types:
        sub = df[df["anomaly_type"] == atype]
        n_steps = len(sub)
        sum_true_steps += n_steps
        
        # Episode count: contiguous blocks per station
        # In generation_metadata or contiguous blocks
        # Let's count episodes by looking at contiguous sequences or station changes
        episodes = 0
        for st, st_df in sub.groupby("station_id"):
            # find breaks in timestamps
            st_df = st_df.sort_values("timestamp")
            # diff in 10 minutes
            diffs = (st_df["timestamp"].diff().dt.total_seconds() > 600) | (st_df["timestamp"].diff().isnull())
            episodes += diffs.sum()
            
        sum_episodes += episodes
        assigned_track = "Track 1" if atype in track1_types else "Track 2"
        
        # Track 1 TP: hard_rule_flagged == True
        t1_tp = int(sub["hard_rule_flagged"].sum())
        
        # Track 2 TP: fusion_flagged == True (or fusion_predicted_type != 'normal')
        t2_tp = int((sub["fusion_predicted_type"] != "normal").sum())
        
        # Combined TP:
        # A row is flagged by the pipeline if:
        # either Track 1 flags it OR Track 2 flags it
        # BUT what is the two-track routing logic?
        # "Using the ACTUAL Two-Track routing logic from Tier 4's final recommendation (Track 1 = 
        # Gated Hard Rule for its 4 assigned types, Track 2 = Learned Classifier for its 3 assigned 
        # types, per-type as specified in Tier 4's final routing table), compute the true combined 
        # end-to-end recall per fault type, on test AND spatial_holdout:
        # This is each type scored by whichever track it's actually assigned to!"
        # Let's check both: scored by assigned track, and scored by either track!
        tp_assigned = t1_tp if atype in track1_types else t2_tp
        tp_either = int((sub["hard_rule_flagged"] | (sub["fusion_predicted_type"] != "normal")).sum())
        
        # Let's check how many were caught:
        recall_assigned = tp_assigned / n_steps * 100 if n_steps > 0 else 0.0
        recall_either = tp_either / n_steps * 100 if n_steps > 0 else 0.0
        
        sum_combined_tp += tp_assigned
        sum_t1_tp += t1_tp
        sum_t2_tp += t2_tp
        
        rows_summary.append({
            "anomaly_type": atype,
            "assigned_track": assigned_track,
            "episodes": episodes,
            "true_steps": n_steps,
            "t1_tp": t1_tp,
            "t2_tp": t2_tp,
            "tp_assigned": tp_assigned,
            "recall_assigned": recall_assigned,
            "tp_either": tp_either,
            "recall_either": recall_either,
        })
        
        print(f"{atype:28s} | {assigned_track} | Ep: {episodes:2d} | Steps: {n_steps:4d} | T1 TP: {t1_tp:4d} | T2 TP: {t2_tp:4d} | Assigned TP: {tp_assigned:4d} ({recall_assigned:6.2f}%) | Either TP: {tp_either:4d} ({recall_either:6.2f}%)")

    print("-" * 80)
    print(f"TOTAL ANOMALOUS STEPS: {sum_true_steps} (matches anomalous count: {sum_true_steps == anom_mask.sum()})")
    print(f"TOTAL EPISODES: {sum_episodes}")
    print(f"TOTAL ASSIGNED TP: {sum_combined_tp} ({sum_combined_tp / sum_true_steps * 100:.2f}%)")
    
    # Check normal FPR
    df_normal = df[df["anomaly_type"] == "normal"]
    n_norm = len(df_normal)
    t1_norm_fp = int(df_normal["hard_rule_flagged"].sum())
    t2_norm_fp = int((df_normal["fusion_predicted_type"] != "normal").sum())
    either_norm_fp = int((df_normal["hard_rule_flagged"] | (df_normal["fusion_predicted_type"] != "normal")).sum())
    
    print("\n--- NORMAL FALSE POSITIVE RATES ---")
    print(f"Total Normal Steps: {n_norm}")
    print(f"Track 1 Normal FP:  {t1_norm_fp} ({t1_norm_fp / n_norm * 100:.4f}%)")
    print(f"Track 2 Normal FP:  {t2_norm_fp} ({t2_norm_fp / n_norm * 100:.4f}%)")
    print(f"Either Track FP:    {either_norm_fp} ({either_norm_fp / n_norm * 100:.4f}%)")
    
    return rows_summary

print("Analyzing Test and Spatial Holdout splits...")
res_test = analyze_split("data/tier4_results/test.parquet", "test")
res_hold = analyze_split("data/tier4_results/spatial_holdout.parquet", "spatial_holdout")
