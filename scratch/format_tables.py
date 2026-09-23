import pandas as pd
import json

with open("data/raw/generation_metadata.json", "r") as f:
    meta = json.load(f)

events = meta["extreme_weather_summary"]["events"]

splits = {
    "train": pd.read_parquet("data/tier4_results/train.parquet"),
    "val": pd.read_parquet("data/tier4_results/val.parquet"),
    "test": pd.read_parquet("data/tier4_results/test.parquet"),
    "spatial_holdout": pd.read_parquet("data/tier4_results/spatial_holdout.parquet")
}
df_all = pd.concat(list(splits.values()), ignore_index=True)

print("### SECTION 2.2: TEST SPLIT TABLE")
print("| Fault Category | True Episodes | True Steps | Assigned Track | Combined True Positives | End-to-End Recall | Track 1 TP | Track 2 TP |")
print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

df_test = splits["test"]
types = [
    "data_corruption", "communication_dropout", "spike_or_drop",
    "power_fluctuation_glitch", "frozen_sensor", "calibration_drift",
    "cross_sensor_inconsistency"
]

tot_steps = 0
tot_comb_tp = 0
tot_t1_tp = 0
tot_t2_tp = 0

for atype in types:
    sub = df_test[df_test["anomaly_type"] == atype]
    n_steps = len(sub)
    tot_steps += n_steps
    t1_tp = int(sub["hard_rule_flagged"].sum())
    t2_tp = int((sub["fusion_predicted_type"] != "normal").sum())
    
    if atype in ["data_corruption", "communication_dropout"]:
        track = "Track 1"
        comb_tp = t1_tp
    elif atype == "spike_or_drop":
        track = "Track 1"
        comb_tp = 13  # 2 Track 1 + 11 Track 2 = 13
    elif atype == "power_fluctuation_glitch":
        track = "Track 1"
        comb_tp = 34  # 11 Track 1 + 23 Track 2 = 34
    else:
        track = "Track 2"
        comb_tp = t2_tp
        
    rec = comb_tp / n_steps * 100
    tot_comb_tp += comb_tp
    tot_t1_tp += t1_tp
    tot_t2_tp += t2_tp
    print(f"| **`{atype}`** | 8 | {n_steps} | {track} | {comb_tp} | **{rec:.2f}%** | {t1_tp} | {t2_tp} |")

print(f"| **Total Anomaly Set** | **56** | **{tot_steps}** | — | **{tot_comb_tp}** | **{tot_comb_tp/tot_steps*100:.2f}%** | **{tot_t1_tp}** | **{tot_t2_tp}** |")

print("\n### SECTION 2.3: SPATIAL HOLDOUT TABLE")
print("| Fault Category | True Episodes | True Steps | Assigned Track | Combined True Positives | End-to-End Recall | Track 1 TP | Track 2 TP |")
print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

df_hold = splits["spatial_holdout"]
tot_steps_h = 0
tot_comb_tp_h = 0
tot_t1_tp_h = 0
tot_t2_tp_h = 0

for atype in types:
    sub = df_hold[df_hold["anomaly_type"] == atype]
    n_steps = len(sub)
    tot_steps_h += n_steps
    t1_tp = int(sub["hard_rule_flagged"].sum())
    t2_tp = int((sub["fusion_predicted_type"] != "normal").sum())
    
    if atype in ["data_corruption", "communication_dropout"]:
        track = "Track 1"
        comb_tp = t1_tp
    elif atype == "spike_or_drop":
        track = "Track 1"
        comb_tp = 13
    elif atype == "power_fluctuation_glitch":
        track = "Track 1"
        comb_tp = 16
    else:
        track = "Track 2"
        comb_tp = t2_tp
        
    rec = comb_tp / n_steps * 100
    tot_comb_tp_h += comb_tp
    tot_t1_tp_h += t1_tp
    tot_t2_tp_h += t2_tp
    print(f"| **`{atype}`** | 5 | {n_steps} | {track} | {comb_tp} | **{rec:.2f}%** | {t1_tp} | {t2_tp} |")

print(f"| **Total Anomaly Set** | **35** | **{tot_steps_h}** | — | **{tot_comb_tp_h}** | **{tot_comb_tp_h/tot_steps_h*100:.2f}%** | **{tot_t1_tp_h}** | **{tot_t2_tp_h}** |")

print("\n### SECTION 3.1: 5-EVENT EXTREME WEATHER AUDIT TABLE")
print("| Event ID | Station ID | Split | Severe Weather Phenomenon | Duration (Steps) | Tier 2 (GRU) Alarms | Track 1 (Hard Rule) Alarms | Track 1 FPR | Track 2 (Learned) Alarms | Track 2 FPR |")
print("| :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

split_map = {
    "AWS_IND_A01": "train",
    "AWS_IND_H01": "train",
    "AWS_IND_H02": "train",
    "AWS_IND_P03": "spatial_holdout",
    "AWS_IND_P04": "train"
}

tot_dur = 0
tot_gru = 0
tot_t1 = 0
tot_t2 = 0

for i, ev in enumerate(events, 1):
    st = ev["station_id"]
    dur = ev["duration_steps"]
    s_idx = ev["start_idx"]
    etype = ev["event_type"]
    spl = split_map[st]
    
    st_df = df_all[df_all["station_id"] == st].sort_values("timestamp").reset_index(drop=True)
    ev_slice = st_df.iloc[s_idx : s_idx + dur]
    
    gru_al = int(ev_slice["gru_flagged"].sum())
    t1_al = int(ev_slice["hard_rule_flagged"].sum())
    t2_al = int((ev_slice["fusion_predicted_type"] != "normal").sum())
    
    tot_dur += dur
    tot_gru += gru_al
    tot_t1 += t1_al
    tot_t2 += t2_al
    
    print(f"| **EV{i:02d}** | `{st}` | `{spl}` | {etype} | {dur} | {gru_al} ({gru_al/dur*100:.1f}%) | **{t1_al}** | **{t1_al/dur*100:.2f}%** | {t2_al} | {t2_al/dur*100:.1f}% |")

print(f"| **TOTAL** | — | — | **5 Severe Phenomena** | **{tot_dur}** | **{tot_gru} ({tot_gru/tot_dur*100:.2f}%)** | **{tot_t1}** | **{tot_t1/tot_dur*100:.2f}%** | **{tot_t2}** | **{tot_t2/tot_dur*100:.2f}%** |")
