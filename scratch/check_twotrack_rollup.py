import pandas as pd
import numpy as np

# Load tier4 / tier5 results
df_test = pd.read_parquet('data/tier5_results/test.parquet')
df_hold = pd.read_parquet('data/tier5_results/spatial_holdout.parquet')

def evaluate_combined_twotrack(df, split_name):
    # Track 1: hard_rule_flagged
    t1_flag = df['hard_rule_flagged'] == True
    
    # What if Track 2 secondary queue includes all maintenance predictions?
    t2_assigned_types = ['frozen_sensor', 'calibration_drift', 'cross_sensor_inconsistency', 'power_fluctuation_glitch', 'spike_or_drop']
    t2_flag = (~t1_flag) & (df['fusion_flagged'] == True) & (df['fusion_predicted_type'].isin(t2_assigned_types))

    
    # What if moderate glitch is also routed to Track 2?
    # Let's test both: strict 3 types vs including glitch
    combined_flag = t1_flag | t2_flag
    
    # Decision assignment
    combined_type = pd.Series('normal', index=df.index)
    which_track = pd.Series(0, index=df.index)
    
    # Where Track 1 fired:
    combined_type[t1_flag] = df.loc[t1_flag, 'fusion_predicted_type']
    which_track[t1_flag] = 1
    
    # Where Track 2 fired:
    combined_type[t2_flag] = df.loc[t2_flag, 'fusion_predicted_type']
    which_track[t2_flag] = 2
    
    # Insufficient context
    insuf = df['tier_coverage'] == 'tier1_only'
    unflagged_insuf = insuf & (~df['tier1_flagged'])
    combined_type[unflagged_insuf] = 'insufficient_context'
    which_track[unflagged_insuf] = -1
    
    print(f"\n=================== {split_name.upper()} COMBINED TWO-TRACK EVALUATION ===================")
    
    # 7 anomaly types recall
    anom_types = [
        'data_corruption',
        'communication_dropout',
        'spike_or_drop',
        'power_fluctuation_glitch',
        'frozen_sensor',
        'calibration_drift',
        'cross_sensor_inconsistency'
    ]
    
    for at in anom_types:
        sub = df[df['anomaly_type'] == at]
        n_tot = len(sub)
        n_det = int((combined_flag & (df['anomaly_type'] == at)).sum())
        rec = n_det / n_tot * 100 if n_tot > 0 else 0.0
        # What track caught it?
        t1_caught = int((t1_flag & (df['anomaly_type'] == at)).sum())
        t2_caught = int((t2_flag & (df['anomaly_type'] == at)).sum())
        print(f"  {at:28s}: {n_det:4d} / {n_tot:4d} ({rec:6.2f}%) [Track 1: {t1_caught}, Track 2: {t2_caught}]")
        
    # Normal FPR
    norm_sub = df[df['anomaly_type'].isna()]
    n_norm = len(norm_sub)
    n_norm_fp = int((combined_flag & df['anomaly_type'].isna()).sum())
    norm_fpr = n_norm_fp / n_norm * 100 if n_norm > 0 else 0.0
    t1_norm_fp = int((t1_flag & df['anomaly_type'].isna()).sum())
    t2_norm_fp = int((t2_flag & df['anomaly_type'].isna()).sum())
    print(f"  Normal FPR: {n_norm_fp:4d} / {n_norm:4d} ({norm_fpr:6.2f}%) [Track 1 FP: {t1_norm_fp} ({t1_norm_fp/n_norm*100:.2f}%), Track 2 FP: {t2_norm_fp} ({t2_norm_fp/n_norm*100:.2f}%)]")

evaluate_combined_twotrack(df_test, 'test')
evaluate_combined_twotrack(df_hold, 'spatial_holdout')
