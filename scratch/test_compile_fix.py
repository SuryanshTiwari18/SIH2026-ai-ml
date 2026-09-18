import pandas as pd

df = pd.read_parquet("data/raw/aws_telemetry_master.parquet")

PRIMARY_SIGNATURES = {
    'calibration_drift': 'Subtle linear slope drift',
    'frozen_sensor': 'Consecutive zero variance flatline',
    'spike_or_drop': 'Sudden 1st-derivative step jump',
    'cross_sensor_inconsistency': 'Thermodynamic & Mahalanobis outlier',
    'communication_dropout': 'Concurrent NaN across all 3 sensors',
    'power_fluctuation_glitch': 'Local high-frequency jitter variance',
    'data_corruption': 'WMO out-of-range sentinel values'
}

FAULT_DISPLAY_NAMES = {
    'calibration_drift': 'Calibration Drift',
    'frozen_sensor': 'Frozen Sensor',
    'spike_or_drop': 'Spike or Drop',
    'cross_sensor_inconsistency': 'Cross-Sensor Inconsistency',
    'communication_dropout': 'Communication Dropout',
    'power_fluctuation_glitch': 'Power Fluctuation Glitch',
    'data_corruption': 'Data Corruption'
}

type_counts = df['anomaly_type'].value_counts(dropna=True).to_dict()
total_anomalies = df['is_anomaly'].sum()

print(f"Total anomalies: {total_anomalies}")
print("```")
print(f"{'Fault Type':<31}{'Affected Steps':<19}{'% of Anomalies':<19}Primary Signature")
print("-" * 92)
sorted_faults = sorted(type_counts.items(), key=lambda item: item[1], reverse=True)
for atype, count in sorted_faults:
    t_name = FAULT_DISPLAY_NAMES.get(atype, atype.replace('_', ' ').title())
    pct = (count / total_anomalies) * 100.0 if total_anomalies > 0 else 0.0
    steps_str = f"{count:>5,} steps"
    pct_str = f"{pct:.1f}%"
    sig = PRIMARY_SIGNATURES.get(atype, 'Station-level sensor anomaly')
    print(f"{t_name:<31}{steps_str:<19}{pct_str:<19}{sig}")
print("```")
