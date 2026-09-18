header = "Fault Type                     Affected Steps     % of Anomalies     Primary Signature"

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

test_counts = {
    'calibration_drift': 4207,
    'frozen_sensor': 999,
    'cross_sensor_inconsistency': 542,
    'communication_dropout': 413,
    'power_fluctuation_glitch': 218,
    'spike_or_drop': 62,
    'data_corruption': 60
}

total = sum(test_counts.values())

print(header)
print("-" * 92)
for t, count in sorted(test_counts.items(), key=lambda x: x[1], reverse=True):
    t_name = FAULT_DISPLAY_NAMES.get(t, t.replace('_', ' ').title())
    pct = (count / total) * 100.0 if total > 0 else 0.0
    steps_str = f"{count:>5,} steps"
    pct_str = f"{pct:.1f}%"
    sig = PRIMARY_SIGNATURES.get(t, 'Station-level sensor anomaly')
    row = f"{t_name:<31}{steps_str:<19}{pct_str:<19}{sig}"
    print(row)
