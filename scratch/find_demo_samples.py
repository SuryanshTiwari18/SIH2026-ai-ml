import json
import pandas as pd
from pathlib import Path

# Load test and train
df_test = pd.read_parquet('data/tier4_results/test.parquet')
df_train = pd.read_parquet('data/tier4_results/train.parquet')

categories = [
    ('Normal Baseline', df_test[df_test['anomaly_type'].isna()].iloc[100]),
    ('Data Corruption', df_test[df_test['anomaly_type'] == 'data_corruption'].iloc[0]),
    ('Communication Dropout', df_test[df_test['anomaly_type'] == 'communication_dropout'].iloc[0]),
    ('Spike or Drop', df_test[df_test['anomaly_type'] == 'spike_or_drop'].iloc[0]),
    ('Frozen Sensor', df_test[df_test['anomaly_type'] == 'frozen_sensor'].iloc[50]),
    ('Power Fluctuation Glitch', df_test[df_test['anomaly_type'] == 'power_fluctuation_glitch'].iloc[10]),
    ('Calibration Drift', df_test[df_test['anomaly_type'] == 'calibration_drift'].iloc[100]),
    ('Cross-Sensor Inconsistency', df_test[df_test['anomaly_type'] == 'cross_sensor_inconsistency'].iloc[50]),
    ('Genuine Extreme Weather (H01 Squall)', df_train[(df_train['station_id'] == 'AWS_IND_H01') & (df_train['timestamp'] == '2026-07-10 20:00:00')].iloc[0]),
]

print("Found sample rows:")
for cat_name, row in categories:
    print(f"  {cat_name:36s}: station={row['station_id']}, ts={row['timestamp']}, T={row['temperature_c']}, P={row['pressure_hpa']}, RH={row['humidity_pct']}")
