import pandas as pd

df_train = pd.read_parquet('data/tier4_results/train.parquet')
sub = df_train[df_train['timestamp'] == '2026-07-10 20:00:00']
print(sub[['station_id', 'temperature_c', 'pressure_hpa', 'humidity_pct', 'isolated_deviation']].head(10).to_string())
