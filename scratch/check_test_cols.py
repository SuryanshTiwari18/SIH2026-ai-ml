import pandas as pd
df = pd.read_parquet('data/tier4_results/test.parquet')
print('Value counts of anomaly_type in test.parquet:')
print(df['anomaly_type'].value_counts(dropna=False))
print('Value counts of is_anomaly in test.parquet:')
print(df['is_anomaly'].value_counts(dropna=False))
