import pandas as pd

for name in ['train', 'val', 'test', 'spatial_holdout_stations']:
    df = pd.read_parquet(f'data/splits/{name}.parquet')
    print(f'=== {name} ===')
    print(f'Shape: {df.shape}')
    print(f'Columns: {list(df.columns)}')
    print(f'Stations: {df["station_id"].nunique()}, Min time: {df["timestamp"].min()}, Max time: {df["timestamp"].max()}')
    print(f'is_anomaly distribution: {df["is_anomaly"].value_counts().to_dict()}')
    print(f'anomaly_type counts: {df["anomaly_type"].value_counts(dropna=False).head(10).to_dict()}')
    print(f'NaN counts: T={df["temperature_c"].isna().sum()}, P={df["pressure_hpa"].isna().sum()}, RH={df["humidity_pct"].isna().sum()}')
