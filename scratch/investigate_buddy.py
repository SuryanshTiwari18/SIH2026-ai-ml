import pandas as pd

for name in ['train', 'val', 'test', 'spatial_holdout']:
    df = pd.read_parquet(f'data/features/{name}.parquet')
    bad = df[df['delta_T_buddy'] > 50]
    bad_p = df[df['delta_P_cluster'].abs() > 500]
    print(f"[{name}] delta_T_buddy > 50: {len(bad)}, delta_P_cluster > 500: {len(bad_p)}")
