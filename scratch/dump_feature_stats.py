import pandas as pd
from src.features import ENGINEERED_FEATURE_COLS

df = pd.read_parquet('data/features/train.parquet')
print("Train rows:", len(df))
for col in ENGINEERED_FEATURE_COLS:
    r = df[col]
    s = df[f'{col}_scaled']
    print(f"| `{col}` | `{col}_scaled` | `[{r.min():.3f}, {r.max():.3f}]` | `{r.mean():.3f}` | `{r.std():.3f}` | `[{s.min():.3f}, {s.max():.3f}]` |")
