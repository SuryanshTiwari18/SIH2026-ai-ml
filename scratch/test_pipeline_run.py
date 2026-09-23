import sys
sys.path.insert(0, '.')
import pandas as pd
from src.skyguard_pipeline import SkyGuardPipeline


print("Instantiating SkyGuardPipeline...")
pipeline = SkyGuardPipeline(models_dir="models")

df_test = pd.read_parquet('data/tier4_results/test.parquet')

# Test single row
row = df_test.iloc[100].to_dict()
res = pipeline.process_row(row)
print("\nSample processed row output:")
for k, v in res.items():
    print(f"  {k}: {v}")

# Benchmark latency over 500 rows
bench = pipeline.benchmark_latency(df_test, n_samples=500)
print("\nLatency Benchmark (500 rows):")
for k, v in bench.items():
    print(f"  {k}: {v}")
