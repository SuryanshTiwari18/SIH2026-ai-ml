import re
from pathlib import Path

# Load all source docs into a single searchable text database
source_files = [
    "docs/METEOROLOGICAL_MODELING.md",
    "docs/DATA_DICTIONARY.md",
    "docs/EDA_INSIGHTS.md",
    "docs/FEATURE_DICTIONARY.md",
    "docs/TIER1_EVALUATION.md",
    "docs/TIER2_EVALUATION.md",
    "docs/TIER3_EVALUATION.md",
    "docs/TIER4_EVALUATION.md",
    "docs/TIER5_EVALUATION.md",
    "docs/FINAL_EVALUATION.md",
    "docs/PIPELINE_USAGE.md",
    "data/raw/generation_metadata.json"
]

sources_text = {}
for sf in source_files:
    p = Path(sf)
    if p.exists():
        sources_text[sf] = p.read_text(encoding="utf-8")
    else:
        print(f"Warning: {sf} not found")

# Key numbers to verify from README.md and WALKTHROUGH.md
numbers_to_check = [
    # Canonical counts
    ("138,240", "Total telemetry rows across 16 stations over 60 days"),
    ("16", "Total AWS stations"),
    ("60.0", "Total duration in days"),
    ("8640", "Total timesteps per station"),
    ("6,501", "Total anomalous readings (4.70%)"),
    ("131,739", "Normal telemetry readings (95.30%)"),
    ("413", "Communication dropout steps (100% NaN)"),
    ("60", "Data corruption steps"),
    ("62", "Spike or drop steps"),
    ("218", "Power fluctuation glitch steps"),
    ("999", "Frozen sensor steps"),
    ("4,207", "Calibration drift steps"),
    ("542", "Cross-sensor inconsistency steps"),
    
    # Severe weather events
    ("1,378", "Total extreme weather steps across 5 events"),
    ("445", "EV01 A01 heatwave duration"),
    ("77", "EV02 H01 squall duration"),
    ("138", "EV03 H02 inversion fog duration"),
    ("584", "EV04 P03 holdout heatwave duration"),
    ("134", "EV05 P04 inversion fog duration"),
    
    # Baselines from EDA
    ("0.017", "Global Z-score baseline F1"),
    ("0.084", "Station Z-score baseline F1"),
    ("0.175", "Station IQR baseline F1"),
    ("-0.68", "Normal T-RH correlation"),
    ("-0.04", "Anomalous T-RH correlation"),
    ("1.60", "Normal Mahalanobis mean D_M"),
    ("7.65", "Cross-sensor Mahalanobis mean D_M"),
    ("6.06", "Normal buddy delta T"),
    ("36.17", "Fault buddy delta T"),
    
    # Features
    ("69,925", "Lookback-complete normal train rows for scaler"),
    ("3,393", "Gap edge excluded rows"),
    
    # Tier 1
    ("100.0000%", "Tier 1 recall"),
    ("0.000000%", "Tier 1 normal FPR"),
    
    # Tier 2
    ("49.35%", "Tier 2 GRU H01 squall FPR (38/77)"),
    ("4.93%", "Tier 2 extreme weather FPR (68/1378)"),
    ("12.21", "Squall pressure drop hPa"),
    ("9.37", "Squall temperature drop C"),
    ("-6.91", "Squall peak delta P per 10-min"),
    ("-5.64", "Squall peak delta T per 10-min"),
    
    # Tier 3
    ("6.50", "Tier 3 Mahalanobis threshold"),
    ("52.81%", "Previous holdout normal FPR before fix"),
    ("17.27%", "New holdout normal FPR after climate fallback"),
    ("29.79", "Puri D_M under 1-NN"),
    ("239.68", "Shillong D_M under 1-NN"),
    ("65.51%", "Remaining Shillong H03 FPR"),
    ("92.11%", "H01 squall suppression rate (35/38)"),
    
    # Tier 4
    ("27.72%", "Initial ungated Hard Rule extreme weather FPR"),
    ("39.55%", "Learned Classifier extreme weather FPR"),
    ("0.22%", "Final Gated Hard Rule extreme weather FPR (3/1378)"),
    
    # Tier 5
    ("5.5 hours", "Calibration drift predictive lead time"),
    ("90.04%", "Minimum SHI during severe weather"),
    ("13.39", "Corrupted anomaly deviation from 3-NN buddy median"),
    
    # Final Rollup & Latency
    ("15,552", "Test split canonical rows"),
    ("34,560", "Spatial holdout canonical rows"),
    ("1,782", "Test split anomalous steps"),
    ("13,770", "Test split normal steps"),
    ("1,135", "Spatial holdout anomalous steps"),
    ("33,425", "Spatial holdout normal steps"),
    ("76.99%", "Test split combined anomaly recall"),
    ("57.00%", "Spatial holdout combined anomaly recall"),
    ("0.21%", "Test split Track 1 normal FPR"),
    ("1.91%", "Test split Track 2 normal FPR"),
    ("2.12%", "Test split combined normal FPR"),
    ("0.01%", "Spatial holdout Track 1 normal FPR"),
    ("21.49%", "Spatial holdout Track 2 normal FPR"),
    ("8.99 ms", "Mean latency"),
    ("0.09 ms", "Median P50 latency"),
    ("28.88 ms", "P95 latency"),
    ("31.74 ms", "P99 latency"),
    ("111.3", "Throughput rows/sec")
]

print(f"Tracing {len(numbers_to_check)} key numerical claims to source docs...")
untraced = []

for num_str, desc in numbers_to_check:
    found_in = []
    for sf, text in sources_text.items():
        if num_str in text:
            found_in.append(sf)
    if found_in:
        print(f"  [PASS] '{num_str}' ({desc}) -> found in {len(found_in)} source doc(s): {found_in[0]}")
    else:
        print(f"  [FAIL] '{num_str}' ({desc}) -> NOT FOUND IN ANY SOURCE DOC")
        untraced.append((num_str, desc))

print("\n" + "=" * 80)
if not untraced:
    print("ALL NUMERICAL CLAIMS TRACED 100% CLEANLY TO SOURCE DOCS!")
else:
    print(f"FAILED TO TRACE {len(untraced)} CLAIMS:")
    for num_str, desc in untraced:
        print(f"  - {num_str}: {desc}")
print("=" * 80)
