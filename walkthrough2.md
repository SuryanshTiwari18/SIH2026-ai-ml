# SkyGuard AI: Feature Engineering Layer (`src/features.py`) Walkthrough

**Smart India Hackathon 2026 | Problem Statement 26073**  
*AI/ML-Based Intelligent Anomaly Detection for Automatic Weather Stations (IMD/MoES)*  
*Theme: Disaster Management*

---

## 1. Executive Summary

We have resolved the sentinel leakage defect in [`src/features.py`](file:///d:/SIH/src/features.py) by enforcing **pre-computation exclusion**, **true temporal gap segmentation**, and **lookback context quarantine**.

### The Fixed Defect:
- In the initial implementation, Tier 1 exclusions (`data_corruption` sentinels `-999.0` and `communication_dropout` `NaN`s) were excluded *after* derivative/rolling features were computed.
- At `AWS_IND_H04` on `2026-06-13 02:20:00`, $\Delta T$ was evaluated as $15.217 - (-999.0) = \mathbf{1014.217^\circ\text{C}}$, leaking sentinel values into the temporal derivatives and inflating the fitted `StandardScaler` standard deviation by **13.4x** on $\Delta T$ and **6663.0x** on $\sigma^2_{1\text{h}}(T)$.
- In addition, spatial buddy check reference telemetry was retaining $-999.0$ sentinels, causing nearest-neighbor $\Delta T_{\text{buddy}}$ to spike up to $1029^\circ\text{C}$ on normal observations.

### The Complete Resolution:
1. **Pre-Computation Tier 1 Segregation**: Sentinels (`-999.0`, `9999.0`) and dropouts are quarantined into `data/features/tier1_excluded_rows.parquet` **prior** to temporal segmentation and buddy joins.
2. **True Temporal Gap Segmentation**: Any interval $> 10\text{ minutes}$ (or station boundary) marks a hard segment boundary. First differences and rolling windows **never** reach across segment boundaries.
3. **Full-Window `min_periods` & Edge Routing**: Full window lookback is strictly required ($W=6$ for $1\text{h}$, $W=36$ for $6\text{h}$, $W=144$ for $24\text{h}$). Edge rows lacking sufficient historical lookback are routed to `data/features/gap_edge_excluded_rows.parquet`.
4. **`near_excluded_gap`**: Boolean flag indicates observations within $\le 1\text{h}$ of ANY exclusion boundary (dropout or corruption).
5. **Fresh Rescaling**: `StandardScaler` was completely refitted on normal train retained rows ($N = 67,097$) and saved to [`models/feature_scaler.joblib`](file:///d:/SIH/models/feature_scaler.joblib).

---

## 2. Before vs After Verification Metrics

All numbers were empirically confirmed via [`scratch/self_verify_fix.py`](file:///d:/SIH/scratch/self_verify_fix.py):

### A. Standard Deviation Restored to Physical Baseline
| Feature | Before Fix (Poisoned) | After Fix (Clean) | Reduction Factor | Target Range | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`delta_T`** | $3.8140^\circ\text{C}$ | **$0.2850^\circ\text{C}$** | **$13.4\times$ reduction** | $\sim 0.3^\circ\text{C}$ | **PASS** |
| **`var_T_1h`** | $2135.8060^\circ\text{C}^2$ | **$0.3205^\circ\text{C}^2$** | **$6663.0\times$ reduction** | $\sim 0.3 - 0.5^\circ\text{C}^2$ | **PASS** |
| **`delta_T_buddy` Max** | $1029.049^\circ\text{C}$ | **$21.442^\circ\text{C}$** | Clean reference join | $< 30^\circ\text{C}$ | **PASS** |

### B. Normal Row $\max(|\Delta T|)$ Audit
- `train`: $\mathbf{6.7331^\circ\text{C}}$ (`AWS_IND_A04` @ `2026-06-10 22:10:00`)
- `test`: $\mathbf{7.0200^\circ\text{C}}$ (`AWS_IND_C01` @ `2026-07-26 06:00:00`)
- `val`: $\mathbf{21.9624^\circ\text{C}}$ (`AWS_IND_A02` @ `2026-07-16 08:20:00`)
  - **Investigation**: Row immediately followed a severe Tier 2 `spike_or_drop` episode ($T$ was forced to $10.75^\circ\text{C}$ at $08:10$). When the sensor resumed normal telemetry at $08:20$ ($T = 32.71^\circ\text{C}$), a legitimate physical recovery step of $+21.96^\circ\text{C}$ occurred. Verified: **NOT a sentinel leak**.
- `spatial_holdout`: $\mathbf{16.9935^\circ\text{C}}$ (`AWS_IND_P03` @ `2026-07-24 20:30:00`)
  - **Investigation**: Recovery edge row following a $13.32^\circ\text{C}$ `spike_or_drop` episode. Verified: **NOT a sentinel leak**.

### C. Byte-for-Byte Exact Parity on Unaffected Features
Max absolute difference between previous and current run on common rows:
- `mahalanobis_dist`: **`0.00e+00`**
- `dew_point_dep`: **`0.00e+00`**
- `vpd`: **`0.00e+00`**
- `sin_hour`: **`0.00e+00`**
- `cos_hour`: **`0.00e+00`**

### D. Row Count Balance Audit
Every raw row is strictly preserved across the three mutually exclusive sets:
$$\text{Raw Count} = \text{Tier 1 Excluded} + \text{Gap Edge Excluded} + \text{Retained Feature Store}$$

| Split Name | Raw Records | Tier 1 Excluded | Gap Edge Excluded | Retained (Final Scaled) | Check ($\Sigma$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`train`** | 72,576 | 111 | 3,930 | **68,535** | $72,576$ (**Match**) |
| **`val`** | 15,552 | 128 | 3,674 | **11,750** | $15,552$ (**Match**) |
| **`test`** | 15,552 | 154 | 3,696 | **11,702** | $15,552$ (**Match**) |
| **`spatial_holdout`** | 34,560 | 80 | 1,967 | **32,513** | $34,560$ (**Match**) |
| **TOTAL** | **138,240** | **473** | **13,267** | **124,500** | **138,240** (**Match**) |

---

## 3. Directory Layout & Artifacts

```
d:/SIH/
├── data/
│   ├── splits/                               # Raw canonical splits (4 splits)
│   └── features/
│       ├── train.parquet                     # 68,535 rows, 49 cols (Cleaned, Scaled)
│       ├── val.parquet                       # 11,750 rows, 49 cols (Cleaned, Scaled)
│       ├── test.parquet                      # 11,702 rows, 49 cols (Cleaned, Scaled)
│       ├── spatial_holdout.parquet           # 32,513 rows, 49 cols (Cleaned, Scaled)
│       ├── tier1_excluded_rows.parquet       # 473 quarantined corrupted/dropout rows
│       └── gap_edge_excluded_rows.parquet    # 13,267 quarantined lookback-deficient edge rows
├── models/
│   ├── feature_scaler.joblib                 # REGENERATED StandardScaler (uncontaminated)
│   ├── mahalanobis_stats.joblib              # Unchanged 288 (station, hour) centroid/covariance models
│   └── spatial_neighbors.json                # Unchanged Haversine 3-NN topology
├── docs/
│   ├── EDA_INSIGHTS.md                       # Comprehensive handoff insights
│   └── FEATURE_DICTIONARY.md                 # Updated with verified clean ranges & stds
└── src/
    └── features.py                           # Updated with gap isolation & lookback quarantine
```
