import time
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

# Define GRUAutoencoder
class GRUAutoencoder(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=32, latent_dim=16, seq_len=72):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.encoder_gru = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.enc_dense = nn.Linear(hidden_dim, latent_dim)
        self.dec_dense = nn.Linear(latent_dim, hidden_dim)
        self.decoder_gru = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.out_dense = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.encoder_gru(x)
        latent = self.enc_dense(h_n.squeeze(0))
        dec_init = self.dec_dense(latent).unsqueeze(1)
        dec_input = dec_init.repeat(1, self.seq_len, 1)
        dec_out, _ = self.decoder_gru(dec_input)
        recon = self.out_dense(dec_out)
        return recon

print("Testing latency of individual pipeline stages on 1 row...")

# Load artifacts
scaler = joblib.load('models/feature_scaler.joblib')
iso = joblib.load('models/isolation_forest.joblib')
gru_pt = torch.load('models/gru_autoencoder.pt', map_location='cpu')
gru = GRUAutoencoder(input_dim=12, hidden_dim=32, latent_dim=16, seq_len=72)
gru.load_state_dict(gru_pt['state_dict'])
gru.eval()
lgbm_dict = joblib.load('models/tier4_fusion_classifier.joblib')
lgbm = lgbm_dict['model']

# Dummy inputs
dummy_feat19 = np.random.randn(1, 19).astype(np.float32)
dummy_feat12 = dummy_feat19[:, :12]
dummy_seq = torch.randn(1, 72, 12)
dummy_tier4_X = pd.DataFrame([{
    'tier1_flagged': 0.0,
    'if_score': 0.45,
    'gru_score': 0.8,
    'mahalanobis_dist': 2.1,
    'buddy_flagged': 0.0,
    'isolated_deviation': 0.0
}])

# Measure Tier 1 (physical rules)
t0 = time.perf_counter()
for _ in range(1000):
    val = -999.0
    is_t1 = (val == -999.0)
t1 = time.perf_counter()
print(f"Tier 1 latency: {(t1 - t0)/1000*1e6:.2f} microseconds / row")

# Measure Feature Scaling
t0 = time.perf_counter()
for _ in range(1000):
    s = scaler.transform(dummy_feat19)
t1 = time.perf_counter()
print(f"Feature scaling latency: {(t1 - t0)/1000*1e3:.3f} ms / row")

# Measure Isolation Forest
t0 = time.perf_counter()
for _ in range(100):
    sc = -iso.score_samples(dummy_feat12)
t1 = time.perf_counter()
print(f"Tier 2 Isolation Forest latency: {(t1 - t0)/100*1e3:.3f} ms / row")

# Measure GRU-AE
t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(100):
        out = gru(dummy_seq)
        err = torch.mean((dummy_seq - out) ** 2).item()
t1 = time.perf_counter()
print(f"Tier 2 GRU-AE latency: {(t1 - t0)/100*1e3:.3f} ms / row")

# Measure LightGBM predict
t0 = time.perf_counter()
for _ in range(100):
    p = lgbm.predict_proba(dummy_tier4_X)
t1 = time.perf_counter()
print(f"Tier 4 LightGBM inference latency: {(t1 - t0)/100*1e3:.3f} ms / row")

# Measure TreeSHAP
t0 = time.perf_counter()
for _ in range(100):
    sh = lgbm.booster_.predict(dummy_tier4_X, pred_contrib=True)
t1 = time.perf_counter()
print(f"Tier 5 TreeSHAP latency: {(t1 - t0)/100*1e3:.3f} ms / row")
