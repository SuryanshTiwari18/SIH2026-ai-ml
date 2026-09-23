import joblib
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Verify GRUAutoencoder class loading
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

pt_data = torch.load('models/gru_autoencoder.pt', map_location='cpu')
print("Keys in gru_autoencoder.pt:", pt_data.keys())
print("Arch:", pt_data['arch'])

model = GRUAutoencoder(input_dim=12, hidden_dim=32, latent_dim=16, seq_len=72)
model.load_state_dict(pt_data['state_dict'])
model.eval()
print("Model loaded and initialized successfully!")

# Test dummy input
dummy = torch.randn(1, 72, 12)
with torch.no_grad():
    out = model(dummy)
print("Output shape:", out.shape)
mse = torch.mean((dummy - out) ** 2).item()
print("Reconstruction MSE:", mse)
