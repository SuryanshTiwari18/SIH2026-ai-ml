import time
import joblib
import json
import numpy as np
import pandas as pd
import torch

# Load all artifacts
print("Loading all artifacts...")
feature_scaler = joblib.load('models/feature_scaler.joblib')
iso_forest = joblib.load('models/isolation_forest.joblib')
gru_ae = torch.jit.load('models/gru_autoencoder.pt') if False else torch.load('models/gru_autoencoder.pt', map_location='cpu')
mahal_stats = joblib.load('models/mahalanobis_stats.joblib')
neighbors = json.load(open('models/spatial_neighbors.json'))
tier4_clf_dict = joblib.load('models/tier4_fusion_classifier.joblib')
tier4_clf = tier4_clf_dict['model']
classes = tier4_clf_dict['classes']

print("Artifacts loaded successfully!")
