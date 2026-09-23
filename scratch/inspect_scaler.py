import joblib
iso = joblib.load('models/isolation_forest.joblib')
print("Isolation forest n_features_in_:", iso.n_features_in_)

print("Isolation forest feature_names_in_:", getattr(iso, 'feature_names_in_', None))

