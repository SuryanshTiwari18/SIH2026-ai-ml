import joblib

stats = joblib.load('models/mahalanobis_stats.joblib')
print("Keys in mahalanobis_stats:", stats.keys())
print("Sample station_hour keys:", list(stats['station_hour'].keys())[:5])
print("Sample zone_hour keys:", list(stats['zone_hour'].keys())[:5])
print("Sample entry in station_hour:", stats['station_hour'][list(stats['station_hour'].keys())[0]])

