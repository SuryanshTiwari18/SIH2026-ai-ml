import json

with open("data/real_climatology/station_month_climatology.json") as f:
    clim = json.load(f)

cities = ["delhi", "leh", "jaisalmer", "mumbai", "chennai", "guwahati", "bengaluru"]

print(f"{'City':12s} | {'Jan Mean [Min..Max]':25s} | {'May Mean [Min..Max]':25s} | {'Jul Mean [Min..Max]':25s}")
print("-" * 95)
for city in cities:
    m1 = clim[city]["1"]
    m5 = clim[city]["5"]
    m7 = clim[city]["7"]
    jan_str = f"{m1['temp_mean']}°C [{m1['temp_min']}..{m1['temp_max']}]"
    may_str = f"{m5['temp_mean']}°C [{m5['temp_min']}..{m5['temp_max']}]"
    jul_str = f"{m7['temp_mean']}°C [{m7['temp_min']}..{m7['temp_max']}]"
    print(f"{city.capitalize():12s} | {jan_str:25s} | {may_str:25s} | {jul_str:25s}")
