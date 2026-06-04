"""Quick diagnostic: why did 19 stations fail satellite extraction?"""
import pandas as pd

locs = pd.read_csv("data/cpcb_station_locations.csv")
wq = pd.read_csv("data/rtwqms_historical/rtwqms_daily_wide.csv")

missing = [11785,11786,11787,11788,11789,11804,11805,11806,11807,11808,
           11809,11810,11811,11812,11813,11819,11820,11821,11822]

present = [11783,11784,11790,11791,11792,11793,11794,11795,11796,11797,
           11798,11799,11800,11801,11802,11803,11814,11815,11816,11817,11818]

print("=== PRESENT stations summary ===")
p = locs[locs["station_id"].isin(present)]
lat_col = "latitude"
lon_col = "longitude"
print(f"  Count: {len(p)}")
print(f"  Lat range: {p[lat_col].min():.2f} - {p[lat_col].max():.2f}")
print(f"  Lon range: {p[lon_col].min():.2f} - {p[lon_col].max():.2f}")

print("\n=== MISSING stations detail ===")
m = locs[locs["station_id"].isin(missing)]
for _, r in m.iterrows():
    sid = r["station_id"]
    lat = r[lat_col]
    lon = r[lon_col]
    state = str(r["state"])
    name = str(r["station_name"])[:50]
    nwq = len(wq[wq["station_id"] == sid])
    print(f"  {sid} | {state:12s} | {lat:8.4f},{lon:8.4f} | WQ rows={nwq:5d} | {name}")

# Categorize failure reasons
print("\n=== Likely failure categories ===")
for _, r in m.iterrows():
    sid = r["station_id"]
    lat = r[lat_col]
    state = str(r["state"])
    if state == "Uttarakhand":
        reason = "NARROW MOUNTAIN RIVER - 500m buffer captures hillside"
    elif state == "Haryana":
        reason = "CANAL/DRAIN - too narrow for 500m buffer"
    elif state in ("Bihar", "Jharkhand") and lat > 25.0:
        reason = "MONSOON CLOUD COVER - Bihar/Jharkhand plains"
    else:
        reason = "NARROW TRIBUTARY or CLOUD"
    print(f"  {sid} ({state:12s}): {reason}")
