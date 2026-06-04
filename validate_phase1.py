"""Validate Phase 1 improvements."""
import pandas as pd
from pathlib import Path

BASE = Path(r"c:\Users\VHK6FSM\Personal\Dissertation\Ganga_Google_Earth")
wide = pd.read_csv(BASE / "data/final_v2/ganga_water_quality_final.csv", low_memory=False)
long = pd.read_csv(BASE / "data/final_v2/ganga_water_quality_long.csv", low_memory=False)

# Check numeric station names
numeric = wide[wide["station_name"].str.match(r"^\d+$", na=False)]
print("=== NUMERIC STATION CHECK ===")
print(f"Numeric-ID stations: {numeric['station_name'].nunique()} unique")
print(f"Rows with numeric names: {len(numeric)}")
if len(numeric) > 0:
    print(f"Sample: {numeric['station_name'].unique()[:5].tolist()}")
    print(f"Sources: {long[long['station_name'].str.match(r'^\\d+$', na=False)]['source'].value_counts().to_dict()}")
print()

# Quality flags
print("=== QUALITY FLAGS ===")
print(long["quality_flag"].value_counts().to_string())
print()

# Check consistency specifically
suspect_cons = long[long["quality_flag"] == "suspect_consistency"]
print(f"Suspect consistency: {len(suspect_cons)} ({100*len(suspect_cons)/len(long):.1f}%)")
print(f"  (Was 16.2% before fix)")
print()

# Summary
print("=== SUMMARY ===")
print(f"Total wide rows: {len(wide)}")
print(f"Total long records: {len(long)}")
print(f"Unique stations: {wide['station_name'].nunique()}")
print(f"Unique states: {wide['state'].nunique()}")
print()

# Spatial filter effect
print("=== SPATIAL VALIDATION ===")
has_coords = wide.dropna(subset=["latitude", "longitude"])
print(f"Rows with coordinates: {len(has_coords)} / {len(wide)} ({100*len(has_coords)/len(wide):.1f}%)")
print()

# Compare with previous
print("=== COMPARISON WITH PREVIOUS RUN ===")
print(f"Previous: 14,288 wide rows, 175,404 long records, 656 stations")
print(f"Current:  {len(wide):,} wide rows, {len(long):,} long records, {wide['station_name'].nunique()} stations")
print(f"Previous numeric stations: 101")
print(f"Current numeric stations:  {numeric['station_name'].nunique()}")
print(f"Previous suspect_consistency: 16.2%")
print(f"Current suspect_consistency:  {100*len(suspect_cons)/len(long):.1f}%")
