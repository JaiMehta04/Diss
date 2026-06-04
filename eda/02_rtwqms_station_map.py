"""
RTWQMS Station Map — All 40 Ganga Basin Monitoring Stations
===========================================================
Generates:
  1. Interactive HTML map (folium) with river labels, color-coded by river
  2. Static PNG map (matplotlib) saved to eda/figures/
"""

import sys, os, re
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import folium
from folium import plugins

ROOT      = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "cpcb_station_locations.csv"
TRAIN_CSV = ROOT / "training_dataset" / "merged_training_dataset.csv"
OUT_DIR   = ROOT / "eda" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load station data ──
df = pd.read_csv(DATA_PATH)

# ── Extract river name from station_name ──
def extract_river(name):
    name_lower = name.lower()
    # Direct river mentions
    river_patterns = [
        (r'river\s+ganga|on\s+river\s+ganga|river\s+ganga\b', 'Ganga'),
        (r'\bganga\b', 'Ganga'),
        (r'river\s+yamuna|yamuna', 'Yamuna'),
        (r'\bhindon\b', 'Hindon'),
        (r'kali\s+east|river\s+kali', 'Kali East'),
        (r'\bghagra\b', 'Ghagra'),
        (r'\bgandak\b', 'Gandak'),
        (r'burhi\s+gandak', 'Burhi Gandak'),
        (r'\bkosi\b', 'Kosi'),
        (r'\bson\b', 'Son'),
        (r'\bpunpun\b', 'Punpun'),
        (r'\bdamodar\b', 'Damodar'),
        (r'\bramganga\b', 'Ramganga'),
        (r'river\s+kosi', 'Kosi (UP)'),
        (r'\bhooghly\b', 'Hooghly'),
    ]
    for pattern, river in river_patterns:
        if re.search(pattern, name_lower):
            return river

    # Location-based inference for stations without explicit river name
    loc_rivers = {
        'haridwar': 'Ganga', 'rishikesh': 'Ganga', 'rudraprayag': 'Ganga',
        'tehri': 'Ganga (Bhagirathi)', 'srinagar': 'Ganga (Alaknanda)',
        'allahabad': 'Ganga', 'fafamau': 'Ganga', 'chunar': 'Ganga',
        'ghazipur': 'Ganga', 'buxar': 'Ganga', 'chausa': 'Ganga',
        'patna': 'Ganga', 'bhagalpur': 'Ganga', 'farakka': 'Ganga',
        'nabadwip': 'Ganga (Hooghly)', 'chinsura': 'Ganga (Hooghly)',
        'durgapur': 'Damodar', 'raghunathpur': 'Damodar',
        'sahebganj': 'Ganga', 'rajmahal': 'Ganga',
        'mathura': 'Yamuna', 'gokul': 'Yamuna',
        'auraiya': 'Yamuna/Ganga tributaries', 'fatehpur': 'Ganga',
        'mohana': 'Yamuna/Western drain', 'sonipat': 'Yamuna/Western drain',
        'pratapgarh': 'Sai/Ganga tributary', 'kashipur': 'Kosi (UP)',
        'ramgarh': 'Damodar', 'pathardih': 'Damodar',
    }
    for loc, river in loc_rivers.items():
        if loc in name_lower:
            return river

    return 'Ganga basin (unspecified)'

df['river'] = df['station_name'].apply(extract_river)

# Check which stations are in our training dataset
train_df = pd.read_csv(TRAIN_CSV)
train_stations = set(train_df['stationId'].unique())
df['in_training'] = df['station_id'].isin(train_stations)

# ── River color mapping ──
RIVER_COLORS = {
    'Ganga': '#1a73e8',
    'Ganga (Bhagirathi)': '#4285f4',
    'Ganga (Alaknanda)': '#5e97f6',
    'Ganga (Hooghly)': '#1565c0',
    'Yamuna': '#e53935',
    'Hindon': '#ff7043',
    'Kali East': '#ff5722',
    'Ghagra': '#7b1fa2',
    'Gandak': '#9c27b0',
    'Burhi Gandak': '#ab47bc',
    'Kosi': '#00897b',
    'Kosi (UP)': '#26a69a',
    'Son': '#558b2f',
    'Punpun': '#8bc34a',
    'Damodar': '#f57f17',
    'Ramganga': '#ff6f00',
    'Sai/Ganga tributary': '#795548',
    'Yamuna/Ganga tributaries': '#d32f2f',
    'Yamuna/Western drain': '#c62828',
    'Ganga basin (unspecified)': '#9e9e9e',
}

print("=" * 70)
print("RTWQMS STATION MAP — 40 GANGA BASIN STATIONS")
print("=" * 70)
print(f"\nStations by river:")
for river, count in df['river'].value_counts().items():
    print(f"  {river:30s}: {count} stations")

print(f"\nIn training dataset: {df['in_training'].sum()} / {len(df)}")

# ═══════════════════════════════════════════════════════════════
# 1. INTERACTIVE FOLIUM MAP
# ═══════════════════════════════════════════════════════════════
print("\nGenerating interactive map...")

center_lat = df['latitude'].mean()
center_lon = df['longitude'].mean()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=6,
    tiles='OpenStreetMap',
)

# Add different tile layers
folium.TileLayer('CartoDB positron', name='Light').add_to(m)
folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri',
    name='Satellite',
).add_to(m)

# Create feature groups per river
river_groups = {}
for river in df['river'].unique():
    fg = folium.FeatureGroup(name=f"🌊 {river}")
    river_groups[river] = fg
    m.add_child(fg)

# Training dataset layer
fg_train = folium.FeatureGroup(name="✅ In Training Dataset", show=True)
fg_nottrain = folium.FeatureGroup(name="❌ Not in Training", show=True)
m.add_child(fg_train)
m.add_child(fg_nottrain)

for _, row in df.iterrows():
    river = row['river']
    color = RIVER_COLORS.get(river, '#9e9e9e')
    in_train = row['in_training']
    
    # Clean station name
    name_parts = row['station_name'].split('_', 1)
    code = name_parts[0]
    desc = name_parts[1] if len(name_parts) > 1 else name_parts[0]
    
    # Popup content
    popup_html = f"""
    <div style="font-family: Arial; min-width: 250px;">
        <h4 style="margin:0; color: {color};">{code}</h4>
        <p style="margin:2px 0;"><b>{desc}</b></p>
        <table style="font-size:12px;">
            <tr><td><b>Station ID:</b></td><td>{row['station_id']}</td></tr>
            <tr><td><b>River:</b></td><td style="color:{color};font-weight:bold;">{river}</td></tr>
            <tr><td><b>State:</b></td><td>{row['state']}</td></tr>
            <tr><td><b>Lat/Lon:</b></td><td>{row['latitude']:.4f}, {row['longitude']:.4f}</td></tr>
            <tr><td><b>In Training:</b></td><td>{'✅ Yes' if in_train else '❌ No'}</td></tr>
        </table>
    </div>
    """
    
    # Marker icon
    icon_color = 'green' if in_train else 'red'
    icon = folium.Icon(color=icon_color, icon='tint', prefix='fa')
    
    marker = folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=folium.Popup(popup_html, max_width=300),
        tooltip=f"{code} — {river} ({row['state']})",
        icon=icon,
    )
    
    # Add to river group
    river_groups[river].add_child(marker)
    
    # Add to training/not-training group
    if in_train:
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=12, color='green', fill=True, fill_opacity=0.3,
            weight=2,
        ).add_to(fg_train)
    else:
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=8, color='red', fill=True, fill_opacity=0.3,
            weight=2,
        ).add_to(fg_nottrain)

# Add approximate river flow lines (simplified coordinates)
ganga_coords = [
    [30.99, 78.94],   # Gangotri
    [30.37, 78.48],   # Tehri
    [30.27, 78.96],   # Rudraprayag
    [30.07, 78.29],   # Rishikesh
    [29.94, 78.16],   # Haridwar
    [29.11, 77.44],   # Baghpat
    [27.44, 77.71],   # Mathura area
    [26.41, 79.49],   # Auraiya
    [25.78, 80.58],   # Fatehpur
    [25.43, 81.86],   # Allahabad
    [25.13, 82.88],   # Chunar
    [25.59, 83.61],   # Ghazipur
    [25.52, 83.90],   # Buxar
    [25.65, 85.10],   # Patna
    [25.31, 87.02],   # Bhagalpur
    [24.80, 87.92],   # Farakka
    [23.48, 87.30],   # Durgapur area
    [23.40, 88.36],   # Nabadwip
    [22.91, 88.40],   # Hooghly/Kolkata
]

folium.PolyLine(
    ganga_coords, color='#1a73e8', weight=3,
    opacity=0.6, tooltip='Ganga River (approximate)',
    dash_array='10',
).add_to(m)

# Legend
legend_html = """
<div style="position: fixed; bottom: 50px; left: 50px; z-index:999;
     background: white; padding: 12px; border-radius: 8px;
     box-shadow: 0 2px 6px rgba(0,0,0,0.3); font-size: 13px;
     font-family: Arial; max-height: 400px; overflow-y: auto;">
<h4 style="margin:0 0 8px;">RTWQMS Stations — Ganga Basin</h4>
<p style="margin:2px 0;"><span style="color:green;">●</span> In Training Dataset (21)</p>
<p style="margin:2px 0;"><span style="color:red;">●</span> Not in Training (19)</p>
<hr style="margin:5px 0;">
<p style="margin:2px 0; font-weight:bold;">Rivers:</p>
"""
for river in sorted(df['river'].unique()):
    c = RIVER_COLORS.get(river, '#9e9e9e')
    n = (df['river'] == river).sum()
    legend_html += f'<p style="margin:1px 0;"><span style="color:{c};">■</span> {river} ({n})</p>\n'
legend_html += "</div>"
m.get_root().html.add_child(folium.Element(legend_html))

# Layer control
folium.LayerControl(collapsed=False).add_to(m)

# Minimap
plugins.MiniMap(toggle_display=True).add_to(m)

map_path = OUT_DIR / "rtwqms_station_map.html"
m.save(str(map_path))
print(f"  Interactive map saved: {map_path}")

# ═══════════════════════════════════════════════════════════════
# 2. STATIC PNG MAP (matplotlib)
# ═══════════════════════════════════════════════════════════════
print("Generating static PNG map...")

fig, ax = plt.subplots(figsize=(16, 10))

# Plot approximate river path
ganga_lons = [c[1] for c in ganga_coords]
ganga_lats = [c[0] for c in ganga_coords]
ax.plot(ganga_lons, ganga_lats, '-', color='#64b5f6', linewidth=3, alpha=0.5,
        label='Ganga (approx.)', zorder=1)

# Plot all stations
for river in sorted(df['river'].unique()):
    sub = df[df['river'] == river]
    color = RIVER_COLORS.get(river, '#9e9e9e')
    
    train_sub = sub[sub['in_training']]
    notrain_sub = sub[~sub['in_training']]
    
    if len(train_sub) > 0:
        ax.scatter(train_sub['longitude'], train_sub['latitude'],
                   c=color, s=120, edgecolors='green', linewidths=2,
                   zorder=3, marker='o', label=f"{river} (train)")
    if len(notrain_sub) > 0:
        ax.scatter(notrain_sub['longitude'], notrain_sub['latitude'],
                   c=color, s=80, edgecolors='red', linewidths=1.5,
                   zorder=2, marker='s', label=f"{river} (not in train)")

# Label each station
for _, row in df.iterrows():
    code = row['station_name'].split('_')[0]
    offset = (5, 5)
    txt = ax.annotate(
        code,
        (row['longitude'], row['latitude']),
        textcoords="offset points", xytext=offset,
        fontsize=6.5, fontweight='bold',
        color='#333333',
        path_effects=[pe.withStroke(linewidth=2, foreground='white')],
    )

# Region boundaries
ax.axvline(x=81, color='gray', linestyle=':', alpha=0.4, label='Region boundaries')
ax.axvline(x=87, color='gray', linestyle=':', alpha=0.4)
ax.text(79, 22.5, 'UPPER\nGANGA', ha='center', fontsize=10, color='gray', alpha=0.5)
ax.text(84, 22.5, 'MIDDLE\nGANGA', ha='center', fontsize=10, color='gray', alpha=0.5)
ax.text(88, 22.5, 'LOWER\nGANGA', ha='center', fontsize=10, color='gray', alpha=0.5)

# State labels (approximate centers)
state_labels = {
    'Uttarakhand': (78.5, 30.6),
    'Haryana': (76.8, 29.2),
    'Uttar Pradesh': (80.5, 27.5),
    'Bihar': (85.5, 26.0),
    'Jharkhand': (86.5, 24.5),
    'West Bengal': (88.0, 23.0),
}
for state, (lon, lat) in state_labels.items():
    ax.text(lon, lat, state, fontsize=9, ha='center', color='#777',
            fontstyle='italic', alpha=0.7)

ax.set_xlabel('Longitude (°E)', fontsize=12)
ax.set_ylabel('Latitude (°N)', fontsize=12)
ax.set_title('CPCB RTWQMS Monitoring Stations — Ganga Basin\n'
             '(Green border = in training dataset, Red border = not used)',
             fontsize=14, fontweight='bold')

# Compact legend
handles, labels = ax.get_legend_handles_labels()
# Remove duplicates
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(),
          loc='lower left', fontsize=7, ncol=2,
          framealpha=0.9, edgecolor='gray')

ax.set_xlim(76.5, 89.5)
ax.set_ylim(22, 31.5)
ax.grid(True, alpha=0.2)
ax.set_aspect('equal')

plt.tight_layout()
png_path = OUT_DIR / "rtwqms_all_stations_map.png"
fig.savefig(png_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"  Static map saved: {png_path}")

# ── Summary table ──
summary = df[['station_id', 'station_name', 'state', 'river', 'latitude', 'longitude', 'in_training']].copy()
summary = summary.sort_values(['river', 'longitude'])
summary.to_csv(OUT_DIR.parent / "rtwqms_station_river_mapping.csv", index=False)
print(f"  Station-river mapping CSV saved: eda/rtwqms_station_river_mapping.csv")

print(f"\n  Total stations: {len(df)}")
print(f"  Rivers monitored: {df['river'].nunique()}")
print(f"  In training: {df['in_training'].sum()}, Not used: {(~df['in_training']).sum()}")
