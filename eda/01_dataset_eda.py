"""
Comprehensive EDA on the Ganga Water Quality Training Dataset
=============================================================
Analyses performed:
  1. Regional distribution (Upper / Middle / Lower Ganga)
  2. Seasonal / monthly distribution
  3. Sentinel-value & missing-data audit
  4. Outlier analysis per target per region
  5. Impact simulation: what would trimming remove, by region & season?
  6. Climatic-season analysis (Pre-Monsoon / Monsoon / Post-Monsoon / Winter)
  7. Correlation heatmaps per region
  8. Saves all tables to eda/ and figures to eda/figures/

Run:  & ganga\Scripts\python.exe eda\01_dataset_eda.py
"""

import sys, os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

# ── Paths ──
ROOT      = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "training_dataset" / "merged_training_dataset.csv"
OUT_DIR   = ROOT / "eda"
FIG_DIR   = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──
WQ_TARGETS = [
    "wq_BOD", "wq_COD", "wq_CL", "wq_EC", "wq_Depth",
    "wq_DO", "wq_NO3", "wq_TOC", "wq_S", "wq_WT", "wq_WTb", "wq_pH",
]
WQ_SHORT = {
    "wq_BOD": "BOD", "wq_COD": "COD", "wq_CL": "Chloride", "wq_EC": "EC",
    "wq_Depth": "Depth", "wq_DO": "DO", "wq_NO3": "Nitrate", "wq_TOC": "TOC",
    "wq_S": "Water Level", "wq_WT": "Temp", "wq_WTb": "Turbidity", "wq_pH": "pH",
}
BAND_COLS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
INDEX_COLS = ["NDWI", "MNDWI", "NDTI", "NDVI", "NDCI", "MCI", "FAI", "SABI"]

# Physical bounds for validation
PHYSICAL_BOUNDS = {
    "wq_pH": (0, 14), "wq_DO": (0, 25), "wq_BOD": (0, 500),
    "wq_COD": (0, 2000), "wq_EC": (0, 5000), "wq_WT": (0, 50),
    "wq_WTb": (0, 5000), "wq_NO3": (0, 500), "wq_TOC": (0, 500),
    "wq_CL": (0, 5000), "wq_Depth": (0, 50), "wq_S": (0, 1000),
}

# ── Region mapping ──
# Ganga flows roughly west → east.  Classify by longitude.
# Upper Ganga:  source to Kanpur   (lon < 81)
# Middle Ganga: Kanpur to Bhagalpur (lon 81–87)
# Lower Ganga:  Bhagalpur to mouth (lon >= 87)
def assign_region(lon):
    if lon < 81:
        return "Upper Ganga"
    elif lon < 87:
        return "Middle Ganga"
    else:
        return "Lower Ganga"

# Indian climatic seasons
def assign_season(month):
    if month in [12, 1, 2]:
        return "Winter (Dec-Feb)"
    elif month in [3, 4, 5]:
        return "Pre-Monsoon (Mar-May)"
    elif month in [6, 7, 8, 9]:
        return "Monsoon (Jun-Sep)"
    else:
        return "Post-Monsoon (Oct-Nov)"

SEASON_ORDER = [
    "Winter (Dec-Feb)", "Pre-Monsoon (Mar-May)",
    "Monsoon (Jun-Sep)", "Post-Monsoon (Oct-Nov)",
]

# ─────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────
print("=" * 70)
print("GANGA WATER QUALITY TRAINING DATASET — EXPLORATORY DATA ANALYSIS")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df["stationId"] = df["stationId"].astype(str)
for col in WQ_TARGETS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["region"] = df["lon"].apply(assign_region)
df["month"] = df["date"].dt.month
df["season"] = df["month"].apply(assign_season)
df["year"] = df["date"].dt.year

print(f"  Total rows: {len(df)}")
print(f"  Stations : {df['stationId'].nunique()}")
print(f"  Date     : {df['date'].min().date()} → {df['date'].max().date()}")
print(f"  Regions  : {df['region'].value_counts().to_dict()}")


# ═══════════════════════════════════════════════════════════════
# 1. REGIONAL DISTRIBUTION
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("1. REGIONAL DISTRIBUTION")
print("=" * 70)

region_summary = df.groupby("region").agg(
    stations=("stationId", "nunique"),
    rows=("stationId", "count"),
    lat_min=("lat", "min"), lat_max=("lat", "max"),
    lon_min=("lon", "min"), lon_max=("lon", "max"),
    date_min=("date", "min"), date_max=("date", "max"),
).reindex(["Upper Ganga", "Middle Ganga", "Lower Ganga"])
print(region_summary.to_string())
region_summary.to_csv(OUT_DIR / "region_summary.csv")

# Station-level detail
station_detail = df.groupby(["region", "stationId"]).agg(
    rows=("date", "count"),
    lat=("lat", "first"),
    lon=("lon", "first"),
    date_start=("date", "min"),
    date_end=("date", "max"),
).reset_index()
station_detail = station_detail.sort_values(["region", "lon"])
print("\nStation detail:")
print(station_detail.to_string(index=False))
station_detail.to_csv(OUT_DIR / "station_detail.csv", index=False)


# ═══════════════════════════════════════════════════════════════
# 2. SEASONAL DISTRIBUTION
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. SEASONAL & MONTHLY DISTRIBUTION")
print("=" * 70)

season_counts = df.groupby(["region", "season"]).size().unstack(fill_value=0)
season_counts = season_counts.reindex(columns=SEASON_ORDER)
print("\nRows per region × season:")
print(season_counts.to_string())
season_counts.to_csv(OUT_DIR / "region_season_counts.csv")

month_counts = df.groupby(["region", "month"]).size().unstack(fill_value=0)
print("\nRows per region × month:")
print(month_counts.to_string())
month_counts.to_csv(OUT_DIR / "region_month_counts.csv")

# Per-station seasonal coverage
station_season = df.groupby(["stationId", "season"]).size().unstack(fill_value=0)
station_season = station_season.reindex(columns=SEASON_ORDER)
print("\nPer-station season coverage:")
print(station_season.to_string())
station_season.to_csv(OUT_DIR / "station_season_counts.csv")


# ═══════════════════════════════════════════════════════════════
# 3. MISSING DATA & SENTINEL VALUE AUDIT
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. MISSING DATA & SENTINEL VALUE AUDIT")
print("=" * 70)

missing_report = []
for col in WQ_TARGETS:
    total = len(df)
    n_nan = df[col].isna().sum()
    n_sentinel = (df[col] <= -900).sum()
    n_out_lo = 0
    n_out_hi = 0
    if col in PHYSICAL_BOUNDS:
        lo, hi = PHYSICAL_BOUNDS[col]
        valid = df[col].dropna()
        valid = valid[valid > -900]  # exclude sentinels
        n_out_lo = (valid < lo).sum()
        n_out_hi = (valid > hi).sum()
    missing_report.append({
        "target": WQ_SHORT[col],
        "total": total,
        "NaN": n_nan,
        "sentinel_-999": n_sentinel,
        "below_physical_min": n_out_lo,
        "above_physical_max": n_out_hi,
        "clean_count": total - n_nan - n_sentinel - n_out_lo - n_out_hi,
        "clean_pct": 100 * (total - n_nan - n_sentinel - n_out_lo - n_out_hi) / total,
    })

missing_df = pd.DataFrame(missing_report)
print(missing_df.to_string(index=False))
missing_df.to_csv(OUT_DIR / "missing_sentinel_audit.csv", index=False)

# Sentinel values per region
print("\nSentinel values (-999) per region:")
for col in WQ_TARGETS:
    sentinel_mask = df[col] <= -900
    if sentinel_mask.sum() > 0:
        per_region = df[sentinel_mask].groupby("region").size()
        print(f"  {WQ_SHORT[col]:10s}: {per_region.to_dict()}")


# ═══════════════════════════════════════════════════════════════
# 4. OUTLIER ANALYSIS PER TARGET PER REGION
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4. TARGET DISTRIBUTIONS BY REGION (raw, with outliers)")
print("=" * 70)

outlier_report = []
for col in WQ_TARGETS:
    # Work on non-sentinel, non-NaN values
    valid = df[col].dropna()
    valid = valid[valid > -900]
    if len(valid) == 0:
        continue

    p2, p98 = valid.quantile([0.02, 0.98]).values
    n_below = (valid < p2).sum()
    n_above = (valid > p98).sum()
    iqr = valid.quantile(0.75) - valid.quantile(0.25)
    fence_lo = valid.quantile(0.25) - 1.5 * iqr
    fence_hi = valid.quantile(0.75) + 1.5 * iqr
    n_iqr_outliers = ((valid < fence_lo) | (valid > fence_hi)).sum()

    outlier_report.append({
        "target": WQ_SHORT[col],
        "count": len(valid),
        "mean": valid.mean(),
        "std": valid.std(),
        "min": valid.min(),
        "p2": p2,
        "p25": valid.quantile(0.25),
        "median": valid.median(),
        "p75": valid.quantile(0.75),
        "p98": p98,
        "max": valid.max(),
        "n_below_p2": n_below,
        "n_above_p98": n_above,
        "n_iqr_outliers": n_iqr_outliers,
        "iqr_outlier_pct": 100 * n_iqr_outliers / len(valid),
    })

    # Per-region stats
    for region in ["Upper Ganga", "Middle Ganga", "Lower Ganga"]:
        rv = df.loc[(df["region"] == region), col].dropna()
        rv = rv[rv > -900]
        if len(rv) > 0:
            print(f"  {WQ_SHORT[col]:10s} [{region:13s}]  n={len(rv):5d}  "
                  f"mean={rv.mean():9.2f}  std={rv.std():9.2f}  "
                  f"[{rv.min():.2f} – {rv.max():.2f}]")

outlier_df = pd.DataFrame(outlier_report)
print("\nOverall outlier summary:")
print(outlier_df.to_string(index=False))
outlier_df.to_csv(OUT_DIR / "outlier_analysis.csv", index=False)


# ═══════════════════════════════════════════════════════════════
# 5. TRIMMING IMPACT SIMULATION
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("5. TRIMMING IMPACT SIMULATION")
print("=" * 70)
print("Simulating 3 levels of data cleaning to check if any region/season is disproportionately affected.\n")

df_sim = df.copy()
for col in WQ_TARGETS:
    df_sim.loc[df_sim[col] <= -900, col] = np.nan

# Level 1: Remove sentinels only
lvl1 = df_sim.copy()
# Level 2: + remove physically impossible
lvl2 = df_sim.copy()
for col, (lo, hi) in PHYSICAL_BOUNDS.items():
    if col in lvl2.columns:
        lvl2.loc[(lvl2[col] < lo) | (lvl2[col] > hi), col] = np.nan
# Level 3: + clip to p2-p98
lvl3 = lvl2.copy()
for col in WQ_TARGETS:
    valid = lvl3[col].dropna()
    if len(valid) > 100:
        lo, hi = valid.quantile([0.02, 0.98]).values
        lvl3[col] = lvl3[col].clip(lo, hi)

# For each level, count rows with >=8 valid targets
for name, dfl in [("Raw", df), ("Lvl1: Sentinels removed", lvl1),
                   ("Lvl2: + Physical bounds", lvl2), ("Lvl3: + p2-p98 clip", lvl3)]:
    n_valid = dfl[WQ_TARGETS].notna().sum(axis=1)
    keep = n_valid >= 8
    kept = dfl[keep]
    print(f"\n  {name}:")
    print(f"    Rows kept (>=8 valid targets): {keep.sum()} / {len(dfl)} ({100*keep.sum()/len(dfl):.1f}%)")

    # By region
    for region in ["Upper Ganga", "Middle Ganga", "Lower Ganga"]:
        r_total = (dfl["region"] == region).sum()
        r_kept = kept[kept["region"] == region].shape[0]
        r_pct = 100 * r_kept / r_total if r_total > 0 else 0
        print(f"    {region:15s}: {r_kept:5d} / {r_total:5d} ({r_pct:.1f}%)")

    # By season
    for season in SEASON_ORDER:
        s_total = (dfl["season"] == season).sum()
        s_kept = kept[kept["season"] == season].shape[0]
        s_pct = 100 * s_kept / s_total if s_total > 0 else 0
        print(f"    {season:30s}: {s_kept:5d} / {s_total:5d} ({s_pct:.1f}%)")

# Deduplication impact
print("\n  Deduplication impact (averaging WQ per unique satellite observation):")
sat_key_cols = ["stationId"] + BAND_COLS
for name, dfl in [("After Lvl3 + >=8 targets", lvl3)]:
    n_valid = dfl[WQ_TARGETS].notna().sum(axis=1)
    dfl_keep = dfl[n_valid >= 8].copy()
    n_before = len(dfl_keep)
    deduped = dfl_keep.groupby(sat_key_cols, as_index=False).first()
    n_after = len(deduped)
    print(f"    {name}: {n_before} → {n_after} unique satellite observations")

    for region in ["Upper Ganga", "Middle Ganga", "Lower Ganga"]:
        rb = dfl_keep[dfl_keep["region"] == region].shape[0]
        ra = deduped[deduped["region"] == region].shape[0]
        print(f"      {region:15s}: {rb:5d} → {ra:5d}")

    for season in SEASON_ORDER:
        sb = dfl_keep[dfl_keep["season"] == season].shape[0]
        sa = deduped[deduped["season"] == season].shape[0]
        print(f"      {season:30s}: {sb:5d} → {sa:5d}")


# ═══════════════════════════════════════════════════════════════
# 6. REGIONAL CLIMATIC ANALYSIS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("6. REGIONAL × SEASON TARGET MEANS (clean values only)")
print("=" * 70)

# Use lvl2 (sentinels + physical bounds removed) for this view
key_targets = ["wq_DO", "wq_BOD", "wq_COD", "wq_pH", "wq_EC", "wq_WT", "wq_WTb"]
for col in key_targets:
    print(f"\n  {WQ_SHORT[col]}:")
    pivot = lvl2.groupby(["region", "season"])[col].agg(["mean", "count"]).unstack(level="season")
    pivot = pivot.reindex(["Upper Ganga", "Middle Ganga", "Lower Ganga"])
    print(pivot.to_string())


# ═══════════════════════════════════════════════════════════════
# 7. SATELLITE FEATURE DISTRIBUTIONS BY REGION
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("7. SATELLITE FEATURE STATS BY REGION")
print("=" * 70)

sat_features = BAND_COLS + INDEX_COLS
for region in ["Upper Ganga", "Middle Ganga", "Lower Ganga"]:
    print(f"\n  {region}:")
    sub = df[df["region"] == region][sat_features]
    stats = sub.describe().T[["mean", "std", "min", "max"]]
    print(stats.to_string())


# ═══════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("GENERATING FIGURES → eda/figures/")
print("=" * 70)

# ---------- Fig 1: Rows per region × season ----------
fig, ax = plt.subplots(figsize=(10, 5))
season_counts.plot(kind="bar", ax=ax, colormap="Set2")
ax.set_title("Data Distribution: Region × Season")
ax.set_ylabel("Number of rows")
ax.set_xlabel("")
ax.legend(title="Season", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
fig.savefig(FIG_DIR / "01_region_season_distribution.png", dpi=150)
plt.close()
print("  Saved 01_region_season_distribution.png")

# ---------- Fig 2: Monthly distribution per region ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for i, region in enumerate(["Upper Ganga", "Middle Ganga", "Lower Ganga"]):
    sub = df[df["region"] == region]
    sub.groupby("month").size().reindex(range(1, 13), fill_value=0).plot(
        kind="bar", ax=axes[i], color=["#2196F3", "#2196F3", "#4CAF50", "#4CAF50",
        "#4CAF50", "#FF9800", "#FF9800", "#FF9800", "#FF9800", "#9C27B0", "#9C27B0", "#2196F3"])
    axes[i].set_title(region)
    axes[i].set_xlabel("Month")
axes[0].set_ylabel("Number of rows")
plt.suptitle("Monthly Distribution by Region", fontsize=14)
plt.tight_layout()
fig.savefig(FIG_DIR / "02_monthly_distribution_by_region.png", dpi=150)
plt.close()
print("  Saved 02_monthly_distribution_by_region.png")

# ---------- Fig 3: Target boxplots per region ----------
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.ravel()
for i, col in enumerate(WQ_TARGETS):
    valid = df[[col, "region"]].copy()
    valid = valid[(valid[col].notna()) & (valid[col] > -900)]
    # Clip for visualization (not for data)
    p1, p99 = valid[col].quantile([0.01, 0.99]).values
    valid_clipped = valid[(valid[col] >= p1) & (valid[col] <= p99)]
    valid_clipped.boxplot(column=col, by="region", ax=axes[i],
                          positions=[1, 2, 3])
    axes[i].set_title(WQ_SHORT[col])
    axes[i].set_xlabel("")
    axes[i].tick_params(axis="x", rotation=30)
    plt.sca(axes[i])
    plt.title(WQ_SHORT[col])
fig.suptitle("WQ Target Distributions by Region (1st-99th percentile)", fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(FIG_DIR / "03_target_boxplots_by_region.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved 03_target_boxplots_by_region.png")

# ---------- Fig 4: Missing data heatmap per station ----------
missing_matrix = df.groupby("stationId")[WQ_TARGETS].apply(
    lambda x: x.isna().sum() + (x <= -900).sum()
) / df.groupby("stationId").size().values[:, None] * 100
missing_matrix.columns = [WQ_SHORT[c] for c in WQ_TARGETS]
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(missing_matrix, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax,
            vmin=0, vmax=30, cbar_kws={"label": "% missing or sentinel"})
ax.set_title("Missing / Sentinel Data per Station (%)")
ax.set_ylabel("Station ID")
plt.tight_layout()
fig.savefig(FIG_DIR / "04_missing_data_heatmap.png", dpi=150)
plt.close()
print("  Saved 04_missing_data_heatmap.png")

# ---------- Fig 5: Temporal coverage ----------
fig, ax = plt.subplots(figsize=(14, 6))
for idx, sid in enumerate(sorted(df["stationId"].unique(), key=lambda x: df[df["stationId"]==x]["lon"].iloc[0])):
    sub = df[df["stationId"] == sid]
    ax.scatter(sub["date"], [idx]*len(sub), s=0.5, alpha=0.3)
ax.set_yticks(range(len(df["stationId"].unique())))
ax.set_yticklabels(sorted(df["stationId"].unique(), key=lambda x: df[df["stationId"]==x]["lon"].iloc[0]))
ax.set_xlabel("Date")
ax.set_ylabel("Station ID (west → east)")
ax.set_title("Temporal Coverage per Station")
plt.tight_layout()
fig.savefig(FIG_DIR / "05_temporal_coverage.png", dpi=150)
plt.close()
print("  Saved 05_temporal_coverage.png")

# ---------- Fig 6: Correlation heatmap — satellite vs WQ per region ----------
fig, axes = plt.subplots(1, 3, figsize=(24, 8))
sat_feats_for_corr = BAND_COLS + INDEX_COLS
for i, region in enumerate(["Upper Ganga", "Middle Ganga", "Lower Ganga"]):
    sub = lvl2[lvl2["region"] == region]
    corr = sub[sat_feats_for_corr + WQ_TARGETS].corr().loc[sat_feats_for_corr, WQ_TARGETS]
    corr.columns = [WQ_SHORT[c] for c in WQ_TARGETS]
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-0.5, vmax=0.5, ax=axes[i], annot_kws={"size": 7})
    axes[i].set_title(f"{region}")
plt.suptitle("Satellite Feature ↔ WQ Target Correlations by Region", fontsize=14)
plt.tight_layout()
fig.savefig(FIG_DIR / "06_correlation_heatmaps_by_region.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved 06_correlation_heatmaps_by_region.png")

# ---------- Fig 7: Seasonal WQ trends ----------
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.ravel()
for i, col in enumerate(WQ_TARGETS):
    sub = lvl2[[col, "season", "region"]].dropna()
    for region in ["Upper Ganga", "Middle Ganga", "Lower Ganga"]:
        rsub = sub[sub["region"] == region]
        means = rsub.groupby("season")[col].mean().reindex(SEASON_ORDER)
        axes[i].plot(range(4), means.values, marker="o", label=region)
    axes[i].set_title(WQ_SHORT[col])
    axes[i].set_xticks(range(4))
    axes[i].set_xticklabels(["Win", "Pre-M", "Mon", "Post-M"], fontsize=8)
    if i == 0:
        axes[i].legend(fontsize=7)
plt.suptitle("Seasonal Trends by Region (mean values, sentinels removed)", fontsize=14)
plt.tight_layout()
fig.savefig(FIG_DIR / "07_seasonal_wq_trends.png", dpi=150)
plt.close()
print("  Saved 07_seasonal_wq_trends.png")

# ---------- Fig 8: Station map (scatter on lat/lon, colored by region) ----------
fig, ax = plt.subplots(figsize=(12, 6))
colors = {"Upper Ganga": "#e74c3c", "Middle Ganga": "#f39c12", "Lower Ganga": "#27ae60"}
for region, color in colors.items():
    sub = station_detail[station_detail["region"] == region]
    ax.scatter(sub["lon"], sub["lat"], c=color, s=sub["rows"]/10, label=region, alpha=0.7, edgecolors="k")
    for _, row in sub.iterrows():
        ax.annotate(row["stationId"], (row["lon"], row["lat"]), fontsize=7,
                    textcoords="offset points", xytext=(5, 5))
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Station Locations (size ∝ data count)")
ax.legend()
plt.tight_layout()
fig.savefig(FIG_DIR / "08_station_map.png", dpi=150)
plt.close()
print("  Saved 08_station_map.png")

# ---------- Fig 9: Trimming impact comparison ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Region impact
labels = ["Upper Ganga", "Middle Ganga", "Lower Ganga"]
raw_counts = [len(df[df["region"]==r]) for r in labels]
# After all cleaning + dedup
dfl_clean = lvl3.copy()
n_valid = dfl_clean[WQ_TARGETS].notna().sum(axis=1)
dfl_clean = dfl_clean[n_valid >= 8]
deduped_clean = dfl_clean.groupby(sat_key_cols + ["region"], as_index=False).first()
clean_counts = [len(deduped_clean[deduped_clean["region"]==r]) for r in labels]

x = np.arange(len(labels))
w = 0.35
axes[0].bar(x - w/2, raw_counts, w, label="Raw", color="#3498db")
axes[0].bar(x + w/2, clean_counts, w, label="After cleaning + dedup", color="#e74c3c")
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels)
axes[0].set_ylabel("Rows")
axes[0].set_title("Trimming Impact by Region")
axes[0].legend()
for j in range(len(labels)):
    pct = 100 * clean_counts[j] / raw_counts[j] if raw_counts[j] > 0 else 0
    axes[0].text(x[j] + w/2, clean_counts[j] + 50, f"{pct:.0f}%", ha="center", fontsize=9)

# Season impact
raw_s = [len(df[df["season"]==s]) for s in SEASON_ORDER]
clean_s = [len(deduped_clean[deduped_clean["season"]==s]) for s in SEASON_ORDER]
x2 = np.arange(len(SEASON_ORDER))
axes[1].bar(x2 - w/2, raw_s, w, label="Raw", color="#3498db")
axes[1].bar(x2 + w/2, clean_s, w, label="After cleaning + dedup", color="#e74c3c")
axes[1].set_xticks(x2)
axes[1].set_xticklabels(["Winter", "Pre-Mon", "Monsoon", "Post-Mon"])
axes[1].set_ylabel("Rows")
axes[1].set_title("Trimming Impact by Season")
axes[1].legend()
for j in range(len(SEASON_ORDER)):
    pct = 100 * clean_s[j] / raw_s[j] if raw_s[j] > 0 else 0
    axes[1].text(x2[j] + w/2, clean_s[j] + 50, f"{pct:.0f}%", ha="center", fontsize=9)

plt.tight_layout()
fig.savefig(FIG_DIR / "09_trimming_impact.png", dpi=150)
plt.close()
print("  Saved 09_trimming_impact.png")

# ---------- Fig 10: Per-station trimming survival ----------
survival = []
for sid in sorted(df["stationId"].unique()):
    raw_n = len(df[df["stationId"] == sid])
    clean_n = len(deduped_clean[deduped_clean["stationId"] == sid])
    region = df[df["stationId"] == sid]["region"].iloc[0]
    survival.append({"stationId": sid, "raw": raw_n, "cleaned": clean_n,
                     "survival_pct": 100 * clean_n / raw_n, "region": region})
survival_df = pd.DataFrame(survival)
survival_df.to_csv(OUT_DIR / "station_trimming_survival.csv", index=False)

fig, ax = plt.subplots(figsize=(14, 5))
c = [colors[r] for r in survival_df["region"]]
ax.bar(range(len(survival_df)), survival_df["survival_pct"], color=c)
ax.set_xticks(range(len(survival_df)))
ax.set_xticklabels(survival_df["stationId"], rotation=45)
ax.set_ylabel("% rows surviving after cleaning + dedup")
ax.set_title("Per-Station Data Survival After Cleaning")
ax.axhline(y=survival_df["survival_pct"].mean(), color="gray", linestyle="--", label=f"Mean: {survival_df['survival_pct'].mean():.0f}%")
ax.legend()
# Add region legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=r) for r, c in colors.items()]
ax.legend(handles=legend_elements, loc="upper right")
plt.tight_layout()
fig.savefig(FIG_DIR / "10_station_survival.png", dpi=150)
plt.close()
print("  Saved 10_station_survival.png")


# ═══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EDA COMPLETE — SUMMARY")
print("=" * 70)
print(f"  Tables saved to: {OUT_DIR}/")
print(f"  Figures saved to: {FIG_DIR}/")
print(f"\n  Key files:")
for f in sorted(OUT_DIR.glob("*.csv")):
    print(f"    {f.name}")
print(f"  Figures:")
for f in sorted(FIG_DIR.glob("*.png")):
    print(f"    {f.name}")
