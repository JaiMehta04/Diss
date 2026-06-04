"""Summarize exported GeoTIFF patches (NDWI/NDTI/MCI + RGB) into a CSV.

Usage:
    python analyze_exports.py

What it does:
- Scans the GangaIndices folder for .tif files.
- For each band in each file, computes count, mean, std, min, max, p5, p50, p95 (ignores nodata/masked pixels).
- Writes a summary CSV next to the script (band_stats.csv).

Dependencies: rasterio, numpy, pandas. Install if needed, e.g.:
    pip install rasterio pandas numpy
"""
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "GangaIndices"
OUTPUT_CSV = PROJECT_ROOT / "band_stats.csv"


def list_tifs(folder: Path) -> List[Path]:
    return sorted([p for p in folder.glob("*.tif") if p.is_file()])


def band_stats(array: np.ndarray) -> dict:
    flat = array.astype(float).ravel()
    flat = flat[~np.isnan(flat)]
    if flat.size == 0:
        return {"count": 0, "mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan,
                "p5": np.nan, "p50": np.nan, "p95": np.nan}
    return {
        "count": int(flat.size),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "p5": float(np.percentile(flat, 5)),
        "p50": float(np.percentile(flat, 50)),
        "p95": float(np.percentile(flat, 95)),
    }


def process_file(path: Path) -> List[dict]:
    rows: List[dict] = []
    with rasterio.open(path) as ds:
        nodata = ds.nodata
        for i in range(1, ds.count + 1):
            band_name = ds.descriptions[i - 1] if ds.descriptions and len(ds.descriptions) >= i else f"band_{i}"
            arr = ds.read(i, resampling=Resampling.nearest).astype(float)
            if nodata is not None:
                arr[arr == nodata] = np.nan
            stats = band_stats(arr)
            stats.update({"file": path.name, "band": band_name})
            rows.append(stats)
    return rows


def main() -> None:
    if not DATA_DIR.exists():
        raise SystemExit(f"Folder not found: {DATA_DIR}")

    tifs = list_tifs(DATA_DIR)
    if not tifs:
        raise SystemExit(f"No .tif files found in {DATA_DIR}")

    all_rows: List[dict] = []
    for tif in tifs:
        print(f"Processing {tif.name}...")
        all_rows.extend(process_file(tif))

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
