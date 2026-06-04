"""Visualize exported GeoTIFFs from GangaIndices.

What it does:
- Loads a GeoTIFF (default: first .tif in GangaIndices or a specific --file).
- Renders NDWI, NDTI, MCI as grayscale with percentile stretch.
- Renders an RGB composite from B4/B3/B2 with percentile stretch.
- Saves a PNG to outputs/vis_<tifname>.png and also shows an interactive window (can disable with --no-show).

Usage examples:
    python visualize_exports.py
    python visualize_exports.py --file GangaIndices/station11783_2025-07-29_pm3d.tif
    python visualize_exports.py --no-show

Dependencies: rasterio, matplotlib, numpy. Install if needed:
    pip install rasterio matplotlib numpy
"""
import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.plot import reshape_as_raster

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "GangaIndices"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
BANDS_RGB = ["B4", "B3", "B2"]
INDEX_BANDS = ["NDWI", "NDTI", "MCI"]


def percentile_stretch(arr: np.ndarray, p_low: float = 2, p_high: float = 98) -> np.ndarray:
    lo, hi = np.nanpercentile(arr, [p_low, p_high])
    if hi <= lo:
        return np.clip(arr, lo, hi)
    stretched = (arr - lo) / (hi - lo)
    return np.clip(stretched, 0, 1)


def load_band_map(ds: rasterio.io.DatasetReader) -> Dict[str, np.ndarray]:
    band_map: Dict[str, np.ndarray] = {}
    for i in range(1, ds.count + 1):
        name = ds.descriptions[i - 1] if ds.descriptions and len(ds.descriptions) >= i else f"band_{i}"
        data = ds.read(i).astype(float)
        nodata = ds.nodata
        if nodata is not None:
            data[data == nodata] = np.nan
        band_map[name] = data
    return band_map


def pick_first_tif(folder: Path) -> Path:
    tifs = sorted(folder.glob("*.tif"))
    if not tifs:
        raise SystemExit(f"No .tif files found in {folder}")
    return tifs[0]


def plot_arrays(band_map: Dict[str, np.ndarray], tif_path: Path, show: bool) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(tif_path.name)

    # RGB
    if all(b in band_map for b in BANDS_RGB):
        # bands stacked as (3, H, W)
        rgb = np.stack([band_map[b] for b in BANDS_RGB], axis=0)
        stretched = []
        for c in range(3):
            stretched.append(percentile_stretch(rgb[c]))
        rgb_img = np.stack(stretched, axis=2)  # (H, W, 3)
        axes[0, 0].imshow(rgb_img)
        axes[0, 0].set_title("RGB (B4/B3/B2)")
    else:
        axes[0, 0].text(0.5, 0.5, "RGB bands missing", ha="center", va="center")
    axes[0, 0].axis("off")

    # Index bands
    for ax, name in zip(axes.flat[1:], INDEX_BANDS):
        if name in band_map:
            arr = band_map[name]
            img = percentile_stretch(arr)
            im = ax.imshow(img, cmap="RdYlBu_r")
            ax.set_title(name)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.axis("off")
        else:
            ax.text(0.5, 0.5, f"{name} missing", ha="center", va="center")
            ax.axis("off")

    out_path = OUTPUT_DIR / f"vis_{tif_path.stem}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved visualization to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize exported GeoTIFFs")
    parser.add_argument("--file", type=str, help="Path to a specific .tif (default: first in GangaIndices)")
    parser.add_argument("--no-show", action="store_true", help="Do not open an interactive window")
    args = parser.parse_args()

    tif_path = Path(args.file) if args.file else pick_first_tif(DATA_DIR)
    if not tif_path.exists():
        raise SystemExit(f"File not found: {tif_path}")

    with rasterio.open(tif_path) as ds:
        band_map = load_band_map(ds)
    plot_arrays(band_map, tif_path, show=not args.no_show)


if __name__ == "__main__":
    main()
