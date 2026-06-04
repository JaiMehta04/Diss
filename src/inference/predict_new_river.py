"""
predict_new_river.py
====================
Predict water quality parameters for a new river using:
  1. A Sentinel-2 GeoTIFF image (exported from GEE), OR
  2. Coordinates + date (auto-fetches from GEE via Earth Engine API)

Loads the trained models from bayesian_wq_model.py and outputs
predicted WQ values with confidence intervals.

Usage
-----
  # From coordinates + date
  python predict_new_river.py --lat 26.85 --lon 80.91 --date 2025-09-15

  # From a GeoTIFF file
  python predict_new_river.py --image path/to/sentinel2.tif

  # Multiple points from a CSV (columns: lat, lon, date)
  python predict_new_river.py --csv new_river_points.csv

  # Specify which model to use
  python predict_new_river.py --lat 26.85 --lon 80.91 --date 2025-09-15 --model "Random Forest"
"""

import argparse
import csv
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Ensure PyTorch is importable
_torch_alt = r"C:\torch_tmp"
if os.path.isdir(_torch_alt) and _torch_alt not in sys.path:
    sys.path.insert(0, _torch_alt)

import torch

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "compat"))
MODELS_DIR = BASE_DIR / "training_dataset" / "results" / "models"

BAND_COLS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
INDEX_COLS = ["NDWI", "MNDWI", "NDTI", "NDVI", "NDCI", "MCI", "FAI", "SABI"]

WQ_SHORT = {
    "wq_BOD": "BOD",
    "wq_COD": "COD",
    "wq_CL": "Cl⁻",
    "wq_EC": "EC",
    "wq_Depth": "Depth",
    "wq_DO": "DO",
    "wq_NO3": "NO₃⁻",
    "wq_TOC": "TOC",
    "wq_S": "WL",
    "wq_WT": "Temp",
    "wq_WTb": "Turb",
    "wq_pH": "pH",
}

SEARCH_WINDOWS = [(3, 20), (10, 60), (30, 100)]
BUFFER_METERS = 500
GANGOTRI_LAT, GANGOTRI_LON = 30.99, 78.94


# ──────────────────────────────────────────────────────────────────────────────
# Load trained artifacts
# ──────────────────────────────────────────────────────────────────────────────
def load_artifacts():
    print("Loading trained models ...")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    all_models = joblib.load(MODELS_DIR / "all_models.joblib")
    with (MODELS_DIR / "feature_config.json").open() as f:
        config = json.load(f)
    print(f"  Loaded {len(all_models)} target models, {len(config['features'])} features")
    return scaler, all_models, config


# ──────────────────────────────────────────────────────────────────────────────
# Extract bands from GeoTIFF
# ──────────────────────────────────────────────────────────────────────────────
def extract_from_geotiff(image_path: str) -> Dict[str, float]:
    """Read a Sentinel-2 GeoTIFF and compute mean band values + indices."""
    try:
        import rasterio
    except ImportError:
        raise ImportError("Install rasterio: pip install rasterio")

    with rasterio.open(image_path) as src:
        band_names = list(src.descriptions) if src.descriptions[0] else [f"band_{i+1}" for i in range(src.count)]
        data = {}
        for i, name in enumerate(band_names):
            band_data = src.read(i + 1).astype(float)
            band_data[band_data == src.nodata] = np.nan if src.nodata is not None else np.nan
            data[name] = np.nanmean(band_data)

    return data


def compute_indices_from_bands(bands: Dict[str, float]) -> Dict[str, float]:
    """Compute spectral indices from band values."""
    b2 = bands.get("B2", 0)
    b3 = bands.get("B3", 0)
    b4 = bands.get("B4", 0)
    b5 = bands.get("B5", 0)
    b6 = bands.get("B6", 0)
    b8 = bands.get("B8", 0)
    b11 = bands.get("B11", 0)

    safe_div = lambda a, b: a / b if abs(b) > 1e-10 else 0

    indices = {
        "NDWI": safe_div(b3 - b8, b3 + b8),
        "MNDWI": safe_div(b3 - b11, b3 + b11),
        "NDTI": safe_div(b4 - b3, b4 + b3),
        "NDVI": safe_div(b8 - b4, b8 + b4),
        "NDCI": safe_div(b5 - b4, b5 + b4),
        "MCI": b5 - b4 - (705 - 665) / (740 - 665) * (b6 - b4),
        "FAI": b8 - b4 - (842 - 665) / (1610 - 665) * (b11 - b4),
        "SABI": safe_div(b8 - b4, b3 + b2) if (b3 + b2) > 1e-10 else 0,
    }
    return indices


# ──────────────────────────────────────────────────────────────────────────────
# Extract from GEE (coordinates + date)
# ──────────────────────────────────────────────────────────────────────────────
def extract_from_gee(lat: float, lon: float, date_str: str) -> Optional[Dict[str, float]]:
    """Fetch Sentinel-2 data from Earth Engine for given coords + date."""
    import ee
    ee.Initialize(project="dissertation-ganga")

    point = ee.Geometry.Point([lon, lat])
    base_bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

    for days, cloud_pct in SEARCH_WINDOWS:
        start = ee.Date(date_str).advance(-days, "day")
        end = ee.Date(date_str).advance(days, "day")
        col = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(point.buffer(BUFFER_METERS))
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
        )
        count = col.size().getInfo()
        if count and count > 0:
            # Cloud mask and scale
            def mask_scale(img):
                qa = img.select("QA60")
                clear = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
                return img.updateMask(clear).divide(10000).select(base_bands)

            composite = col.map(mask_scale).median()

            values = composite.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point.buffer(BUFFER_METERS),
                scale=10,
                maxPixels=1e8,
            ).getInfo()

            if values and any(v is not None for v in values.values()):
                print(f"  Found {count} scenes (±{days}d, cloud<{cloud_pct}%)")
                return values

    print(f"  No Sentinel-2 data found for ({lat}, {lon}) near {date_str}")
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Build feature vector
# ──────────────────────────────────────────────────────────────────────────────
def build_feature_vector(
    bands: Dict[str, float],
    lat: float,
    lon: float,
    date_str: str,
    feature_list: List[str],
) -> np.ndarray:
    """Construct the full feature vector matching training configuration."""
    dt = pd.Timestamp(date_str)

    indices = compute_indices_from_bands(bands)
    all_vals = {**bands, **indices}

    # Temporal features
    month = dt.month
    all_vals["month_sin"] = np.sin(2 * np.pi * month / 12)
    all_vals["month_cos"] = np.cos(2 * np.pi * month / 12)

    # Spatial feature
    dist = np.sqrt(
        ((lat - GANGOTRI_LAT) * 111) ** 2 +
        ((lon - GANGOTRI_LON) * 111 * np.cos(np.radians(lat))) ** 2
    )
    all_vals["dist_from_source_km"] = dist

    # Band ratios
    safe = lambda a, b: a / max(b, 1e-6)
    all_vals["B5_B4"] = safe(bands.get("B5", 0), bands.get("B4", 0))
    all_vals["B3_B2"] = safe(bands.get("B3", 0), bands.get("B2", 0))
    all_vals["B4_B3"] = safe(bands.get("B4", 0), bands.get("B3", 0))
    all_vals["B8_B4"] = safe(bands.get("B8", 0), bands.get("B4", 0))
    all_vals["B11_B8"] = safe(bands.get("B11", 0), bands.get("B8", 0))
    all_vals["B5_B6"] = safe(bands.get("B5", 0), bands.get("B6", 0))

    vector = np.array([all_vals.get(f, 0.0) for f in feature_list]).reshape(1, -1)
    return vector


# ──────────────────────────────────────────────────────────────────────────────
# Predict with uncertainty (RF-based confidence intervals)
# ──────────────────────────────────────────────────────────────────────────────
def predict_with_uncertainty(
    X_scaled: np.ndarray,
    all_models: Dict,
    model_name: str = "Random Forest",
) -> Dict[str, Dict[str, float]]:
    """Predict all WQ targets and estimate uncertainty."""
    predictions = {}

    # BNN path — load from separate file
    if model_name == "BNN":
        bnn_path = MODELS_DIR / "bnn_models.pt"
        if not bnn_path.exists():
            print("BNN models not found. Run bayesian_wq_model.py first.")
            return {}

        # Import BNN classes from training script
        sys.path.insert(0, str(BASE_DIR))
        from bayesian_wq_model import BayesianNeuralNetwork, BayesianLinear, _bnn_mc_predict, BNN_CONFIG

        bnn_models = torch.load(bnn_path, weights_only=False)
        X_t = torch.tensor(X_scaled.astype(np.float32), dtype=torch.float32)

        for target, info in bnn_models.items():
            tname = WQ_SHORT.get(target, target)
            model = info["model"]
            y_mean = info["y_mean"]
            y_std = info["y_std"]

            preds = _bnn_mc_predict(model, X_t, n_samples=BNN_CONFIG["mc_samples"])
            pred_mean = preds.mean(axis=0).squeeze() * y_std + y_mean
            pred_std = preds.std(axis=0).squeeze() * y_std

            predictions[tname] = {
                "predicted": round(float(pred_mean), 4),
                "ci_low": round(float(pred_mean - 1.96 * pred_std), 4),
                "ci_high": round(float(pred_mean + 1.96 * pred_std), 4),
            }
        return predictions

    # Sklearn / XGBoost path
    for target, models in all_models.items():
        tname = WQ_SHORT.get(target, target)
        model = models.get(model_name)
        if model is None:
            continue

        pred = model.predict(X_scaled)[0]

        ci_low, ci_high = None, None
        if model_name == "Random Forest" and hasattr(model, "estimators_"):
            tree_preds = np.array([t.predict(X_scaled)[0] for t in model.estimators_])
            ci_low = np.percentile(tree_preds, 5)
            ci_high = np.percentile(tree_preds, 95)

        predictions[tname] = {
            "predicted": round(float(pred), 4),
            "ci_low": round(float(ci_low), 4) if ci_low is not None else None,
            "ci_high": round(float(ci_high), 4) if ci_high is not None else None,
        }

    return predictions


# ──────────────────────────────────────────────────────────────────────────────
# Display results
# ──────────────────────────────────────────────────────────────────────────────
def print_predictions(predictions: Dict, lat: float, lon: float, date_str: str):
    print(f"\n{'=' * 60}")
    print(f"  WATER QUALITY PREDICTION")
    print(f"  Location: ({lat}, {lon})  Date: {date_str}")
    print(f"{'=' * 60}")
    print(f"  {'Parameter':<12} {'Predicted':>10} {'90% CI':>20}")
    print(f"  {'-'*12} {'-'*10} {'-'*20}")
    for name, vals in predictions.items():
        pred_str = f"{vals['predicted']:>10.3f}"
        if vals["ci_low"] is not None:
            ci_str = f"[{vals['ci_low']:.3f}, {vals['ci_high']:.3f}]"
        else:
            ci_str = "—"
        print(f"  {name:<12} {pred_str} {ci_str:>20}")
    print(f"{'=' * 60}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Predict water quality for a new river")
    parser.add_argument("--lat", type=float, help="Latitude")
    parser.add_argument("--lon", type=float, help="Longitude")
    parser.add_argument("--date", type=str, help="Date (YYYY-MM-DD)")
    parser.add_argument("--image", type=str, help="Path to Sentinel-2 GeoTIFF")
    parser.add_argument("--csv", type=str, help="CSV with lat, lon, date columns")
    parser.add_argument("--model", type=str, default="Random Forest",
                        choices=["Linear Regression", "Random Forest", "XGBoost", "BNN"],
                        help="Which model to use (default: Random Forest)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path for batch predictions")
    args = parser.parse_args()

    scaler, all_models, config = load_artifacts()
    features = config["features"]

    points = []  # list of (lat, lon, date_str, bands_dict)

    if args.csv:
        # Batch mode from CSV
        with open(args.csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    lat = float(row["lat"])
                    lon = float(row["lon"])
                    date_str = row["date"]
                except (KeyError, ValueError):
                    continue
                bands = extract_from_gee(lat, lon, date_str)
                if bands:
                    points.append((lat, lon, date_str, bands))

    elif args.image:
        # GeoTIFF mode
        lat = args.lat or 0.0
        lon = args.lon or 0.0
        date_str = args.date or "2025-01-01"
        bands = extract_from_geotiff(args.image)
        indices = compute_indices_from_bands(bands)
        bands.update(indices)
        points.append((lat, lon, date_str, bands))

    elif args.lat and args.lon and args.date:
        # Single point mode
        bands = extract_from_gee(args.lat, args.lon, args.date)
        if bands:
            points.append((args.lat, args.lon, args.date, bands))
        else:
            print("Could not retrieve satellite data. Exiting.")
            return
    else:
        parser.print_help()
        return

    # Predict for all points
    all_predictions = []
    for lat, lon, date_str, bands in points:
        X = build_feature_vector(bands, lat, lon, date_str, features)
        X_scaled = scaler.transform(X)
        preds = predict_with_uncertainty(X_scaled, all_models, args.model)
        print_predictions(preds, lat, lon, date_str)
        row = {"lat": lat, "lon": lon, "date": date_str}
        for name, vals in preds.items():
            row[f"{name}_predicted"] = vals["predicted"]
            row[f"{name}_ci_low"] = vals["ci_low"]
            row[f"{name}_ci_high"] = vals["ci_high"]
        all_predictions.append(row)

    # Save batch results
    if args.output and all_predictions:
        out_path = Path(args.output)
        header = list(all_predictions[0].keys())
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(all_predictions)
        print(f"\nBatch predictions saved to: {out_path}")


if __name__ == "__main__":
    main()
