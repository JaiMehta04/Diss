"""
train_cnn_wq.py
===============
Hybrid CNN + Tabular model that maps Sentinel-2 **image patches** plus
station metadata directly to 12 water-quality parameters.

Key improvements over v1:
  - On-the-fly spectral indices as extra channels (NDWI, NDVI, NDCI, NDTI, MNDWI)
    → 15 input channels instead of 10
  - Hybrid architecture: CNN spatial branch + tabular branch (month, lat, lon,
    distance from source) merged before the regression head
  - Much lighter CNN (~45K params vs ~500K) to avoid overfitting on ~3,500 samples
  - Log-transform for heavily skewed targets (BOD, COD, Turbidity, EC, TOC)
  - Huber loss (robust to outliers) instead of MSE
  - Gaussian noise augmentation on spectral channels

Architecture
------------
  === CNN branch (spatial features) ===
    Conv(15,32,3) → BN → ReLU → MaxPool
    Conv(32,48,3) → BN → ReLU → MaxPool
    Conv(48,64,3) → BN → ReLU → AdaptiveAvgPool(1)  → (64,)

  === Tabular branch ===
    Linear(5, 32) → BN → ReLU                        → (32,)

  === Merged head ===
    Concat(64 + 32 = 96)
    Linear(96, 64) → BN → ReLU → Dropout
    Linear(64, 12)

Usage
-----
  $env:PYTHONPATH = "C:\\torch_tmp"
  python train_cnn_wq.py --augment
  python train_cnn_wq.py --epochs 300 --lr 5e-4
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure PyTorch is importable
_torch_alt = r"C:\torch_tmp"
if os.path.isdir(_torch_alt) and _torch_alt not in sys.path:
    sys.path.insert(0, _torch_alt)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
import joblib

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).resolve().parents[2]
PATCH_DIR = BASE_DIR / "training_dataset" / "patches"
META_CSV = BASE_DIR / "training_dataset" / "merged_training_dataset.csv"
RESULTS_DIR = BASE_DIR / "training_dataset" / "results_cnn"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"
TABLES_DIR = RESULTS_DIR / "tables"

BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
N_BANDS = len(BANDS)

# On-the-fly spectral indices added as extra channels
INDEX_NAMES = ["NDWI", "NDVI", "NDCI", "NDTI", "MNDWI"]
N_CHANNELS = N_BANDS + len(INDEX_NAMES)   # 10 + 5 = 15

# Tabular features appended alongside image
TABULAR_FEATURES = ["lat", "lon", "month_sin", "month_cos", "dist_from_source_km"]
N_TABULAR = len(TABULAR_FEATURES)

# Targets to log-transform (heavy right skew)
LOG_TARGETS = [
    "wq_BOD",
    "wq_COD",
    "wq_EC",
    "wq_TOC",
    "wq_WTb",
]

GANGOTRI_LAT, GANGOTRI_LON = 30.99, 78.94

WQ_TARGETS = [
    "wq_BOD",
    "wq_COD",
    "wq_CL",
    "wq_EC",
    "wq_Depth",
    "wq_DO",
    "wq_NO3",
    "wq_TOC",
    "wq_S",
    "wq_WT",
    "wq_WTb",
    "wq_pH",
]

WQ_SHORT = {
    "wq_BOD": "BOD",
    "wq_COD": "COD",
    "wq_CL": "Cl",
    "wq_EC": "EC",
    "wq_Depth": "Depth",
    "wq_DO": "DO",
    "wq_NO3": "NO3",
    "wq_TOC": "TOC",
    "wq_S": "WL",
    "wq_WT": "Temp",
    "wq_WTb": "Turb",
    "wq_pH": "pH",
}

PRIORITY_TARGETS = [
    "wq_BOD",
    "wq_DO",
    "wq_WTb",
    "wq_pH",
    "wq_COD",
    "wq_EC",
]

N_TARGETS = len(WQ_TARGETS)
SEED = 42
DPI = 300

np.random.seed(SEED)
torch.manual_seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# SPECTRAL INDICES (computed on-the-fly from 10-band patches)
# ══════════════════════════════════════════════════════════════════════════════
def _safe_nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Normalised difference: (a-b)/(a+b+1e-8), NaN/Inf-safe."""
    bad = np.isnan(a) | np.isnan(b) | np.isinf(a) | np.isinf(b)
    num = np.where(bad, 0.0, a - b)
    den = np.where(bad, 1.0, a + b + 1e-8)
    return num / den


def compute_indices_patch(patch: np.ndarray) -> np.ndarray:
    """Given (10, H, W) raw-band patch → (15, H, W) with 5 index channels appended.

    Band order: B2(0), B3(1), B4(2), B5(3), B6(4), B7(5), B8(6), B8A(7), B11(8), B12(9)
    Indices:    NDWI = (B3-B8)/(B3+B8)
               NDVI = (B8-B4)/(B8+B4)
               NDCI = (B5-B4)/(B5+B4)
               NDTI = (B4-B3)/(B4+B3)
               MNDWI = (B3-B11)/(B3+B11)
    """
    B3, B4, B5, B8, B11 = patch[1], patch[2], patch[3], patch[6], patch[8]
    ndwi  = _safe_nd(B3, B8)
    ndvi  = _safe_nd(B8, B4)
    ndci  = _safe_nd(B5, B4)
    ndti  = _safe_nd(B4, B3)
    mndwi = _safe_nd(B3, B11)
    return np.concatenate([patch, ndwi[None], ndvi[None], ndci[None],
                           ndti[None], mndwi[None]], axis=0)  # (15, H, W)


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════
class PatchDataset(Dataset):
    """Loads (patch, tabular, target) triples.

    patch  : (15, H, W)  — 10 bands + 5 spectral indices
    tabular: (5,)         — lat, lon, month_sin, month_cos, dist_from_source
    target : (12,)        — WQ parameters (z-normalised, some log-transformed)
    """

    def __init__(
        self,
        indices: np.ndarray,
        patch_files: List[Path],
        targets: np.ndarray,
        tabular: np.ndarray,
        augment: bool = False,
        band_mean: np.ndarray = None,
        band_std: np.ndarray = None,
    ):
        self.indices = indices
        self.patch_files = patch_files
        self.targets = targets
        self.tabular = tabular       # (N, N_TABULAR) already z-scored
        self.augment = augment
        self.band_mean = band_mean   # (N_CHANNELS=15,)
        self.band_std = band_std

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        patch = np.load(self.patch_files[idx]).astype(np.float32)  # (10, H, W)
        np.nan_to_num(patch, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # Compute spectral indices → (15, H, W)
        patch = compute_indices_patch(patch)

        # Per-channel normalisation
        if self.band_mean is not None:
            for c in range(patch.shape[0]):
                patch[c] = (patch[c] - self.band_mean[c]) / (self.band_std[c] + 1e-8)

        target = self.targets[idx].astype(np.float32)
        tab = self.tabular[idx].astype(np.float32)

        # Data augmentation
        if self.augment:
            if np.random.rand() > 0.5:
                patch = patch[:, ::-1, :].copy()
            if np.random.rand() > 0.5:
                patch = patch[:, :, ::-1].copy()
            k = np.random.randint(0, 4)
            if k > 0:
                patch = np.rot90(patch, k, axes=(1, 2)).copy()
            # Gaussian noise on spectral channels (σ=0.02)
            patch = patch + np.random.normal(0, 0.02, patch.shape).astype(np.float32)

        return torch.from_numpy(patch), torch.from_numpy(tab), torch.from_numpy(target)


# ══════════════════════════════════════════════════════════════════════════════
# HYBRID CNN + TABULAR MODEL
# ══════════════════════════════════════════════════════════════════════════════
class ConvBlock(nn.Module):
    """Conv2d → BN → ReLU."""
    def __init__(self, in_ch, out_ch, kernel=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class HybridWQModel(nn.Module):
    """Lightweight CNN + tabular branch → multi-output WQ regression.

    CNN branch  : (B, 15, H, W) → 64-d feature vector  (~45K params)
    Tabular branch: (B, 5) → 32-d feature vector
    Merged head : 96-d → 64 → 12 outputs
    """

    def __init__(self, n_channels: int = N_CHANNELS, n_tabular: int = N_TABULAR,
                 n_targets: int = N_TARGETS, dropout: float = 0.25):
        super().__init__()

        # --- CNN branch (lightweight) ---
        self.cnn = nn.Sequential(
            ConvBlock(n_channels, 32),
            nn.MaxPool2d(2),
            ConvBlock(32, 48),
            nn.MaxPool2d(2),
            ConvBlock(48, 64),
            nn.AdaptiveAvgPool2d(1),  # (64, 1, 1)
            nn.Flatten(),             # (64,)
        )

        # --- Tabular branch ---
        self.tab = nn.Sequential(
            nn.Linear(n_tabular, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        # --- Merged head ---
        self.head = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_targets),
        )

    def forward(self, patch, tabular):
        cnn_feat = self.cnn(patch)      # (B, 64)
        tab_feat = self.tab(tabular)    # (B, 32)
        merged = torch.cat([cnn_feat, tab_feat], dim=1)  # (B, 96)
        return self.head(merged)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_data():
    """Load training dataset, filter to rows with patches, impute, log-transform, normalise.

    Returns dict with keys:
      patch_files, Y_raw, Y_norm, y_means, y_stds,
      log_mask, station_ids, tabular, tab_mean, tab_std, df
    """
    print("=" * 70)
    print("LOADING PATCH DATA")
    print("=" * 70)

    df = pd.read_csv(META_CSV)
    # Keep only rows that have a patch file recorded
    df = df[df["patch_file"].notna()].reset_index(drop=True)
    print(f"  Rows with patches: {len(df)}")

    # Verify files exist
    patch_files = []
    valid_mask = []
    for _, row in df.iterrows():
        p = PATCH_DIR / row["patch_file"]
        if p.exists():
            patch_files.append(p)
            valid_mask.append(True)
        else:
            valid_mask.append(False)
    df = df[valid_mask].reset_index(drop=True)
    print(f"  Files confirmed on disk: {len(df)}")

    # Parse WQ targets
    for col in WQ_TARGETS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Impute missing WQ
    imputer = KNNImputer(n_neighbors=5)
    df[WQ_TARGETS] = pd.DataFrame(
        imputer.fit_transform(df[WQ_TARGETS]),
        columns=WQ_TARGETS, index=df.index,
    )
    mask = df[WQ_TARGETS].notna().all(axis=1)
    df = df[mask].reset_index(drop=True)
    patch_files = [patch_files[i] for i in df.index]
    print(f"  After WQ cleaning: {len(df)} samples")

    # --- Tabular features ---
    df["date_parsed"] = pd.to_datetime(df["date"])
    df["month"] = df["date_parsed"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dist_from_source_km"] = np.sqrt(
        ((df["lat"] - GANGOTRI_LAT) * 111) ** 2 +
        ((df["lon"] - GANGOTRI_LON) * 111 * np.cos(np.radians(df["lat"]))) ** 2
    )
    tab_raw = df[TABULAR_FEATURES].values.astype(np.float32)
    tab_mean = tab_raw.mean(axis=0)
    tab_std = tab_raw.std(axis=0) + 1e-8
    tabular = (tab_raw - tab_mean) / tab_std
    print(f"  Tabular features: {TABULAR_FEATURES}")

    # --- Log-transform skewed targets ---
    log_mask = np.array([t in LOG_TARGETS for t in WQ_TARGETS])  # (12,) bool
    Y_raw = df[WQ_TARGETS].values.astype(np.float32)
    Y_work = Y_raw.copy()
    for j in range(N_TARGETS):
        if log_mask[j]:
            Y_work[:, j] = np.log1p(np.clip(Y_work[:, j], 0, None))
    print(f"  Log-transformed: {[WQ_SHORT[t] for t in LOG_TARGETS]}")

    # z-score normalise
    y_means = Y_work.mean(axis=0)
    y_stds = Y_work.std(axis=0) + 1e-8
    Y_norm = (Y_work - y_means) / y_stds

    station_ids = df["stationId"].values

    return {
        "patch_files": patch_files,
        "Y_raw": Y_raw,
        "Y_norm": Y_norm,
        "y_means": y_means,
        "y_stds": y_stds,
        "log_mask": log_mask,
        "station_ids": station_ids,
        "tabular": tabular,
        "tab_mean": tab_mean,
        "tab_std": tab_std,
        "df": df,
    }


def compute_band_stats(patch_files: List[Path], max_samples: int = 500):
    """Compute per-channel mean/std over 15 channels (10 bands + 5 indices)."""
    n = min(len(patch_files), max_samples)
    chosen = np.random.choice(len(patch_files), n, replace=False)

    running_sum = np.zeros(N_CHANNELS, dtype=np.float64)
    running_sq = np.zeros(N_CHANNELS, dtype=np.float64)
    total_pixels = 0

    for i in chosen:
        raw = np.load(patch_files[i]).astype(np.float64)
        np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        patch = compute_indices_patch(raw)  # (15, H, W)
        np.nan_to_num(patch, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        npx = patch.shape[1] * patch.shape[2]
        for c in range(N_CHANNELS):
            running_sum[c] += patch[c].sum()
            running_sq[c] += (patch[c] ** 2).sum()
        total_pixels += npx

    band_mean = (running_sum / total_pixels).astype(np.float32)
    band_std = np.sqrt(
        running_sq / total_pixels - band_mean.astype(np.float64) ** 2
    ).astype(np.float32)
    band_std = np.maximum(band_std, 1e-6)

    print(f"  Channel means (15): {np.round(band_mean, 5)}")
    print(f"  Channel stds  (15): {np.round(band_std, 5)}")
    return band_mean, band_std


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════
def train_one(
    model: HybridWQModel,
    train_loader: DataLoader,
    cfg: Dict,
    label: str = "",
) -> HybridWQModel:
    """Train one model instance, return best state."""

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=1e-6,
    )
    criterion = nn.HuberLoss(delta=1.0)  # robust to outliers
    device = next(model.parameters()).device

    best_loss = float("inf")
    best_state = None
    patience_ctr = 0

    for epoch in range(cfg["epochs"]):
        model.train()
        epoch_loss = 0.0

        for xb, tab, yb in train_loader:
            xb, tab, yb = xb.to(device), tab.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb, tab)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(train_loader.dataset)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"\r    {label} Epoch {epoch+1:3d}/{cfg['epochs']}  "
                f"loss={epoch_loss:.6f}  lr={lr:.1e}  best={best_loss:.6f}",
                end="", flush=True,
            )

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= cfg["patience"]:
                print(
                    f"\r    {label} Early stop epoch {epoch+1}  "
                    f"best_loss={best_loss:.6f}                         "
                )
                break

    if best_state:
        model.load_state_dict(best_state)
    if patience_ctr < cfg["patience"]:
        print(
            f"\r    {label} Finished {cfg['epochs']} epochs  "
            f"best_loss={best_loss:.6f}                              "
        )
    return model


# ══════════════════════════════════════════════════════════════════════════════
# CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
def _denorm_preds(Y_pred_norm, y_means, y_stds, log_mask):
    """Reverse z-score + log1p to get raw-scale predictions."""
    Y = Y_pred_norm * y_stds + y_means
    for j in range(Y.shape[1]):
        if log_mask[j]:
            Y[:, j] = np.expm1(Y[:, j])
    return Y


def cross_validate(data, band_mean, band_std, cfg) -> Tuple[Dict, np.ndarray]:
    print("\n" + "=" * 70)
    print("SPATIAL CROSS-VALIDATION (Leave-Station-Group-Out, Hybrid CNN)")
    print("=" * 70)

    patch_files = data["patch_files"]
    Y_norm = data["Y_norm"]
    Y_raw = data["Y_raw"]
    y_means, y_stds = data["y_means"], data["y_stds"]
    log_mask = data["log_mask"]
    station_ids = data["station_ids"]
    tabular = data["tabular"]

    n_stations = len(np.unique(station_ids))
    n_folds = min(n_stations, 10)
    gkf = GroupKFold(n_splits=n_folds)

    Y_pred_all = np.full_like(Y_norm, np.nan)
    all_indices = np.arange(len(patch_files))

    for fold_i, (tr_idx, te_idx) in enumerate(
        gkf.split(all_indices, Y_norm, station_ids), 1
    ):
        t0 = time.time()
        print(f"\n  Fold {fold_i}/{n_folds}  "
              f"(train={len(tr_idx)}, test={len(te_idx)})")

        train_ds = PatchDataset(
            tr_idx, patch_files, Y_norm, tabular,
            augment=cfg["augment"],
            band_mean=band_mean, band_std=band_std,
        )
        test_ds = PatchDataset(
            te_idx, patch_files, Y_norm, tabular,
            augment=False,
            band_mean=band_mean, band_std=band_std,
        )
        train_loader = DataLoader(
            train_ds, batch_size=cfg["batch_size"], shuffle=True,
            num_workers=0, pin_memory=False,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = HybridWQModel(
            n_channels=N_CHANNELS, n_tabular=N_TABULAR,
            n_targets=N_TARGETS, dropout=cfg["dropout"],
        ).to(device)
        model = train_one(model, train_loader, cfg, label=f"Fold {fold_i}/{n_folds}")

        # Evaluate
        model.eval()
        test_loader = DataLoader(
            test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0,
        )
        preds = []
        with torch.no_grad():
            for xb, tab, _ in test_loader:
                xb, tab = xb.to(device), tab.to(device)
                preds.append(model(xb, tab).cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        Y_pred_all[te_idx] = preds

        elapsed = time.time() - t0
        print(f"    Fold {fold_i} done in {elapsed:.1f}s")

    # Denormalize + reverse log-transform
    Y_pred_denorm = _denorm_preds(Y_pred_all, y_means, y_stds, log_mask)

    print("\n" + "=" * 70)
    print("CROSS-VALIDATION RESULTS (Hybrid CNN + Tabular)")
    print("=" * 70)
    print(f"  {'Parameter':<8} {'R²':>8} {'RMSE':>10} {'MAE':>10}")
    print(f"  {'-'*8} {'-'*8} {'-'*10} {'-'*10}")

    results = {}
    for i, target in enumerate(WQ_TARGETS):
        tname = WQ_SHORT[target]
        valid = ~np.isnan(Y_pred_denorm[:, i]) & ~np.isinf(Y_pred_denorm[:, i])
        if valid.sum() < 2:
            print(f"  {tname:<8} {'N/A':>8} {'N/A':>10} {'N/A':>10}  (no valid predictions)")
            results[target] = {"R2": float('nan'), "RMSE": float('nan'), "MAE": float('nan')}
            continue
        r2 = r2_score(Y_raw[valid, i], Y_pred_denorm[valid, i])
        rmse = np.sqrt(mean_squared_error(Y_raw[valid, i], Y_pred_denorm[valid, i]))
        mae = mean_absolute_error(Y_raw[valid, i], Y_pred_denorm[valid, i])
        results[target] = {"R2": r2, "RMSE": rmse, "MAE": mae}
        print(f"  {tname:<8} {r2:>8.4f} {rmse:>10.4f} {mae:>10.4f}")

    return results, Y_pred_denorm


# ══════════════════════════════════════════════════════════════════════════════
# FINAL MODEL
# ══════════════════════════════════════════════════════════════════════════════
def train_final(data, band_mean, band_std, cfg) -> HybridWQModel:
    print("\n" + "=" * 70)
    print("TRAINING FINAL HYBRID CNN ON ALL DATA")
    print("=" * 70)

    patch_files = data["patch_files"]
    Y_norm = data["Y_norm"]
    tabular = data["tabular"]

    all_idx = np.arange(len(patch_files))
    ds = PatchDataset(
        all_idx, patch_files, Y_norm, tabular,
        augment=cfg["augment"],
        band_mean=band_mean, band_std=band_std,
    )
    loader = DataLoader(
        ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridWQModel(
        n_channels=N_CHANNELS, n_tabular=N_TABULAR,
        n_targets=N_TARGETS, dropout=cfg["dropout"],
    ).to(device)
    model = train_one(model, loader, cfg, label="Final")

    # Save
    save_bundle = {
        "model_state": model.state_dict(),
        "n_channels": N_CHANNELS,
        "n_tabular": N_TABULAR,
        "n_targets": N_TARGETS,
        "dropout": cfg["dropout"],
        "band_mean": band_mean.tolist(),
        "band_std": band_std.tolist(),
        "y_means": data["y_means"].tolist(),
        "y_stds": data["y_stds"].tolist(),
        "log_mask": data["log_mask"].tolist(),
        "tab_mean": data["tab_mean"].tolist(),
        "tab_std": data["tab_std"].tolist(),
        "bands": BANDS,
        "index_names": INDEX_NAMES,
        "tabular_features": TABULAR_FEATURES,
        "targets": WQ_TARGETS,
        "target_short": WQ_SHORT,
    }
    save_path = MODELS_DIR / "cnn_wq.pt"
    torch.save(save_bundle, save_path)
    print(f"\n  Saved: {save_path}")

    return model


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════
def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [FIG] {name}.png")


def plot_actual_vs_predicted(Y_true_raw, Y_pred, results):
    print("\n  --- Hybrid CNN Actual vs Predicted ---")
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    for i, target in enumerate(PRIORITY_TARGETS):
        ax = axes.ravel()[i]
        idx = WQ_TARGETS.index(target)
        tname = WQ_SHORT[target]
        r2 = results[target]["R2"]

        yt = Y_true_raw[:, idx]
        yp = Y_pred[:, idx]
        valid = ~np.isnan(yp)

        ax.scatter(yt[valid], yp[valid], alpha=0.3, s=10, c="teal")
        lims = [min(yt[valid].min(), yp[valid].min()),
                max(yt[valid].max(), yp[valid].max())]
        ax.plot(lims, lims, "r--", lw=2, label="Perfect fit")
        ax.set_xlabel(f"Actual {tname}")
        ax.set_ylabel(f"Predicted {tname}")
        ax.set_title(f"{tname}  (R²={r2:.4f})", fontweight="bold")
        ax.legend(fontsize=9)

    fig.suptitle("Hybrid CNN+Tabular: Actual vs Predicted (Spatial CV)",
                 fontsize=14, fontweight="bold", y=1.01)
    save_fig(fig, "cnn_actual_vs_predicted")


def plot_r2_bars(results):
    print("\n  --- Hybrid CNN R² Bar Chart ---")
    names = [WQ_SHORT[t] for t in WQ_TARGETS]
    r2s = [results[t]["R2"] for t in WQ_TARGETS]

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#4CAF50" if r > 0.5 else "#FF9800" if r > 0 else "#F44336" for r in r2s]
    ax.bar(names, r2s, color=colors, edgecolor="white")
    ax.set_ylabel("R²")
    ax.set_title("Hybrid CNN+Tabular R² (Spatial CV)", fontweight="bold")
    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=45)
    save_fig(fig, "cnn_r2_bars")


def plot_residuals(Y_true_raw, Y_pred, results):
    print("\n  --- CNN Residuals ---")
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    for i, target in enumerate(PRIORITY_TARGETS):
        ax = axes.ravel()[i]
        idx = WQ_TARGETS.index(target)
        tname = WQ_SHORT[target]

        yt = Y_true_raw[:, idx]
        yp = Y_pred[:, idx]
        valid = ~np.isnan(yp)
        resid = yt[valid] - yp[valid]

        ax.scatter(yp[valid], resid, alpha=0.3, s=10, c="teal")
        ax.axhline(0, color="red", ls="--")
        ax.set_xlabel(f"Predicted {tname}")
        ax.set_ylabel("Residual")
        ax.set_title(f"{tname} Residuals", fontweight="bold")

    fig.suptitle("Hybrid CNN+Tabular Residuals", fontsize=14, fontweight="bold", y=1.01)
    save_fig(fig, "cnn_residuals")


def save_results_csv(results):
    rows = []
    for target in WQ_TARGETS:
        r = results[target]
        rows.append({
            "Parameter": WQ_SHORT[target],
            "R2": round(r["R2"], 4),
            "RMSE": round(r["RMSE"], 4),
            "MAE": round(r["MAE"], 4),
        })
    pd.DataFrame(rows).to_csv(TABLES_DIR / "cnn_results.csv", index=False)
    print(f"  Saved: cnn_results.csv")


def save_latex_table(results):
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Hybrid CNN+Tabular Performance (Image Patches, Spatial CV)}",
        r"\label{tab:cnn_results}",
        r"\begin{tabular}{lrrr}",
        r"\hline",
        r"\textbf{Parameter} & \textbf{R\textsuperscript{2}} & \textbf{RMSE} & \textbf{MAE} \\",
        r"\hline",
    ]
    for target in WQ_TARGETS:
        r = results[target]
        tname = WQ_SHORT[target]
        lines.append(
            f"  {tname} & {r['R2']:.4f} & {r['RMSE']:.4f} & {r['MAE']:.4f} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    (TABLES_DIR / "cnn_results.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved: cnn_results.tex")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Train Hybrid CNN+Tabular on patches")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--augment", action="store_true",
                        help="Enable random flip/rotate/noise augmentation")
    args = parser.parse_args()

    cfg = {
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "dropout": args.dropout,
        "weight_decay": 1e-4,
        "augment": args.augment,
    }

    for d in [RESULTS_DIR, FIGURES_DIR, MODELS_DIR, TABLES_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    if not META_CSV.exists():
        print(f"ERROR: {META_CSV} not found.")
        print("Run download_patches.py first to download image patches.")
        sys.exit(1)

    data = load_data()

    # ── Compute per-channel normalisation (15 channels) ──
    band_mean, band_std = compute_band_stats(data["patch_files"])

    # Save stats for later inference
    json.dump(
        {
            "band_mean": band_mean.tolist(),
            "band_std": band_std.tolist(),
            "y_means": data["y_means"].tolist(),
            "y_stds": data["y_stds"].tolist(),
            "log_mask": data["log_mask"].tolist(),
            "tab_mean": data["tab_mean"].tolist(),
            "tab_std": data["tab_std"].tolist(),
        },
        (MODELS_DIR / "cnn_norm_stats.json").open("w"),
        indent=2,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\n  [GPU DETECTED] {torch.cuda.get_device_name(0)}")
        print(f"    VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"    CUDA: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        print("\n  [NO GPU DETECTED] Running on CPU (training will be slower)")
    print(f"  Config: {cfg}")
    print(f"  Patches: {len(data['patch_files'])}  Channels: {N_CHANNELS} (10 bands + 5 indices)")
    print(f"  Tabular: {N_TABULAR} features")
    print(f"  Targets: {N_TARGETS}  (log-transformed: {sum(data['log_mask'])})")

    # ── Cross-validate ──
    results, Y_pred = cross_validate(data, band_mean, band_std, cfg)

    # ── Train final ──
    final_model = train_final(data, band_mean, band_std, cfg)

    # ── Plots & tables ──
    Y_true_raw = data["Y_raw"]
    plot_actual_vs_predicted(Y_true_raw, Y_pred, results)
    plot_r2_bars(results)
    plot_residuals(Y_true_raw, Y_pred, results)
    save_results_csv(results)
    save_latex_table(results)

    print("\n" + "=" * 70)
    print("HYBRID CNN+TABULAR TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Model : {MODELS_DIR / 'cnn_wq.pt'}")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"  Tables : {TABLES_DIR}")


if __name__ == "__main__":
    main()
