"""
train_bnn.py
============
Train a single multi-output Bayesian Neural Network that maps
Sentinel-2 features → ALL 12 water quality parameters simultaneously.

Architecture: Bayes by Backprop with local reparameterization trick.
  - Gaussian posteriors q(w) = N(mu, softplus(rho)^2) on every weight
  - ELBO loss = MSE + (1/N)*KL(q || prior)
  - Inference: T Monte-Carlo forward passes → mean + uncertainty

Inspired by Qadir & Bilgin (2023), "Hyperspectral Images Classification
with Deep Bayesian Neural Networks", IJACEN.

Usage
-----
  $env:PYTHONPATH = "C:\\torch_tmp"
  python train_bnn.py                      # train + evaluate
  python train_bnn.py --epochs 500         # override epochs
  python train_bnn.py --fresh              # ignore checkpoint, retrain
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# ── Ensure PyTorch is importable ──────────────────────────────────────────────
_torch_alt = r"C:\torch_tmp"
if os.path.isdir(_torch_alt) and _torch_alt not in sys.path:
    sys.path.insert(0, _torch_alt)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import joblib

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "training_dataset" / "merged_training_dataset.csv"
RESULTS_DIR = BASE_DIR / "training_dataset" / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"
TABLES_DIR = RESULTS_DIR / "tables"

BAND_COLS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
INDEX_COLS = ["NDWI", "MNDWI", "NDTI", "NDVI", "NDCI", "MCI", "FAI", "SABI"]
SAT_FEATURES = BAND_COLS + INDEX_COLS

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

# Targets with heavy right skew → log1p transform before z-norm
LOG_TARGETS = ["wq_BOD", "wq_COD", "wq_EC", "wq_TOC", "wq_WTb", "wq_NO3", "wq_CL"]

SEED = 42
DPI = 300
np.random.seed(SEED)
torch.manual_seed(SEED)

N_TARGETS = len(WQ_TARGETS)


# ══════════════════════════════════════════════════════════════════════════════
# BAYESIAN LINEAR LAYER  (Local Reparameterization Trick)
# ══════════════════════════════════════════════════════════════════════════════
class BayesianLinear(nn.Module):
    """Linear layer with Gaussian weight posteriors.

    q(w) = N(mu, softplus(rho)^2),  prior p(w) = N(0, prior_sigma^2)
    Uses local reparameterization: sample activations, not weights.
    """

    def __init__(self, in_features: int, out_features: int,
                 prior_sigma: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma

        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), -3.0))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.full((out_features,), -3.0))

        nn.init.kaiming_normal_(self.weight_mu, nonlinearity="relu")
        self.kl_divergence = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_sigma = F.softplus(self.weight_rho)
        b_sigma = F.softplus(self.bias_rho)

        # Local reparameterization: act ~ N(x@mu, sqrt(x^2 @ sigma^2))
        act_mu = F.linear(x, self.weight_mu, self.bias_mu)
        act_var = F.linear(x.pow(2), w_sigma.pow(2), b_sigma.pow(2))
        act_std = torch.sqrt(act_var + 1e-8)

        eps = torch.randn_like(act_mu)
        activation = act_mu + act_std * eps

        # Closed-form KL divergence
        self.kl_divergence = self._kl(
            self.weight_mu, w_sigma, self.bias_mu, b_sigma
        )
        return activation

    def _kl(self, w_mu, w_sigma, b_mu, b_sigma):
        """KL(q || prior) summed over all parameters."""
        var_p = self.prior_sigma ** 2
        kl_w = 0.5 * (
            w_sigma.pow(2) / var_p
            + w_mu.pow(2) / var_p
            - 1.0
            + np.log(var_p) - torch.log(w_sigma.pow(2))
        ).sum()
        kl_b = 0.5 * (
            b_sigma.pow(2) / var_p
            + b_mu.pow(2) / var_p
            - 1.0
            + np.log(var_p) - torch.log(b_sigma.pow(2))
        ).sum()
        return kl_w + kl_b


# ══════════════════════════════════════════════════════════════════════════════
# MULTI-OUTPUT BNN MODEL
# ══════════════════════════════════════════════════════════════════════════════
class MultiOutputBNN(nn.Module):
    """Single BNN: satellite features → ALL 12 WQ parameters at once.

    Architecture with BatchNorm for stable training:
      Input(n_features)
        → BayesLinear(512) → BatchNorm → LeakyReLU → Dropout
        → BayesLinear(256) → BatchNorm → LeakyReLU → Dropout
        → BayesLinear(128) → BatchNorm → LeakyReLU → Dropout
        → BayesLinear(64)  → BatchNorm → LeakyReLU
        → BayesLinear(12)  ← 12 WQ outputs
    """

    def __init__(self, n_features: int, n_targets: int = N_TARGETS,
                 hidden_dims: Tuple = (512, 256, 128, 64),
                 dropout: float = 0.05, prior_sigma: float = 1.0):
        super().__init__()
        layers = []
        prev = n_features
        for i, h in enumerate(hidden_dims):
            layers.append(BayesianLinear(prev, h, prior_sigma=prior_sigma))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.LeakyReLU(0.1))
            # Only apply dropout on first few layers
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(BayesianLinear(prev, n_targets, prior_sigma=prior_sigma))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def kl_loss(self) -> torch.Tensor:
        kl = torch.tensor(0.0, device=next(self.parameters()).device)
        for m in self.modules():
            if isinstance(m, BayesianLinear):
                kl = kl + m.kl_divergence
        return kl


class ELBOLoss(nn.Module):
    """ELBO = NLL + kl_weight * KL.

    kl_weight is controlled externally via KL annealing.
    """

    def __init__(self, n_train: int):
        super().__init__()
        self.n_train = n_train

    def forward(self, pred, true, kl, kl_weight: float = 1.0):
        nll = F.mse_loss(pred, true, reduction="mean")
        return nll + (kl_weight / self.n_train) * kl


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
def load_and_prepare() -> Tuple[pd.DataFrame, List[str]]:
    print("=" * 70)
    print("LOADING DATA & ENGINEERING FEATURES")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df["stationId"] = df["stationId"].astype(str)
    print(f"  Raw dataset: {df.shape[0]} rows, {df['stationId'].nunique()} stations")

    for col in WQ_TARGETS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── STEP 1: Replace -999 sentinel values with NaN ──
    # RTWQMS uses -999 as a missing data marker
    for col in WQ_TARGETS:
        n_sentinel = (df[col] <= -900).sum()
        if n_sentinel > 0:
            df.loc[df[col] <= -900, col] = np.nan
            print(f"    {col}: replaced {n_sentinel} sentinel values (-999) with NaN")

    # ── STEP 2: Remove physically impossible values ──
    PHYSICAL_BOUNDS = {
        "wq_pH": (0, 14), "wq_DO": (0, 25), "wq_BOD": (0, 500),
        "wq_COD": (0, 2000), "wq_EC": (0, 5000), "wq_WT": (0, 50),
        "wq_WTb": (0, 5000), "wq_NO3": (0, 500), "wq_TOC": (0, 500),
        "wq_CL": (0, 5000), "wq_Depth": (0, 50), "wq_S": (0, 1000),
    }
    for col, (lo, hi) in PHYSICAL_BOUNDS.items():
        if col in df.columns:
            invalid = (df[col] < lo) | (df[col] > hi)
            n_invalid = invalid.sum()
            if n_invalid > 0:
                df.loc[invalid, col] = np.nan
                print(f"    {col}: removed {n_invalid} out-of-bounds values")

    # ── STEP 3: Clip remaining outliers to 2nd-98th percentile ──
    for col in WQ_TARGETS:
        valid_vals = df[col].dropna()
        if len(valid_vals) > 100:
            lo, hi = valid_vals.quantile([0.02, 0.98]).values
            n_clipped = ((df[col] < lo) | (df[col] > hi)).sum()
            df[col] = df[col].clip(lo, hi)
            if n_clipped > 0:
                print(f"    {col}: clipped {n_clipped} values to [{lo:.2f}, {hi:.2f}]")

    # ── STEP 4: Drop rows with too many NaN targets ──
    # Keep rows that have at least 8 of 12 WQ targets
    n_valid_targets = df[WQ_TARGETS].notna().sum(axis=1)
    df = df[n_valid_targets >= 8].reset_index(drop=True)
    print(f"  After quality filter (>=8/12 targets valid): {len(df)} rows")

    # ── STEP 5: Deduplicate by averaging WQ for same satellite observation ──
    # Many consecutive days share the same satellite composite.
    # Group by station + satellite bands and average the WQ targets.
    sat_key_cols = ["stationId"] + BAND_COLS
    n_before = len(df)
    agg_dict = {col: "mean" for col in WQ_TARGETS}
    agg_dict.update({col: "first" for col in INDEX_COLS})
    agg_dict["date"] = "first"
    agg_dict["lat"] = "first"
    agg_dict["lon"] = "first"
    if "station_no" in df.columns:
        agg_dict["station_no"] = "first"

    df = df.groupby(sat_key_cols, as_index=False).agg(agg_dict)
    print(f"  Deduplicated: {n_before} → {len(df)} rows (averaged WQ per satellite obs)")

    # ── STEP 6: KNN impute remaining NaN targets ──
    imputer = KNNImputer(n_neighbors=5)
    df[WQ_TARGETS] = pd.DataFrame(
        imputer.fit_transform(df[WQ_TARGETS]),
        columns=WQ_TARGETS, index=df.index,
    )
    df = df.dropna(subset=WQ_TARGETS).reset_index(drop=True)

    # ── STEP 7: Feature engineering ──
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    gangotri_lat, gangotri_lon = 30.99, 78.94
    df["dist_from_source_km"] = np.sqrt(
        ((df["lat"] - gangotri_lat) * 111) ** 2 +
        ((df["lon"] - gangotri_lon) * 111 * np.cos(np.radians(df["lat"]))) ** 2
    )

    df["B5_B4"] = df["B5"] / df["B4"].clip(lower=1e-6)
    df["B3_B2"] = df["B3"] / df["B2"].clip(lower=1e-6)
    df["B4_B3"] = df["B4"] / df["B3"].clip(lower=1e-6)
    df["B8_B4"] = df["B8"] / df["B4"].clip(lower=1e-6)
    df["B11_B8"] = df["B11"] / df["B8"].clip(lower=1e-6)
    df["B5_B6"] = df["B5"] / df["B6"].clip(lower=1e-6)

    engineered = [
        "month_sin", "month_cos", "dist_from_source_km",
        "B5_B4", "B3_B2", "B4_B3", "B8_B4", "B11_B8", "B5_B6",
    ]
    features = SAT_FEATURES + engineered

    # Remove features with >0.95 inter-correlation
    corr = df[features].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = [c for c in upper.columns if any(upper[c] > 0.95)]
    if drop:
        print(f"  Dropping correlated features: {drop}")
        features = [f for f in features if f not in drop]

    # ── STEP 8: Print final data summary ──
    print(f"\n  Final dataset: {len(df)} rows, {len(features)} features → {N_TARGETS} targets")
    print(f"  Features: {features}")
    print(f"  Target stats after cleaning:")
    for col in WQ_TARGETS:
        vals = df[col]
        print(f"    {WQ_SHORT[col]:6s}: mean={vals.mean():8.2f}  std={vals.std():8.2f}  "
              f"[{vals.min():.2f} – {vals.max():.2f}]")

    return df, features


# ══════════════════════════════════════════════════════════════════════════════
# MC INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
def mc_predict(model: MultiOutputBNN, X: torch.Tensor,
               n_samples: int = 50) -> np.ndarray:
    """Run T stochastic forward passes → (T, N, n_targets)."""
    model.eval()
    # Re-enable dropout for MC sampling
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            preds.append(model(X).cpu().numpy())
    return np.array(preds)  # (T, N, 12)


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP (one fold or final)
# ══════════════════════════════════════════════════════════════════════════════
def _kl_annealing_weight(epoch: int, warmup_epochs: int) -> float:
    """Linear KL warm-up: 0 → 1 over warmup_epochs, then stay at 1.

    This is critical for BNNs.  During early epochs the network learns
    to fit the data (MSE dominates).  The KL term is gradually introduced
    so it regularises without preventing initial learning.
    """
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    return 1.0


def train_one_model(
    X_train: np.ndarray, Y_train: np.ndarray,
    n_features: int, cfg: Dict,
    label: str = "",
) -> MultiOutputBNN:
    """Train a single MultiOutputBNN and return it."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    scaler_amp = torch.amp.GradScaler(enabled=use_amp)

    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    Y_t = torch.tensor(Y_train, dtype=torch.float32, device=device)

    model = MultiOutputBNN(
        n_features=n_features, n_targets=N_TARGETS,
        hidden_dims=cfg["hidden_dims"], dropout=cfg["dropout"],
        prior_sigma=cfg["prior_sigma"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6
    )
    criterion = ELBOLoss(n_train=len(X_train))
    loader = DataLoader(
        TensorDataset(X_t, Y_t), batch_size=cfg["batch_size"], shuffle=True,
        drop_last=False,
    )

    warmup_epochs = cfg.get("kl_warmup", 50)
    best_nll = float("inf")
    best_state = None
    patience_ctr = 0

    for epoch in range(cfg["epochs"]):
        kl_w = _kl_annealing_weight(epoch, warmup_epochs)
        model.train()
        epoch_nll = 0.0
        epoch_kl = 0.0
        n_samples = 0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                pred = model(xb)
                kl = model.kl_loss()
                loss = criterion(pred, yb, kl, kl_weight=kl_w)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            with torch.no_grad():
                epoch_nll += F.mse_loss(pred, yb, reduction="sum").item()
                epoch_kl += kl.item() * xb.size(0)
            n_samples += xb.size(0)

        epoch_nll /= n_samples
        epoch_kl /= n_samples
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(
                f"\r    {label} Epoch {epoch+1:3d}/{cfg['epochs']}  "
                f"NLL={epoch_nll:.6f}  KL={epoch_kl:.2f}  "
                f"kl_w={kl_w:.3f}  lr={lr:.1e}  best_NLL={best_nll:.6f}",
                end="", flush=True,
            )

        if epoch_nll < best_nll:
            best_nll = epoch_nll
            best_state = model.state_dict()
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= cfg["patience"]:
                print(
                    f"\r    {label} Early stop at epoch {epoch+1}  "
                    f"best_NLL={best_nll:.6f}  kl_w={kl_w:.3f}                    "
                )
                break

    if best_state:
        model.load_state_dict(best_state)

    if patience_ctr < cfg["patience"]:
        print(
            f"\r    {label} Finished {cfg['epochs']} epochs  "
            f"best_NLL={best_nll:.6f}                                    "
        )

    return model


# ══════════════════════════════════════════════════════════════════════════════
# SPATIAL CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
def cross_validate(
    df: pd.DataFrame, features: List[str],
    X_scaled: np.ndarray, Y_norm: np.ndarray,
    y_means: np.ndarray, y_stds: np.ndarray,
    cfg: Dict,
) -> Dict:
    print("\n" + "=" * 70)
    print("SPATIAL CROSS-VALIDATION (Leave-Station-Group-Out)")
    print("=" * 70)

    groups = df["stationId"].values
    n_stations = len(np.unique(groups))
    n_folds = min(n_stations, 5)
    gkf = GroupKFold(n_splits=n_folds)

    Y_pred_all = np.full_like(Y_norm, np.nan)
    Y_unc_all = np.full_like(Y_norm, np.nan)

    for fold_i, (tr_idx, te_idx) in enumerate(gkf.split(X_scaled, Y_norm, groups), 1):
        t0 = time.time()
        print(f"\n  Fold {fold_i}/{n_folds}  "
              f"(train={len(tr_idx)}, test={len(te_idx)})")

        model = train_one_model(
            X_scaled[tr_idx], Y_norm[tr_idx],
            n_features=X_scaled.shape[1], cfg=cfg,
            label=f"Fold {fold_i}/{n_folds}",
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_te = torch.tensor(X_scaled[te_idx], dtype=torch.float32, device=device)
        preds = mc_predict(model, X_te, n_samples=cfg["mc_samples"])  # (T, N, 12)
        Y_pred_all[te_idx] = preds.mean(axis=0)
        Y_unc_all[te_idx] = preds.std(axis=0)

        elapsed = time.time() - t0
        print(f"    Fold {fold_i} done in {elapsed:.1f}s")

    # ── Denormalize & compute metrics ──
    # Y_true comes from df which was already clipped in main()
    Y_true = df[WQ_TARGETS].values

    # Reverse z-norm → log-space
    Y_pred_log = Y_pred_all * y_stds + y_means
    Y_unc_log = Y_unc_all * y_stds

    # Reverse log-transform where applied
    log_mask = np.array([t in LOG_TARGETS for t in WQ_TARGETS])
    Y_pred_denorm = Y_pred_log.copy()
    Y_unc_denorm = Y_unc_log.copy()
    for j in range(len(WQ_TARGETS)):
        if log_mask[j]:
            Y_pred_denorm[:, j] = np.expm1(Y_pred_log[:, j])
            Y_unc_denorm[:, j] = Y_unc_log[:, j] * np.exp(Y_pred_log[:, j])
        # For non-log targets, Y_pred_denorm is already correct

    print("\n" + "=" * 70)
    print("CROSS-VALIDATION RESULTS (single multi-output BNN)")
    print("=" * 70)
    print(f"  {'Parameter':<8} {'R²':>8} {'RMSE':>10} {'MAE':>10} {'MeanUnc':>10}")
    print(f"  {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")

    results = {}
    for i, target in enumerate(WQ_TARGETS):
        tname = WQ_SHORT[target]
        valid = ~np.isnan(Y_pred_denorm[:, i])
        r2 = r2_score(Y_true[valid, i], Y_pred_denorm[valid, i])
        rmse = np.sqrt(mean_squared_error(Y_true[valid, i], Y_pred_denorm[valid, i]))
        mae = mean_absolute_error(Y_true[valid, i], Y_pred_denorm[valid, i])
        unc = float(np.nanmean(Y_unc_denorm[:, i]))
        results[target] = {"R2": r2, "RMSE": rmse, "MAE": mae, "uncertainty": unc}
        print(f"  {tname:<8} {r2:>8.4f} {rmse:>10.4f} {mae:>10.4f} {unc:>10.4f}")

    return results, Y_pred_denorm, Y_unc_denorm


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN FINAL MODEL ON ALL DATA
# ══════════════════════════════════════════════════════════════════════════════
def train_final(
    X_scaled: np.ndarray, Y_norm: np.ndarray,
    y_means: np.ndarray, y_stds: np.ndarray,
    features: List[str], cfg: Dict,
) -> MultiOutputBNN:
    print("\n" + "=" * 70)
    print("TRAINING FINAL MODEL ON ALL DATA")
    print("=" * 70)

    model = train_one_model(
        X_scaled, Y_norm,
        n_features=X_scaled.shape[1], cfg=cfg,
        label="Final",
    )

    # Save everything needed for inference
    save_bundle = {
        "model_state": model.state_dict(),
        "n_features": X_scaled.shape[1],
        "n_targets": N_TARGETS,
        "hidden_dims": cfg["hidden_dims"],
        "dropout": cfg["dropout"],
        "prior_sigma": cfg["prior_sigma"],
        "mc_samples": cfg["mc_samples"],
        "y_means": y_means.tolist(),
        "y_stds": y_stds.tolist(),
        "features": features,
        "targets": WQ_TARGETS,
        "target_short": WQ_SHORT,
    }
    save_path = MODELS_DIR / "bnn_multioutput.pt"
    torch.save(save_bundle, save_path)
    print(f"\n  Saved: {save_path}")
    print(f"  Input: {X_scaled.shape[1]} features → Output: {N_TARGETS} WQ parameters")

    return model


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════
def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [FIG] {name}.png")


def plot_actual_vs_predicted(df, Y_pred, results):
    """Scatter: actual vs predicted for the 6 priority targets."""
    print("\n  --- Actual vs Predicted (BNN) ---")
    Y_true = df[WQ_TARGETS].values

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    for i, target in enumerate(PRIORITY_TARGETS):
        ax = axes.ravel()[i]
        idx = WQ_TARGETS.index(target)
        tname = WQ_SHORT[target]
        r2 = results[target]["R2"]

        yt = Y_true[:, idx]
        yp = Y_pred[:, idx]
        valid = ~np.isnan(yp)

        ax.scatter(yt[valid], yp[valid], alpha=0.3, s=10, c="steelblue")
        lims = [min(yt[valid].min(), yp[valid].min()),
                max(yt[valid].max(), yp[valid].max())]
        ax.plot(lims, lims, "r--", lw=2, label="Perfect fit")
        ax.set_xlabel(f"Actual {tname}")
        ax.set_ylabel(f"Predicted {tname}")
        ax.set_title(f"{tname}  (R²={r2:.4f})", fontweight="bold")
        ax.legend(fontsize=9)

    fig.suptitle("BNN: Actual vs Predicted (Spatial CV, Multi-Output Model)",
                 fontsize=14, fontweight="bold", y=1.01)
    save_fig(fig, "16_bnn_actual_vs_predicted")


def plot_uncertainty_bands(df, features, model, y_means, y_stds, cfg):
    """Plot predictions + 95% CI sorted by actual value."""
    print("\n  --- BNN Uncertainty Bands ---")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    X_scaled = scaler.transform(df[features].values).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_t = torch.tensor(X_scaled, dtype=torch.float32, device=device)
    model = model.to(device)

    preds = mc_predict(model, X_t, n_samples=cfg["mc_samples"])  # (T, N, 12)
    pred_log = preds.mean(axis=0) * y_stds + y_means
    pred_std_log = preds.std(axis=0) * y_stds

    # Reverse log-transform
    log_mask = np.array([t in LOG_TARGETS for t in WQ_TARGETS])
    pred_mean = pred_log.copy()
    pred_std = pred_std_log.copy()
    for j in range(len(WQ_TARGETS)):
        if log_mask[j]:
            pred_mean[:, j] = np.expm1(pred_log[:, j])
            pred_std[:, j] = pred_std_log[:, j] * np.exp(pred_log[:, j])

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    for i, target in enumerate(PRIORITY_TARGETS):
        ax = axes.ravel()[i]
        idx = WQ_TARGETS.index(target)
        tname = WQ_SHORT[target]

        y_true = df[target].values
        order = np.argsort(y_true)
        x_ax = np.arange(len(order))
        ys = y_true[order]
        ms = pred_mean[order, idx]
        ss = pred_std[order, idx]

        ax.plot(x_ax, ys, "k.", markersize=1, alpha=0.4, label="Actual")
        ax.plot(x_ax, ms, color="#1565C0", lw=0.8, label="BNN mean")
        ax.fill_between(x_ax, ms - 1.96 * ss, ms + 1.96 * ss,
                        alpha=0.25, color="#42A5F5", label="95% CI")
        ax.set_title(f"{tname}", fontweight="bold")
        ax.set_xlabel("Sample (sorted)")
        ax.set_ylabel(tname)
        ax.legend(fontsize=8)

    fig.suptitle("BNN: Predictions with 95% Credible Intervals (All outputs from one model)",
                 fontsize=14, fontweight="bold", y=1.01)
    save_fig(fig, "17_bnn_uncertainty_bands")


def plot_r2_and_uncertainty(results):
    """Bar charts: R² and uncertainty per target."""
    print("\n  --- BNN R² & Uncertainty Summary ---")
    names = [WQ_SHORT[t] for t in WQ_TARGETS]
    r2s = [results[t]["R2"] for t in WQ_TARGETS]
    uncs = [results[t]["uncertainty"] for t in WQ_TARGETS]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    colors = ["#4CAF50" if r > 0.5 else "#FF9800" if r > 0 else "#F44336" for r in r2s]
    ax1.bar(names, r2s, color=colors, edgecolor="white")
    ax1.set_ylabel("R²")
    ax1.set_title("BNN R² (Spatial CV, Multi-Output)", fontweight="bold")
    ax1.axhline(0, color="gray", ls="--", alpha=0.5)
    ax1.grid(axis="y", alpha=0.3)
    ax1.tick_params(axis="x", rotation=45)

    ax2.bar(names, uncs, color="#42A5F5", edgecolor="white")
    ax2.set_ylabel("Mean Uncertainty (std)")
    ax2.set_title("BNN Predictive Uncertainty", fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    ax2.tick_params(axis="x", rotation=45)

    fig.suptitle("Bayesian Neural Network — Single Multi-Output Model",
                 fontsize=14, fontweight="bold", y=1.02)
    save_fig(fig, "18_bnn_r2_uncertainty")


def plot_residuals(df, Y_pred, results):
    """Residual plots for priority targets."""
    print("\n  --- BNN Residual Analysis ---")
    Y_true = df[WQ_TARGETS].values

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    for i, target in enumerate(PRIORITY_TARGETS):
        ax = axes.ravel()[i]
        idx = WQ_TARGETS.index(target)
        tname = WQ_SHORT[target]

        valid = ~np.isnan(Y_pred[:, idx])
        resid = Y_true[valid, idx] - Y_pred[valid, idx]

        ax.scatter(Y_pred[valid, idx], resid, alpha=0.3, s=10, c="steelblue")
        ax.axhline(0, color="red", ls="--")
        ax.set_xlabel(f"Predicted {tname}")
        ax.set_ylabel("Residual")
        ax.set_title(f"{tname} Residuals", fontweight="bold")

    fig.suptitle("BNN Residuals (Multi-Output)", fontsize=14, fontweight="bold", y=1.01)
    save_fig(fig, "19_bnn_residuals")


def save_results_csv(results):
    """Save results table."""
    rows = []
    for target in WQ_TARGETS:
        r = results[target]
        rows.append({
            "Parameter": WQ_SHORT[target],
            "R2": round(r["R2"], 4),
            "RMSE": round(r["RMSE"], 4),
            "MAE": round(r["MAE"], 4),
            "Mean_Uncertainty": round(r["uncertainty"], 4),
        })
    pd.DataFrame(rows).to_csv(TABLES_DIR / "bnn_multioutput_results.csv", index=False)
    print(f"  Saved: bnn_multioutput_results.csv")


def save_latex_table(results):
    """LaTeX table for the dissertation."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Bayesian Neural Network Performance (Multi-Output, Spatial CV)}",
        r"\label{tab:bnn_results}",
        r"\begin{tabular}{lrrrr}",
        r"\hline",
        r"\textbf{Parameter} & \textbf{R\textsuperscript{2}} & \textbf{RMSE} & \textbf{MAE} & \textbf{Uncertainty} \\",
        r"\hline",
    ]
    for target in WQ_TARGETS:
        r = results[target]
        tname = WQ_SHORT[target]
        lines.append(
            f"  {tname} & {r['R2']:.4f} & {r['RMSE']:.4f} & {r['MAE']:.4f} & {r['uncertainty']:.4f} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    (TABLES_DIR / "bnn_multioutput.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved: bnn_multioutput.tex")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Train multi-output BNN")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--mc_samples", type=int, default=30)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--prior_sigma", type=float, default=1.0)
    parser.add_argument("--kl_warmup", type=int, default=50,
                        help="KL annealing warm-up epochs (0=full KL from start)")
    parser.add_argument("--hidden", type=str, default="256,128,64",
                        help="Hidden layer sizes, comma-separated")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    cfg = {
        "hidden_dims": tuple(int(x) for x in args.hidden.split(",")),
        "dropout": args.dropout,
        "prior_sigma": args.prior_sigma,
        "lr": args.lr,
        "weight_decay": 1e-5,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "mc_samples": args.mc_samples,
        "patience": args.patience,
        "kl_warmup": args.kl_warmup,
    }

    for d in [RESULTS_DIR, FIGURES_DIR, MODELS_DIR, TABLES_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Load ──
    df, features = load_and_prepare()

    # ── Scale features (always refit to current data) ──
    scaler_path = MODELS_DIR / "scaler.joblib"
    scaler = StandardScaler()
    scaler.fit(df[features].values)
    joblib.dump(scaler, scaler_path)
    X_scaled = scaler.transform(df[features].values).astype(np.float32)

    # ── Scale targets (log-transform skewed + z-norm) ──
    # Data is already cleaned, clipped, and deduplicated in load_and_prepare()
    Y_raw = df[WQ_TARGETS].values.astype(np.float32)

    # Log-transform heavily skewed targets
    log_mask = np.array([t in LOG_TARGETS for t in WQ_TARGETS])
    Y_work = Y_raw.copy()
    for j in range(N_TARGETS):
        if log_mask[j]:
            Y_work[:, j] = np.log1p(np.clip(Y_work[:, j], 0, None))
    print(f"  Log-transformed: {[WQ_SHORT[t] for t in LOG_TARGETS]}")

    y_means = Y_work.mean(axis=0)
    y_stds = Y_work.std(axis=0) + 1e-8
    Y_norm = (Y_work - y_means) / y_stds

    # Save normalisation stats
    json.dump(
        {
            "features": features,
            "y_means": y_means.tolist(),
            "y_stds": y_stds.tolist(),
            "log_targets": LOG_TARGETS,
            "log_mask": log_mask.tolist(),
        },
        (MODELS_DIR / "bnn_norm_stats.json").open("w"),
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
    print(f"  Input: {X_scaled.shape[1]} features → {N_TARGETS} outputs")

    # ── Cross-validate ──
    results, Y_pred, Y_unc = cross_validate(
        df, features, X_scaled, Y_norm, y_means, y_stds, cfg
    )

    # ── Train final model ──
    final_model = train_final(X_scaled, Y_norm, y_means, y_stds, features, cfg)

    # ── Plots ──
    plot_actual_vs_predicted(df, Y_pred, results)
    plot_uncertainty_bands(df, features, final_model, y_means, y_stds, cfg)
    plot_r2_and_uncertainty(results)
    plot_residuals(df, Y_pred, results)

    # ── Tables ──
    save_results_csv(results)
    save_latex_table(results)

    print("\n" + "=" * 70)
    print("BNN TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Model: {MODELS_DIR / 'bnn_multioutput.pt'}")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"  Tables: {TABLES_DIR}")


if __name__ == "__main__":
    main()
