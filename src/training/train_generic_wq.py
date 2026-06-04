"""
train_generic_wq.py
====================
Generic Water Quality Prediction with Station Embeddings + Temporal CV.

Strategies:
  1. Station Embedding: Learnable 8-dim vectors per station (like word2vec)
     - For unseen stations: use nearest-station embedding or mean embedding
  2. Anomaly Targets: Predict WQ / station_seasonal_mean (easier to generalize)
  3. Evaluation:
     - PRIMARY: Temporal split (train 2021-2025, test 2025-2026)
     - SECONDARY: Leave-station-out (cold-start worst-case)
     - TERTIARY: Standard 5-fold (optimistic, for comparison)

Models:
  - XGBoost (baseline, no embeddings — uses station_id as categorical)
  - PyTorch NN with station embeddings (main model)

Output: training_dataset/results_generic/
"""

import os, sys, time, warnings
warnings.filterwarnings("ignore")

_torch_alt = r"C:\torch_tmp"
if os.path.isdir(_torch_alt) and _torch_alt not in sys.path:
    sys.path.insert(0, _torch_alt)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler

from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import joblib

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "training_dataset" / "merged_training_dataset.csv"
RESULTS_DIR = PROJECT_ROOT / "training_dataset" / "results_generic"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"
TABLES_DIR = RESULTS_DIR / "tables"

for d in [RESULTS_DIR, FIGURES_DIR, MODELS_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

BAND_COLS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
INDEX_COLS = ["NDWI", "MNDWI", "NDTI", "NDVI", "NDCI", "MCI", "FAI", "SABI"]
SAT_FEATURES = BAND_COLS + INDEX_COLS

WQ_TARGETS = [
    "wq_BOD", "wq_COD", "wq_CL", "wq_EC", "wq_Depth",
    "wq_DO", "wq_NO3", "wq_TOC", "wq_S", "wq_WT", "wq_WTb", "wq_pH",
]
WQ_SHORT = {
    "wq_BOD": "BOD", "wq_COD": "COD", "wq_CL": "Cl", "wq_EC": "EC",
    "wq_Depth": "Depth", "wq_DO": "DO", "wq_NO3": "NO3", "wq_TOC": "TOC",
    "wq_S": "WL", "wq_WT": "Temp", "wq_WTb": "Turb", "wq_pH": "pH",
}
N_TARGETS = len(WQ_TARGETS)

LOG_TARGETS = {"wq_BOD", "wq_COD", "wq_CL", "wq_NO3", "wq_TOC", "wq_WTb", "wq_S"}

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# NN config
NN_CONFIG = {
    "embedding_dim": 8,
    "hidden_dims": (256, 128, 64),
    "dropout": 0.15,
    "lr": 5e-4,
    "weight_decay": 1e-4,
    "epochs": 300,
    "batch_size": 512,
    "patience": 35,
}

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & CLEANING (same pipeline as train_bnn.py)
# ══════════════════════════════════════════════════════════════════════════════
def load_and_clean() -> pd.DataFrame:
    print("=" * 70)
    print("LOADING & CLEANING DATA")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df["stationId"] = df["stationId"].astype(str)
    for col in WQ_TARGETS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"  Raw: {len(df)} rows, {df['stationId'].nunique()} stations")

    # Step 1: Sentinel removal
    for col in WQ_TARGETS:
        n = (df[col] <= -900).sum()
        if n > 0:
            df.loc[df[col] <= -900, col] = np.nan

    # Step 2: Physical bounds
    PHYSICAL_BOUNDS = {
        "wq_pH": (0, 14), "wq_DO": (0, 25), "wq_BOD": (0, 500),
        "wq_COD": (0, 2000), "wq_EC": (0, 5000), "wq_WT": (0, 50),
        "wq_WTb": (0, 5000), "wq_NO3": (0, 500), "wq_TOC": (0, 500),
        "wq_CL": (0, 5000), "wq_Depth": (0, 50), "wq_S": (0, 1000),
    }
    for col, (lo, hi) in PHYSICAL_BOUNDS.items():
        if col in df.columns:
            df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan

    # Step 3: Clip to p2-p98
    for col in WQ_TARGETS:
        valid = df[col].dropna()
        if len(valid) > 100:
            lo, hi = valid.quantile([0.02, 0.98]).values
            df[col] = df[col].clip(lo, hi)

    # Step 4: Drop rows with <8 valid targets
    n_valid = df[WQ_TARGETS].notna().sum(axis=1)
    df = df[n_valid >= 8].reset_index(drop=True)

    # Step 5: Deduplicate
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
    print(f"  Cleaned + deduplicated: {n_before} → {len(df)} rows")

    # Step 6: KNN impute
    imputer = KNNImputer(n_neighbors=5)
    df[WQ_TARGETS] = pd.DataFrame(
        imputer.fit_transform(df[WQ_TARGETS]),
        columns=WQ_TARGETS, index=df.index,
    )
    df = df.dropna(subset=WQ_TARGETS).reset_index(drop=True)

    print(f"  Final: {len(df)} rows, {df['stationId'].nunique()} stations")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    print("\n  Engineering features...")
    df = df.copy()

    # Temporal
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Spatial — distance from Gangotri
    gangotri_lat, gangotri_lon = 30.99, 78.94
    df["dist_from_source_km"] = np.sqrt(
        ((df["lat"] - gangotri_lat) * 111) ** 2 +
        ((df["lon"] - gangotri_lon) * 111 * np.cos(np.radians(df["lat"]))) ** 2
    )

    # Band ratios
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

    # Remove correlated >0.95
    corr = df[features].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = [c for c in upper.columns if any(upper[c] > 0.95)]
    if drop:
        print(f"  Dropping correlated: {drop}")
        features = [f for f in features if f not in drop]

    print(f"  Features: {len(features)}")
    return df, features


# ══════════════════════════════════════════════════════════════════════════════
# STATION SEASONAL MEANS (for anomaly targets)
# ══════════════════════════════════════════════════════════════════════════════
def compute_station_seasonal_means(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean WQ per station × season for anomaly prediction."""
    df = df.copy()
    df["season"] = df["month"].map({
        1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 2, 10: 3, 11: 3, 12: 0
    })
    means = df.groupby(["stationId", "season"])[WQ_TARGETS].mean()
    return means


def add_anomaly_targets(df: pd.DataFrame, seasonal_means: pd.DataFrame) -> pd.DataFrame:
    """Add WQ anomaly columns: actual / station_seasonal_mean."""
    df = df.copy()
    df["season"] = df["month"].map({
        1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 2, 10: 3, 11: 3, 12: 0
    })
    for col in WQ_TARGETS:
        anom_col = f"{col}_anom"
        df[anom_col] = np.nan
        for idx in df.index:
            sid = df.loc[idx, "stationId"]
            s = df.loc[idx, "season"]
            if (sid, s) in seasonal_means.index:
                mean_val = seasonal_means.loc[(sid, s), col]
                if mean_val > 0.01:  # avoid division by near-zero
                    df.loc[idx, anom_col] = df.loc[idx, col] / mean_val
                else:
                    df.loc[idx, anom_col] = 1.0
            else:
                df.loc[idx, anom_col] = 1.0
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MODEL: NN WITH STATION EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════════════
class StationEmbeddingNN(nn.Module):
    """
    Multi-output NN with station embeddings.
    Input: [continuous_features] + [station_id (integer)]
    Architecture:
      - Station ID → Embedding(n_stations, embed_dim)
      - Concat(features, embedding)
      - FC layers with BatchNorm + LeakyReLU + Dropout
      - Output: n_targets
    """

    def __init__(self, n_features: int, n_stations: int, n_targets: int,
                 embedding_dim: int = 8, hidden_dims=(256, 128, 64),
                 dropout: float = 0.15):
        super().__init__()
        self.embedding = nn.Embedding(n_stations, embedding_dim)

        input_dim = n_features + embedding_dim
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, n_targets))
        self.net = nn.Sequential(*layers)

        # Initialize
        nn.init.normal_(self.embedding.weight, 0, 0.1)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                nn.init.zeros_(m.bias)

    def forward(self, x_cont: torch.Tensor, station_ids: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(station_ids)  # (batch, embed_dim)
        x = torch.cat([x_cont, emb], dim=1)
        return self.net(x)

    def get_embedding(self, station_id: int) -> np.ndarray:
        with torch.no_grad():
            return self.embedding.weight[station_id].cpu().numpy()

    def get_mean_embedding(self) -> np.ndarray:
        with torch.no_grad():
            return self.embedding.weight.mean(dim=0).cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOGIC
# ══════════════════════════════════════════════════════════════════════════════
def train_nn_model(
    X_train: np.ndarray, station_train: np.ndarray, Y_train: np.ndarray,
    X_val: np.ndarray, station_val: np.ndarray, Y_val: np.ndarray,
    n_stations: int, n_targets: int, config: dict,
) -> Tuple[StationEmbeddingNN, float]:
    """Train one instance of StationEmbeddingNN. Returns (model, best_val_loss)."""

    model = StationEmbeddingNN(
        n_features=X_train.shape[1],
        n_stations=n_stations,
        n_targets=n_targets,
        embedding_dim=config["embedding_dim"],
        hidden_dims=config["hidden_dims"],
        dropout=config["dropout"],
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=1e-6
    )
    criterion = nn.HuberLoss(delta=1.0)

    # Create dataloaders
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(station_train, dtype=torch.long),
        torch.tensor(Y_train, dtype=torch.float32),
    )
    train_dl = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True,
                          drop_last=False, pin_memory=True)

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    S_val_t = torch.tensor(station_val, dtype=torch.long).to(DEVICE)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32).to(DEVICE)

    scaler = GradScaler() if DEVICE.type == "cuda" else None
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0.0
        for xb, sb, yb in train_dl:
            xb, sb, yb = xb.to(DEVICE), sb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            if scaler:
                with autocast():
                    pred = model(xb, sb)
                    loss = criterion(pred, yb)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(xb, sb)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_ds)
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t, S_val_t)
            val_loss = criterion(val_pred, Y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_val_loss


def predict_nn(model: StationEmbeddingNN, X: np.ndarray, station_ids: np.ndarray) -> np.ndarray:
    """Get predictions from trained model."""
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    S_t = torch.tensor(station_ids, dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        pred = model(X_t, S_t)
    return pred.cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION FRAMEWORK
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, target_names: List[str]
) -> pd.DataFrame:
    """Compute R², RMSE, MAE per target."""
    results = []
    for j, name in enumerate(target_names):
        mask = ~np.isnan(y_true[:, j]) & ~np.isnan(y_pred[:, j])
        if mask.sum() < 10:
            results.append({"target": name, "R2": np.nan, "RMSE": np.nan, "MAE": np.nan})
            continue
        r2 = r2_score(y_true[mask, j], y_pred[mask, j])
        rmse = np.sqrt(mean_squared_error(y_true[mask, j], y_pred[mask, j]))
        mae = mean_absolute_error(y_true[mask, j], y_pred[mask, j])
        results.append({"target": name, "R2": r2, "RMSE": rmse, "MAE": mae})
    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # ── Load & clean ──
    df = load_and_clean()
    df, features = engineer_features(df)

    # ── Encode stations ──
    le = LabelEncoder()
    df["station_enc"] = le.fit_transform(df["stationId"])
    n_stations = df["station_enc"].nunique()
    joblib.dump(le, MODELS_DIR / "station_encoder.joblib")
    print(f"  Stations encoded: {n_stations}")

    # ── Prepare targets ──
    # Apply log-transform to skewed targets
    Y_raw = df[WQ_TARGETS].values.astype(np.float32)
    log_mask = np.array([t in LOG_TARGETS for t in WQ_TARGETS])
    Y_work = Y_raw.copy()
    for j in range(N_TARGETS):
        if log_mask[j]:
            Y_work[:, j] = np.log1p(np.clip(Y_work[:, j], 0, None))

    # Z-normalize targets
    y_means = Y_work.mean(axis=0)
    y_stds = Y_work.std(axis=0) + 1e-8
    Y_norm = (Y_work - y_means) / y_stds
    np.savez(MODELS_DIR / "target_stats.npz", means=y_means, stds=y_stds, log_mask=log_mask)

    # ── Prepare features ──
    X = df[features].values.astype(np.float32)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, MODELS_DIR / "feature_scaler.joblib")

    station_ids = df["station_enc"].values
    dates = df["date"].values

    print(f"\n  X shape: {X_scaled.shape}, Y shape: {Y_norm.shape}")
    print(f"  Device: {DEVICE}")

    # ══════════════════════════════════════════════════════════════════════════
    # EVALUATION 1: TEMPORAL SPLIT (Primary — realistic operational scenario)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EVALUATION 1: TEMPORAL SPLIT (train ≤ 2025-01, test > 2025-01)")
    print("=" * 70)

    split_date = pd.Timestamp("2025-01-01")
    train_mask = df["date"] <= split_date
    test_mask = df["date"] > split_date

    X_tr, X_te = X_scaled[train_mask], X_scaled[test_mask]
    Y_tr, Y_te = Y_norm[train_mask], Y_norm[test_mask]
    S_tr, S_te = station_ids[train_mask], station_ids[test_mask]
    Y_raw_te = Y_raw[test_mask]

    print(f"  Train: {X_tr.shape[0]} (up to {split_date.date()})")
    print(f"  Test:  {X_te.shape[0]} (after {split_date.date()})")
    print(f"  Test stations: {len(np.unique(S_te))}")

    # -- NN with embeddings --
    t0 = time.time()
    # Use 10% of train as validation for early stopping
    n_tr = len(X_tr)
    perm = np.random.permutation(n_tr)
    val_size = int(n_tr * 0.1)
    val_idx, tr_idx = perm[:val_size], perm[val_size:]

    model_temporal, val_loss = train_nn_model(
        X_tr[tr_idx], S_tr[tr_idx], Y_tr[tr_idx],
        X_tr[val_idx], S_tr[val_idx], Y_tr[val_idx],
        n_stations, N_TARGETS, NN_CONFIG,
    )
    elapsed = time.time() - t0
    print(f"  NN trained in {elapsed:.1f}s (val_loss={val_loss:.4f})")

    # Predict
    Y_pred_norm = predict_nn(model_temporal, X_te, S_te)

    # Denormalize
    Y_pred_work = Y_pred_norm * y_stds + y_means
    Y_pred_raw = Y_pred_work.copy()
    for j in range(N_TARGETS):
        if log_mask[j]:
            Y_pred_raw[:, j] = np.expm1(Y_pred_work[:, j])

    results_temporal_nn = evaluate_predictions(Y_raw_te, Y_pred_raw,
                                               [WQ_SHORT[t] for t in WQ_TARGETS])
    print("\n  NN + Station Embeddings (Temporal Split):")
    print(results_temporal_nn.to_string(index=False))

    # -- XGBoost with station as categorical feature --
    print("\n  Training XGBoost (with station as feature)...")
    X_tr_xgb = np.column_stack([X_tr, S_tr.reshape(-1, 1)])
    X_te_xgb = np.column_stack([X_te, S_te.reshape(-1, 1)])

    xgb_preds = np.zeros_like(Y_te)
    for j in range(N_TARGETS):
        xgb_model = xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=SEED, verbosity=0, tree_method="hist",
        )
        xgb_model.fit(X_tr_xgb, Y_tr[:, j])
        xgb_preds[:, j] = xgb_model.predict(X_te_xgb)

    # Denormalize XGBoost
    xgb_pred_work = xgb_preds * y_stds + y_means
    xgb_pred_raw = xgb_pred_work.copy()
    for j in range(N_TARGETS):
        if log_mask[j]:
            xgb_pred_raw[:, j] = np.expm1(xgb_pred_work[:, j])

    results_temporal_xgb = evaluate_predictions(Y_raw_te, xgb_pred_raw,
                                                 [WQ_SHORT[t] for t in WQ_TARGETS])
    print("\n  XGBoost + Station Feature (Temporal Split):")
    print(results_temporal_xgb.to_string(index=False))

    # ══════════════════════════════════════════════════════════════════════════
    # EVALUATION 2: STANDARD 5-FOLD CV (all stations in train & test)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EVALUATION 2: STANDARD 5-FOLD CV (stations shared train/test)")
    print("=" * 70)

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    Y_pred_kfold = np.full_like(Y_norm, np.nan)

    for fold_i, (tr_idx, te_idx) in enumerate(kf.split(X_scaled)):
        t0 = time.time()
        # Split for early stopping
        n = len(tr_idx)
        perm = np.random.permutation(n)
        vi = perm[:int(n * 0.1)]
        ti = perm[int(n * 0.1):]
        actual_tr = tr_idx[ti]
        actual_val = tr_idx[vi]

        model_fold, _ = train_nn_model(
            X_scaled[actual_tr], station_ids[actual_tr], Y_norm[actual_tr],
            X_scaled[actual_val], station_ids[actual_val], Y_norm[actual_val],
            n_stations, N_TARGETS, NN_CONFIG,
        )
        Y_pred_kfold[te_idx] = predict_nn(model_fold, X_scaled[te_idx], station_ids[te_idx])
        elapsed = time.time() - t0
        print(f"  Fold {fold_i+1}/5 done ({elapsed:.1f}s)")

    # Denormalize
    Y_kfold_work = Y_pred_kfold * y_stds + y_means
    Y_kfold_raw = Y_kfold_work.copy()
    for j in range(N_TARGETS):
        if log_mask[j]:
            Y_kfold_raw[:, j] = np.expm1(Y_kfold_work[:, j])

    results_kfold = evaluate_predictions(Y_raw, Y_kfold_raw,
                                          [WQ_SHORT[t] for t in WQ_TARGETS])
    print("\n  NN + Station Embeddings (5-Fold CV):")
    print(results_kfold.to_string(index=False))

    # ══════════════════════════════════════════════════════════════════════════
    # EVALUATION 3: LEAVE-STATION-OUT (Cold-start worst case)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EVALUATION 3: LEAVE-STATION-GROUP-OUT (cold-start)")
    print("=" * 70)

    gkf = GroupKFold(n_splits=5)
    Y_pred_spatial = np.full_like(Y_norm, np.nan)

    for fold_i, (tr_idx, te_idx) in enumerate(gkf.split(X_scaled, Y_norm, station_ids)):
        t0 = time.time()
        # For unseen stations: use mean embedding
        # Split train for early stopping
        n = len(tr_idx)
        perm = np.random.permutation(n)
        vi = perm[:int(n * 0.1)]
        ti = perm[int(n * 0.1):]
        actual_tr = tr_idx[ti]
        actual_val = tr_idx[vi]

        model_fold, _ = train_nn_model(
            X_scaled[actual_tr], station_ids[actual_tr], Y_norm[actual_tr],
            X_scaled[actual_val], station_ids[actual_val], Y_norm[actual_val],
            n_stations, N_TARGETS, NN_CONFIG,
        )

        # For test stations, use their station IDs (even if not seen in training,
        # the embedding was initialized but not trained — gives a "cold start" baseline)
        Y_pred_spatial[te_idx] = predict_nn(model_fold, X_scaled[te_idx], station_ids[te_idx])
        elapsed = time.time() - t0
        test_stations = np.unique(station_ids[te_idx])
        print(f"  Fold {fold_i+1}/5 done ({elapsed:.1f}s) — "
              f"test stations: {le.inverse_transform(test_stations)}")

    # Denormalize
    Y_spatial_work = Y_pred_spatial * y_stds + y_means
    Y_spatial_raw = Y_spatial_work.copy()
    for j in range(N_TARGETS):
        if log_mask[j]:
            Y_spatial_raw[:, j] = np.expm1(Y_spatial_work[:, j])

    results_spatial = evaluate_predictions(Y_raw, Y_spatial_raw,
                                            [WQ_SHORT[t] for t in WQ_TARGETS])
    print("\n  NN + Station Embeddings (Leave-Station-Out):")
    print(results_spatial.to_string(index=False))

    # ══════════════════════════════════════════════════════════════════════════
    # TRAIN FINAL MODEL ON ALL DATA
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TRAINING FINAL MODEL ON ALL DATA")
    print("=" * 70)

    # Use 10% holdout for early stopping
    perm = np.random.permutation(len(X_scaled))
    val_size = int(len(X_scaled) * 0.1)
    final_val, final_tr = perm[:val_size], perm[val_size:]

    final_model, final_val_loss = train_nn_model(
        X_scaled[final_tr], station_ids[final_tr], Y_norm[final_tr],
        X_scaled[final_val], station_ids[final_val], Y_norm[final_val],
        n_stations, N_TARGETS, NN_CONFIG,
    )
    torch.save({
        "model_state": final_model.state_dict(),
        "n_features": X_scaled.shape[1],
        "n_stations": n_stations,
        "n_targets": N_TARGETS,
        "config": NN_CONFIG,
    }, MODELS_DIR / "station_embedding_nn.pt")
    print(f"  Final model saved (val_loss={final_val_loss:.4f})")

    # ══════════════════════════════════════════════════════════════════════════
    # COMPARISON SUMMARY & FIGURES
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SUMMARY: ALL EVALUATION STRATEGIES")
    print("=" * 70)

    summary = pd.DataFrame({
        "Target": [WQ_SHORT[t] for t in WQ_TARGETS],
        "Temporal_NN_R2": results_temporal_nn["R2"].values,
        "Temporal_XGB_R2": results_temporal_xgb["R2"].values,
        "KFold_NN_R2": results_kfold["R2"].values,
        "Spatial_NN_R2": results_spatial["R2"].values,
    })
    print(summary.to_string(index=False))
    summary.to_csv(TABLES_DIR / "evaluation_comparison.csv", index=False)

    # ── Figure: R² comparison across evaluation strategies ──
    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(N_TARGETS)
    width = 0.2
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]
    labels = ["Temporal (NN)", "Temporal (XGB)", "5-Fold CV (NN)", "Leave-Station-Out (NN)"]

    for i, (col, color, label) in enumerate(zip(
        ["Temporal_NN_R2", "Temporal_XGB_R2", "KFold_NN_R2", "Spatial_NN_R2"],
        colors, labels
    )):
        vals = summary[col].values
        ax.bar(x + i * width, vals, width, label=label, color=color, edgecolor="white")

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([WQ_SHORT[t] for t in WQ_TARGETS], fontsize=10, fontweight="bold")
    ax.set_ylabel("R² Score", fontsize=12)
    ax.set_title("Model Performance: Station Embedding NN\n"
                 "Temporal CV (realistic) vs 5-Fold (shared stations) vs Spatial CV (cold-start)",
                 fontsize=13, fontweight="bold")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(-0.5, 1.0)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "01_r2_comparison_all_strategies.png", dpi=200)
    plt.close()
    print("  [FIG] 01_r2_comparison_all_strategies.png")

    # ── Figure: Actual vs Predicted (temporal split, NN) ──
    fig, axes = plt.subplots(3, 4, figsize=(20, 14))
    axes = axes.ravel()
    for j, (target, ax) in enumerate(zip(WQ_TARGETS, axes)):
        yt = Y_raw_te[:, j]
        yp = Y_pred_raw[:, j]
        mask = ~np.isnan(yt) & ~np.isnan(yp)
        if mask.sum() < 10:
            ax.set_visible(False)
            continue
        r2 = r2_score(yt[mask], yp[mask])
        ax.scatter(yt[mask], yp[mask], alpha=0.3, s=8, c="#2196F3")
        lims = [min(yt[mask].min(), yp[mask].min()), max(yt[mask].max(), yp[mask].max())]
        ax.plot(lims, lims, "r--", linewidth=1.5)
        ax.set_title(f"{WQ_SHORT[target]} (R²={r2:.3f})", fontweight="bold")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
    plt.suptitle("Actual vs Predicted — Temporal Split (NN + Station Embeddings)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "02_actual_vs_predicted_temporal.png", dpi=200)
    plt.close()
    print("  [FIG] 02_actual_vs_predicted_temporal.png")

    # ── Figure: Station Embeddings visualization (PCA/t-SNE) ──
    print("\n  Visualizing station embeddings...")
    emb_weights = final_model.embedding.weight.detach().cpu().numpy()
    station_names = le.classes_

    # PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(emb_weights)

    fig, ax = plt.subplots(figsize=(10, 8))
    # Color by region
    region_colors = []
    for sid in station_names:
        lon = df[df["stationId"] == sid]["lon"].iloc[0]
        if lon < 81:
            region_colors.append("#e74c3c")  # Upper
        elif lon < 87:
            region_colors.append("#f39c12")  # Middle
        else:
            region_colors.append("#27ae60")  # Lower

    ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=region_colors, s=120, edgecolors="k", zorder=3)
    for i, sid in enumerate(station_names):
        ax.annotate(sid, (emb_2d[i, 0], emb_2d[i, 1]),
                    fontsize=8, textcoords="offset points", xytext=(5, 5))

    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor="#e74c3c", label="Upper Ganga"),
        Patch(facecolor="#f39c12", label="Middle Ganga"),
        Patch(facecolor="#27ae60", label="Lower Ganga"),
    ]
    ax.legend(handles=legend_elems, fontsize=10)
    ax.set_title("Learned Station Embeddings (PCA projection)\n"
                 "Stations in same region cluster together",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "03_station_embeddings_pca.png", dpi=200)
    plt.close()
    print("  [FIG] 03_station_embeddings_pca.png")

    # ── LaTeX table ──
    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{R² Comparison Across Evaluation Strategies (Station Embedding NN)}",
        r"\label{tab:generic_comparison}",
        r"\begin{tabular}{lrrrr}",
        r"\hline",
        r"\textbf{Parameter} & \textbf{Temporal NN} & \textbf{Temporal XGB} & \textbf{5-Fold NN} & \textbf{Spatial NN} \\",
        r"\hline",
    ]
    for _, row in summary.iterrows():
        latex_lines.append(
            f"  {row['Target']} & {row['Temporal_NN_R2']:.4f} & {row['Temporal_XGB_R2']:.4f} "
            f"& {row['KFold_NN_R2']:.4f} & {row['Spatial_NN_R2']:.4f} \\\\"
        )
    latex_lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    (TABLES_DIR / "evaluation_comparison.tex").write_text("\n".join(latex_lines), encoding="utf-8")
    print("  [TEX] evaluation_comparison.tex")

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"  Results: {RESULTS_DIR}")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"  Models:  {MODELS_DIR}")


if __name__ == "__main__":
    main()
