"""
bayesian_wq_model.py
====================
Dissertation-grade water quality prediction from Sentinel-2 imagery.

Phases:
  1. EDA  — statistical profiling, correlations, spatial/temporal analysis
  2. Feature Engineering — temporal, spatial, band ratios, feature selection
  3. Model Training — BN, Random Forest, XGBoost, Linear Regression, GPR
  4. Bayesian Network Deep Dive — structure learning, DAG visualisation,
     sensitivity analysis, uncertainty quantification
  5. Evaluation — LOSO-CV, temporal split, metrics comparison tables
  6. Export — saved models, figures, LaTeX tables

Outputs go to:  GangaIndices_All/results/
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Ensure PyTorch is importable (may be in alternate install path on Windows)
_torch_alt = r"C:\torch_tmp"
if os.path.isdir(_torch_alt) and _torch_alt not in sys.path:
    sys.path.insert(0, _torch_alt)

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
import joblib
import xgboost as xgb

from pgmpy.estimators import HillClimbSearch, MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.estimators.StructureScore import K2, BDeu
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "training_dataset" / "merged_training_dataset.csv"
RESULTS_DIR = PROJECT_ROOT / "training_dataset" / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"
TABLES_DIR = RESULTS_DIR / "tables"

# Feature groups
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

# Short labels for plots
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

PRIORITY_TARGETS = [
    "wq_BOD",
    "wq_DO",
    "wq_WTb",
    "wq_pH",
    "wq_COD",
    "wq_EC",
]

DPI = 300
SEED = 42
np.random.seed(SEED)

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def setup_dirs():
    for d in [RESULTS_DIR, FIGURES_DIR, MODELS_DIR, TABLES_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def save_fig(fig, name: str, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [FIG] {name}.png")


def short(col: str) -> str:
    return WQ_SHORT.get(col, col.replace("wq_", ""))


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: DATA LOADING & EDA
# ══════════════════════════════════════════════════════════════════════════════
def load_data() -> pd.DataFrame:
    print("=" * 70)
    print("PHASE 1: DATA LOADING & EXPLORATORY ANALYSIS")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df["stationId"] = df["stationId"].astype(str)

    for col in WQ_TARGETS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"Raw dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Stations: {df['stationId'].nunique()}")
    print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")

    # ── STEP 1: Replace -999 sentinel values with NaN ──
    for col in WQ_TARGETS:
        n_sentinel = (df[col] <= -900).sum()
        if n_sentinel > 0:
            df.loc[df[col] <= -900, col] = np.nan
            print(f"  {col}: replaced {n_sentinel} sentinel values with NaN")

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
                print(f"  {col}: removed {n_invalid} out-of-bounds values")

    # ── STEP 3: Clip remaining outliers to 2nd-98th percentile ──
    for col in WQ_TARGETS:
        valid_vals = df[col].dropna()
        if len(valid_vals) > 100:
            lo, hi = valid_vals.quantile([0.02, 0.98]).values
            df[col] = df[col].clip(lo, hi)

    # ── STEP 4: Drop rows with too many NaN targets ──
    n_valid_targets = df[WQ_TARGETS].notna().sum(axis=1)
    df = df[n_valid_targets >= 8].reset_index(drop=True)
    print(f"  After quality filter (>=8/12 targets valid): {len(df)} rows")

    # ── STEP 5: Deduplicate by averaging WQ per unique satellite obs ──
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
    print(f"  Deduplicated: {n_before} → {len(df)} rows")

    missing = df[WQ_TARGETS + SAT_FEATURES].isnull().sum()
    missing = missing[missing > 0]
    if len(missing):
        print(f"  Remaining missing values:\n{missing}")
    return df


def eda_statistical_profiling(df: pd.DataFrame):
    print("\n--- Statistical Profiling ---")
    desc = df[SAT_FEATURES + WQ_TARGETS].describe().T
    desc["skew"] = df[SAT_FEATURES + WQ_TARGETS].skew()
    desc["kurtosis"] = df[SAT_FEATURES + WQ_TARGETS].kurtosis()
    desc.to_csv(TABLES_DIR / "statistical_summary.csv")
    print(f"  Saved statistical_summary.csv ({desc.shape[0]} variables)")

    # Distribution plots — satellite features
    fig, axes = plt.subplots(3, 6, figsize=(24, 12))
    axes = axes.ravel()
    for i, col in enumerate(SAT_FEATURES):
        ax = axes[i]
        df[col].dropna().hist(bins=50, ax=ax, color="steelblue", edgecolor="none", alpha=0.8)
        ax.set_title(col, fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=8)
    for j in range(len(SAT_FEATURES), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Sentinel-2 Feature Distributions", fontsize=14, fontweight="bold", y=1.01)
    save_fig(fig, "01_satellite_distributions")

    # Distribution plots — WQ targets
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.ravel()
    for i, col in enumerate(WQ_TARGETS):
        ax = axes[i]
        df[col].dropna().hist(bins=50, ax=ax, color="darkorange", edgecolor="none", alpha=0.8)
        ax.set_title(short(col), fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=8)
    fig.suptitle("Water Quality Parameter Distributions", fontsize=14, fontweight="bold", y=1.01)
    save_fig(fig, "02_wq_distributions")


def eda_correlation_analysis(df: pd.DataFrame):
    print("\n--- Correlation Analysis ---")
    # Satellite ↔ WQ Pearson correlation
    corr_df = pd.DataFrame(index=SAT_FEATURES, columns=[short(c) for c in WQ_TARGETS])
    pval_df = corr_df.copy()
    for si, sat in enumerate(SAT_FEATURES):
        for wi, wq in enumerate(WQ_TARGETS):
            mask = df[[sat, wq]].dropna().index
            if len(mask) < 30:
                corr_df.iloc[si, wi] = np.nan
                continue
            r, p = stats.pearsonr(df.loc[mask, sat], df.loc[mask, wq])
            corr_df.iloc[si, wi] = r
            pval_df.iloc[si, wi] = p

    corr_df = corr_df.astype(float)
    corr_df.to_csv(TABLES_DIR / "sat_wq_pearson_correlation.csv")

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr_df, annot=True, fmt=".2f", center=0, cmap="RdBu_r",
                vmin=-1, vmax=1, ax=ax, linewidths=0.5,
                annot_kws={"size": 8})
    ax.set_title("Pearson Correlation: Satellite Features vs Water Quality", fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    save_fig(fig, "03_sat_wq_correlation_heatmap")

    # Spearman correlation
    spear_df = pd.DataFrame(index=SAT_FEATURES, columns=[short(c) for c in WQ_TARGETS])
    for si, sat in enumerate(SAT_FEATURES):
        for wi, wq in enumerate(WQ_TARGETS):
            mask = df[[sat, wq]].dropna().index
            if len(mask) < 30:
                spear_df.iloc[si, wi] = np.nan
                continue
            r, _ = stats.spearmanr(df.loc[mask, sat], df.loc[mask, wq])
            spear_df.iloc[si, wi] = r
    spear_df = spear_df.astype(float)
    spear_df.to_csv(TABLES_DIR / "sat_wq_spearman_correlation.csv")

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(spear_df, annot=True, fmt=".2f", center=0, cmap="RdBu_r",
                vmin=-1, vmax=1, ax=ax, linewidths=0.5, annot_kws={"size": 8})
    ax.set_title("Spearman Correlation: Satellite Features vs Water Quality", fontsize=13, fontweight="bold")
    save_fig(fig, "04_sat_wq_spearman_heatmap")

    # Inter-feature correlation
    fig, ax = plt.subplots(figsize=(12, 10))
    feat_corr = df[SAT_FEATURES].corr()
    sns.heatmap(feat_corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
                linewidths=0.5, annot_kws={"size": 7})
    ax.set_title("Inter-Feature Correlation (Multicollinearity Check)", fontsize=13, fontweight="bold")
    save_fig(fig, "05_feature_intercorrelation")

    return corr_df


def eda_spatial_analysis(df: pd.DataFrame):
    print("\n--- Spatial Analysis (per-station means) ---")
    station_means = df.groupby("stationId")[SAT_FEATURES + WQ_TARGETS + ["lon", "lat"]].mean()
    station_means = station_means.sort_values("lon")  # roughly west→east along Ganga

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    for i, col in enumerate(PRIORITY_TARGETS):
        ax = axes.ravel()[i]
        ax.scatter(station_means["lon"], station_means[col], c=station_means["lat"],
                   cmap="viridis", s=80, edgecolors="k", linewidth=0.5)
        ax.set_xlabel("Longitude")
        ax.set_ylabel(short(col))
        ax.set_title(f"{short(col)} along Ganga (W→E)", fontweight="bold")
    fig.suptitle("Spatial Variation of Key WQ Parameters", fontsize=14, fontweight="bold", y=1.01)
    save_fig(fig, "06_spatial_variation")


def eda_temporal_analysis(df: pd.DataFrame):
    print("\n--- Temporal Analysis ---")
    df_t = df.copy()
    df_t["month"] = df_t["date"].dt.month
    df_t["season"] = df_t["month"].map(
        {1: "Winter", 2: "Winter", 3: "Pre-Monsoon", 4: "Pre-Monsoon",
         5: "Pre-Monsoon", 6: "Monsoon", 7: "Monsoon", 8: "Monsoon",
         9: "Monsoon", 10: "Post-Monsoon", 11: "Post-Monsoon", 12: "Winter"}
    )
    season_order = ["Pre-Monsoon", "Monsoon", "Post-Monsoon", "Winter"]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    for i, col in enumerate(PRIORITY_TARGETS):
        ax = axes.ravel()[i]
        data_list = [df_t[df_t["season"] == s][col].dropna() for s in season_order]
        bp = ax.boxplot(data_list, labels=season_order, patch_artist=True)
        colors = ["#FFD700", "#2196F3", "#FF9800", "#9C27B0"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_title(short(col), fontweight="bold", fontsize=12)
        ax.tick_params(labelsize=9)
    fig.suptitle("Seasonal Variation of Water Quality Parameters", fontsize=14, fontweight="bold", y=1.01)
    save_fig(fig, "07_seasonal_variation")

    # Monthly trends for satellite indices
    monthly_idx = df_t.groupby("month")[INDEX_COLS].mean()
    fig, ax = plt.subplots(figsize=(12, 6))
    for col in INDEX_COLS:
        ax.plot(monthly_idx.index, monthly_idx[col], marker="o", label=col, linewidth=2)
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Mean Index Value", fontsize=12)
    ax.set_title("Monthly Trends of Spectral Indices", fontsize=13, fontweight="bold")
    ax.legend(ncol=4, fontsize=9)
    ax.set_xticks(range(1, 13))
    ax.grid(alpha=0.3)
    save_fig(fig, "08_monthly_index_trends")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    print("\n" + "=" * 70)
    print("PHASE 2: FEATURE ENGINEERING")
    print("=" * 70)

    df = df.copy()

    # Temporal features
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["season"] = df["month"].map(
        {1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 2, 10: 3, 11: 3, 12: 0}
    )

    # Spatial — distance from Gangotri (source: 30.99°N, 78.94°E)
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

    engineered_cols = [
        "month_sin", "month_cos", "dist_from_source_km",
        "B5_B4", "B3_B2", "B4_B3", "B8_B4", "B11_B8", "B5_B6",
    ]
    all_features = SAT_FEATURES + engineered_cols

    # Impute missing values
    imputer = KNNImputer(n_neighbors=5)
    df[WQ_TARGETS] = pd.DataFrame(
        imputer.fit_transform(df[WQ_TARGETS]),
        columns=WQ_TARGETS, index=df.index,
    )
    print(f"  KNN-imputed {df[WQ_TARGETS].isnull().sum().sum()} remaining NaNs in WQ targets")

    # Feature selection — remove highly correlated features (VIF proxy)
    corr_matrix = df[all_features].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
    if to_drop:
        print(f"  Dropping highly correlated (>0.95): {to_drop}")
        all_features = [f for f in all_features if f not in to_drop]

    print(f"  Final feature set: {len(all_features)} features")
    for f in all_features:
        print(f"    {f}")

    # Save feature config
    feature_config = {"features": all_features, "targets": WQ_TARGETS}
    with (MODELS_DIR / "feature_config.json").open("w") as fp:
        json.dump(feature_config, fp, indent=2)

    return df, all_features


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
def train_all_models(
    df: pd.DataFrame,
    features: List[str],
) -> Dict:
    print("\n" + "=" * 70)
    print("PHASE 3: MODEL TRAINING (Multi-Model Comparison)")
    print("=" * 70)

    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")

    results = {}  # {target: {model_name: {metrics...}}}
    trained_models = {}  # {target: {model_name: model}}

    for target in WQ_TARGETS:
        tname = short(target)
        y = df[target].values
        mask = ~np.isnan(y)
        X_t, y_t = X_scaled[mask], y[mask]

        print(f"\n  ── {tname} ({sum(mask)} samples) ──")
        results[target] = {}
        trained_models[target] = {}

        # Spatial cross-validation: GroupKFold by station
        groups = df.loc[mask, "stationId"].values
        unique_stations = np.unique(groups)
        n_stations = len(unique_stations)

        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(
                n_estimators=200, max_depth=15, min_samples_leaf=5,
                n_jobs=-1, random_state=SEED,
            ),
            "XGBoost": xgb.XGBRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                random_state=SEED, verbosity=0,
            ),
        }

        for mname, model in models.items():
            y_pred_all = np.full(len(y_t), np.nan)
            gkf = GroupKFold(n_splits=min(n_stations, 5))

            for train_idx, test_idx in gkf.split(X_t, y_t, groups):
                model_clone = _clone_model(model)
                model_clone.fit(X_t[train_idx], y_t[train_idx])
                y_pred_all[test_idx] = model_clone.predict(X_t[test_idx])

            valid = ~np.isnan(y_pred_all)
            r2 = r2_score(y_t[valid], y_pred_all[valid])
            rmse = np.sqrt(mean_squared_error(y_t[valid], y_pred_all[valid]))
            mae = mean_absolute_error(y_t[valid], y_pred_all[valid])
            results[target][mname] = {"R2": r2, "RMSE": rmse, "MAE": mae}
            print(f"    {mname:25s}  R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")

            # Train final model on all data
            final_model = _clone_model(model)
            final_model.fit(X_t, y_t)
            trained_models[target][mname] = final_model

    # Save all trained models
    joblib.dump(trained_models, MODELS_DIR / "all_models.joblib")
    print("\n  All models saved to models/all_models.joblib")

    return results, trained_models


def _clone_model(model):
    from sklearn.base import clone
    return clone(model)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3B: BAYESIAN NEURAL NETWORK  (Bayes by Backprop)
# Inspired by: Qadir & Bilgin, "Hyperspectral Images Classification with
# Deep Bayesian Neural Networks", IJACEN 2023.
#
# Architecture: Variational inference via local reparameterization trick.
# Each linear layer has Gaussian distributions over weights (mu, rho→sigma).
# Training minimises the ELBO (KL divergence + negative log-likelihood).
# Inference uses T Monte-Carlo forward passes → mean prediction + uncertainty.
# ══════════════════════════════════════════════════════════════════════════════

# ── Bayesian Linear Layer ─────────────────────────────────────────────────────
class BayesianLinear(nn.Module):
    """Linear layer with Gaussian weight posteriors (Bayes by Backprop).

    weight ~ N(mu_w, softplus(rho_w)^2)
    bias   ~ N(mu_b, softplus(rho_b)^2)
    Prior  ~ N(0, prior_sigma^2)
    """

    def __init__(self, in_features: int, out_features: int, prior_sigma: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Variational parameters
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), -3.0))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.full((out_features,), -3.0))

        # Prior
        self.prior_sigma = prior_sigma
        self.prior_log_sigma = np.log(prior_sigma)

        # Initialise mu with Kaiming-like scale
        nn.init.kaiming_normal_(self.weight_mu, nonlinearity="relu")

        # Track KL for this layer
        self.kl_divergence = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # sigma = log(1 + exp(rho))   (softplus, always positive)
        weight_sigma = F.softplus(self.weight_rho)
        bias_sigma = F.softplus(self.bias_rho)

        # ── Local reparameterisation trick ──
        # Instead of sampling W then computing xW, sample the activation
        # distribution directly:  act ~ N(x @ mu, sqrt(x^2 @ sigma^2))
        act_mu = F.linear(x, self.weight_mu, self.bias_mu)
        act_var = F.linear(x.pow(2), weight_sigma.pow(2), bias_sigma.pow(2))
        act_std = torch.sqrt(act_var + 1e-8)
        eps = torch.randn_like(act_mu)
        activation = act_mu + act_std * eps

        # ── KL divergence (closed-form for two Gaussians) ──
        kl_w = self._kl_gaussian(
            self.weight_mu, weight_sigma, 0.0, self.prior_sigma
        )
        kl_b = self._kl_gaussian(
            self.bias_mu, bias_sigma, 0.0, self.prior_sigma
        )
        self.kl_divergence = kl_w + kl_b

        return activation

    @staticmethod
    def _kl_gaussian(mu_q, sigma_q, mu_p, sigma_p):
        """KL( N(mu_q, sigma_q^2) || N(mu_p, sigma_p^2) ) summed over all params."""
        var_q = sigma_q.pow(2)
        var_p = sigma_p ** 2
        kl = 0.5 * (
            (var_q / var_p)
            + ((mu_q - mu_p) ** 2) / var_p
            - 1.0
            + np.log(var_p) - torch.log(var_q)
        )
        return kl.sum()


# ── BNN Model ─────────────────────────────────────────────────────────────────
class BayesianNeuralNetwork(nn.Module):
    """Multi-target regression BNN with Bayes by Backprop.

    Architecture: Input → BayesLinear(256) → ReLU → Dropout
                       → BayesLinear(128) → ReLU → Dropout
                       → BayesLinear(64)  → ReLU → Dropout
                       → BayesLinear(n_targets)
    """

    def __init__(self, n_features: int, n_targets: int,
                 hidden_dims=(256, 128, 64), dropout: float = 0.1,
                 prior_sigma: float = 1.0):
        super().__init__()
        layers = []
        prev_dim = n_features
        for h in hidden_dims:
            layers.append(BayesianLinear(prev_dim, h, prior_sigma=prior_sigma))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(BayesianLinear(prev_dim, n_targets, prior_sigma=prior_sigma))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def kl_loss(self) -> torch.Tensor:
        """Sum KL divergence over all Bayesian layers."""
        kl = torch.tensor(0.0, device=next(self.parameters()).device)
        for module in self.modules():
            if isinstance(module, BayesianLinear):
                kl = kl + module.kl_divergence
        return kl


# ── ELBO Loss ─────────────────────────────────────────────────────────────────
class ELBOLoss(nn.Module):
    """Evidence Lower Bound loss = NLL + KL_weight * KL_divergence."""

    def __init__(self, n_train_samples: int):
        super().__init__()
        self.n_train = n_train_samples

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor,
                kl: torch.Tensor) -> torch.Tensor:
        nll = F.mse_loss(y_pred, y_true, reduction="mean")
        # Scale KL by 1/N (number of training points) — standard ELBO weighting
        kl_weight = 1.0 / self.n_train
        return nll + kl_weight * kl


# ── BNN Training ──────────────────────────────────────────────────────────────
BNN_CONFIG = {
    "hidden_dims": (256, 128, 64),
    "dropout": 0.1,
    "prior_sigma": 1.0,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "epochs": 200,
    "batch_size": 512,
    "mc_samples": 30,       # Monte Carlo forward passes at inference
    "patience": 25,         # early stopping patience
}


def train_bnn(
    df: pd.DataFrame,
    features: List[str],
    ml_results: Dict,
) -> Tuple[Dict, Dict]:
    """Train a Bayesian Neural Network (Bayes by Backprop) for all WQ targets.

    Returns
    -------
    bnn_results : dict   {target: {R2, RMSE, MAE, mean_uncertainty}}
    bnn_models  : dict   {target: trained BayesianNeuralNetwork}
    """
    print("\n" + "=" * 70)
    print("PHASE 3B: BAYESIAN NEURAL NETWORK (Bayes by Backprop)")
    print("=" * 70)
    cfg = BNN_CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Config: hidden={cfg['hidden_dims']}, lr={cfg['lr']}, "
          f"epochs={cfg['epochs']}, MC_samples={cfg['mc_samples']}, device={device}")

    X_all = df[features].values.astype(np.float32)
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    X_scaled = scaler.transform(X_all).astype(np.float32)

    bnn_results = {}
    bnn_models = {}
    total_targets = len(WQ_TARGETS)

    for t_idx, target in enumerate(WQ_TARGETS, 1):
        tname = short(target)
        y = df[target].values.astype(np.float32)
        mask = ~np.isnan(y)
        X_t = X_scaled[mask]
        y_t = y[mask]

        # Target-wise standardisation for stable BNN training
        y_mean, y_std = y_t.mean(), y_t.std() + 1e-8
        y_norm = (y_t - y_mean) / y_std

        # ── Spatial CV (GroupKFold) to get honest metrics ──
        groups = df.loc[mask, "stationId"].values
        n_stations = len(np.unique(groups))
        gkf = GroupKFold(n_splits=min(n_stations, 5))
        n_folds = min(n_stations, 5)

        y_pred_cv = np.full(len(y_t), np.nan)
        y_unc_cv = np.full(len(y_t), np.nan)

        print(f"\n  ── [{t_idx}/{total_targets}] {tname} ({sum(mask)} samples, {n_folds} folds) ──")

        for fold_i, (train_idx, test_idx) in enumerate(gkf.split(X_t, y_norm, groups)):
            fold_start = __import__('time').time()
            print(f"    Fold {fold_i+1}/{n_folds}  "
                  f"(train={len(train_idx)}, test={len(test_idx)})", end="", flush=True)
            X_tr = torch.tensor(X_t[train_idx], dtype=torch.float32).to(device)
            y_tr = torch.tensor(y_norm[train_idx], dtype=torch.float32).unsqueeze(1).to(device)
            X_te = torch.tensor(X_t[test_idx], dtype=torch.float32).to(device)

            model = BayesianNeuralNetwork(
                n_features=X_tr.shape[1], n_targets=1,
                hidden_dims=cfg["hidden_dims"], dropout=cfg["dropout"],
                prior_sigma=cfg["prior_sigma"],
            ).to(device)

            optimizer = torch.optim.Adam(
                model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=10
            )
            criterion = ELBOLoss(n_train_samples=len(train_idx))

            dataset = TensorDataset(X_tr, y_tr)
            loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)

            best_loss = float("inf")
            patience_counter = 0
            best_state = None

            for epoch in range(cfg["epochs"]):
                model.train()
                epoch_loss = 0.0
                epoch_kl = 0.0
                for xb, yb in loader:
                    optimizer.zero_grad()
                    pred = model(xb)
                    kl = model.kl_loss()
                    loss = criterion(pred, yb, kl)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    epoch_loss += loss.item() * xb.size(0)
                    epoch_kl += kl.item() * xb.size(0)

                epoch_loss /= len(train_idx)
                epoch_kl /= len(train_idx)
                cur_lr = optimizer.param_groups[0]['lr']
                scheduler.step(epoch_loss)

                # Print every 50 epochs or at key milestones
                if (epoch + 1) % 50 == 0 or epoch == 0:
                    print(f"\r    Fold {fold_i+1}/{n_folds}  "
                          f"Epoch {epoch+1:3d}/{cfg['epochs']}  "
                          f"ELBO={epoch_loss:.6f}  KL={epoch_kl:.4f}  "
                          f"lr={cur_lr:.1e}  best={best_loss:.6f}",
                          end="", flush=True)

                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= cfg["patience"]:
                        print(f"\r    Fold {fold_i+1}/{n_folds}  "
                              f"Early stop at epoch {epoch+1}  "
                              f"best_ELBO={best_loss:.6f}              ")
                        break

            # Restore best checkpoint
            if best_state:
                model.load_state_dict(best_state)

            fold_time = __import__('time').time() - fold_start
            print(f"\r    Fold {fold_i+1}/{n_folds}  "
                  f"DONE  epochs={epoch+1}  best_ELBO={best_loss:.6f}  "
                  f"({fold_time:.1f}s)              ")

            # ── MC inference ──
            preds = _bnn_mc_predict(model, X_te, n_samples=cfg["mc_samples"])
            pred_mean = preds.mean(axis=0).squeeze()
            pred_std = preds.std(axis=0).squeeze()

            # Denormalise
            y_pred_cv[test_idx] = pred_mean * y_std + y_mean
            y_unc_cv[test_idx] = pred_std * y_std

        valid = ~np.isnan(y_pred_cv)
        r2 = r2_score(y_t[valid], y_pred_cv[valid])
        rmse = np.sqrt(mean_squared_error(y_t[valid], y_pred_cv[valid]))
        mae = mean_absolute_error(y_t[valid], y_pred_cv[valid])
        mean_unc = float(np.nanmean(y_unc_cv))

        bnn_results[target] = {
            "R2": r2, "RMSE": rmse, "MAE": mae, "mean_uncertainty": mean_unc,
        }
        ml_results[target]["BNN"] = {"R2": r2, "RMSE": rmse, "MAE": mae}
        print(f"  >>> [{t_idx}/{total_targets}] {tname}  "
              f"R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}  unc={mean_unc:.4f}")

        # ── Train final model on ALL data ──
        X_full_t = torch.tensor(X_t, dtype=torch.float32).to(device)
        y_full_t = torch.tensor(y_norm, dtype=torch.float32).unsqueeze(1).to(device)
        final_model = BayesianNeuralNetwork(
            n_features=X_full_t.shape[1], n_targets=1,
            hidden_dims=cfg["hidden_dims"], dropout=cfg["dropout"],
            prior_sigma=cfg["prior_sigma"],
        ).to(device)

        opt = torch.optim.Adam(final_model.parameters(), lr=cfg["lr"],
                               weight_decay=cfg["weight_decay"])
        crit = ELBOLoss(n_train_samples=len(y_norm))
        ds = TensorDataset(X_full_t, y_full_t)
        dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True)

        for epoch in range(cfg["epochs"]):
            final_model.train()
            epoch_loss = 0.0
            for xb, yb in dl:
                opt.zero_grad()
                p = final_model(xb)
                loss = crit(p, yb, final_model.kl_loss())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(final_model.parameters(), 5.0)
                opt.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(y_norm)
            if (epoch + 1) % 50 == 0 or epoch == 0:
                print(f"    Final model: epoch {epoch+1:3d}/{cfg['epochs']}  "
                      f"ELBO={epoch_loss:.6f}", flush=True)

        bnn_models[target] = {
            "model": final_model,
            "y_mean": float(y_mean),
            "y_std": float(y_std),
        }

    # Save BNN models
    torch.save(bnn_models, MODELS_DIR / "bnn_models.pt")
    print("\n  BNN models saved to models/bnn_models.pt")

    # ── BNN-specific plots ──
    _plot_bnn_uncertainty(df, features, bnn_models, bnn_results)
    _plot_bnn_comparison(bnn_results)

    return bnn_results, bnn_models


def _bnn_mc_predict(
    model: BayesianNeuralNetwork,
    X: torch.Tensor,
    n_samples: int = 50,
) -> np.ndarray:
    """Run T stochastic forward passes and return (T, N, out) predictions."""
    model.eval()  # keep dropout active for MC
    # Re-enable dropout at test time for MC
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = model(X)
            preds.append(out.detach().cpu().numpy())
    return np.array(preds)


def _plot_bnn_uncertainty(df, features, bnn_models, bnn_results):
    """Plot BNN predictions with uncertainty bands for priority targets."""
    print("\n  --- BNN Uncertainty Plots ---")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    X_scaled = scaler.transform(df[features].values).astype(np.float32)
    device = next(iter(bnn_models.values()))["model"].parameters().__next__().device

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    for i, target in enumerate(PRIORITY_TARGETS):
        ax = axes.ravel()[i]
        tname = short(target)
        if target not in bnn_models:
            continue

        y_true = df[target].values
        mask = ~np.isnan(y_true)
        X_t = torch.tensor(X_scaled[mask], dtype=torch.float32).to(device)

        info = bnn_models[target]
        preds = _bnn_mc_predict(info["model"], X_t, n_samples=BNN_CONFIG["mc_samples"])
        pred_mean = preds.mean(axis=0).squeeze() * info["y_std"] + info["y_mean"]
        pred_std = preds.std(axis=0).squeeze() * info["y_std"]

        # Sort by true value for cleaner plot
        order = np.argsort(y_true[mask])
        x_axis = np.arange(len(order))
        y_sorted = y_true[mask][order]
        m_sorted = pred_mean[order]
        s_sorted = pred_std[order]

        ax.plot(x_axis, y_sorted, "k.", markersize=1, alpha=0.4, label="Actual")
        ax.plot(x_axis, m_sorted, color="#1565C0", linewidth=0.8, label="BNN mean")
        ax.fill_between(x_axis,
                        m_sorted - 1.96 * s_sorted,
                        m_sorted + 1.96 * s_sorted,
                        alpha=0.25, color="#42A5F5", label="95% CI")
        ax.set_title(f"{tname} — BNN Prediction + Uncertainty", fontweight="bold")
        ax.set_xlabel("Sample (sorted by actual)")
        ax.set_ylabel(tname)
        ax.legend(fontsize=8)

    fig.suptitle("Bayesian Neural Network: Predictions with 95% Credible Intervals",
                 fontsize=14, fontweight="bold", y=1.01)
    save_fig(fig, "16_bnn_uncertainty_bands")


def _plot_bnn_comparison(bnn_results):
    """Bar chart: BNN R² and uncertainty for each target."""
    print("\n  --- BNN Performance Summary ---")
    targets_short = [short(t) for t in WQ_TARGETS if t in bnn_results]
    r2_vals = [bnn_results[t]["R2"] for t in WQ_TARGETS if t in bnn_results]
    unc_vals = [bnn_results[t]["mean_uncertainty"] for t in WQ_TARGETS if t in bnn_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    colors_r2 = ["#4CAF50" if r > 0.5 else "#FF9800" if r > 0 else "#F44336" for r in r2_vals]
    ax1.bar(targets_short, r2_vals, color=colors_r2, edgecolor="white")
    ax1.set_ylabel("R² Score", fontsize=12)
    ax1.set_title("BNN R² per Water Quality Parameter", fontweight="bold")
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.grid(axis="y", alpha=0.3)
    ax1.tick_params(axis="x", rotation=45)

    ax2.bar(targets_short, unc_vals, color="#42A5F5", edgecolor="white")
    ax2.set_ylabel("Mean Predictive Uncertainty (std)", fontsize=12)
    ax2.set_title("BNN Uncertainty per Parameter", fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    ax2.tick_params(axis="x", rotation=45)

    fig.suptitle("Bayesian Neural Network Performance", fontsize=14, fontweight="bold", y=1.02)
    save_fig(fig, "17_bnn_performance_summary")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4: BAYESIAN NETWORK (Deep Dive)
# ══════════════════════════════════════════════════════════════════════════════
def train_bayesian_network(
    df: pd.DataFrame,
    features: List[str],
) -> Tuple[BayesianNetwork, Dict]:
    print("\n" + "=" * 70)
    print("PHASE 4: BAYESIAN NETWORK — Structure & Parameter Learning")
    print("=" * 70)

    # Discretize continuous variables for BN
    disc_df = df[features + WQ_TARGETS].copy().dropna()

    # Use quantile-based discretization (5 bins)
    n_bins = 5
    bin_labels = {0: "VL", 1: "L", 2: "M", 3: "H", 4: "VH"}
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")

    all_cols = features + WQ_TARGETS
    disc_values = discretizer.fit_transform(disc_df[all_cols])
    disc_data = pd.DataFrame(disc_values.astype(int), columns=all_cols, index=disc_df.index)

    # Map to labels
    for col in disc_data.columns:
        disc_data[col] = disc_data[col].map(bin_labels)

    joblib.dump(discretizer, MODELS_DIR / "bn_discretizer.joblib")
    print(f"  Discretized {len(all_cols)} variables into {n_bins} quantile bins")
    print(f"  Training samples: {len(disc_data)}")

    # Structure learning with domain constraints
    # Satellite features can influence WQ (not vice versa)
    # WQ params can influence each other
    print("\n  Learning BN structure (Hill-Climb + BIC) ...")
    black_list = []
    # WQ should not cause satellite features
    for wq in WQ_TARGETS:
        for sat in features:
            black_list.append((wq, sat))

    hc = HillClimbSearch(disc_data)
    best_model = hc.estimate(
        scoring_method=K2(disc_data),
        black_list=black_list,
        max_iter=200,
    )

    print(f"  Learned DAG: {len(best_model.nodes())} nodes, {len(best_model.edges())} edges")

    # Build BN model
    bn = BayesianNetwork(best_model.edges())
    bn.fit(disc_data, estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=10)
    print("  Parameters fitted (Bayesian estimation, BDeu prior)")

    # Save model
    with (MODELS_DIR / "bayesian_network.pkl").open("wb") as fp:
        pickle.dump(bn, fp)
    print("  BN model saved to models/bayesian_network.pkl")

    # ── DAG Visualization ──
    _visualize_dag(bn, features)

    # ── Edge analysis ──
    edge_info = _analyze_edges(bn, features)

    # ── Sensitivity analysis (for priority targets) ──
    bn_metrics = _bn_sensitivity_and_inference(bn, disc_data, disc_df, discretizer,
                                                features, n_bins, bin_labels)

    return bn, bn_metrics


def _visualize_dag(bn: BayesianNetwork, features: List[str]):
    print("\n  --- DAG Visualization ---")
    G = nx.DiGraph(bn.edges())

    # Color nodes
    color_map = []
    for node in G.nodes():
        if node in features:
            color_map.append("#4FC3F7")  # blue for satellite
        elif node in WQ_TARGETS:
            color_map.append("#FF8A65")  # orange for WQ
        else:
            color_map.append("#E0E0E0")

    # Rename nodes for readability
    label_map = {}
    for n in G.nodes():
        if n in WQ_SHORT:
            label_map[n] = WQ_SHORT[n]
        else:
            label_map[n] = n.replace("wq_", "")

    fig, ax = plt.subplots(figsize=(24, 16))
    pos = nx.spring_layout(G, k=2.5, iterations=100, seed=SEED)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=color_map,
                           node_size=800, edgecolors="black", linewidths=1)
    nx.draw_networkx_labels(G, pos, labels=label_map, ax=ax, font_size=7, font_weight="bold")

    # Separate sat→wq edges from wq→wq edges
    sat_wq_edges = [(u, v) for u, v in G.edges() if u in features and v in WQ_TARGETS]
    wq_wq_edges = [(u, v) for u, v in G.edges() if u in WQ_TARGETS and v in WQ_TARGETS]
    sat_sat_edges = [(u, v) for u, v in G.edges() if u in features and v in features]
    other_edges = [(u, v) for u, v in G.edges()
                   if (u, v) not in sat_wq_edges + wq_wq_edges + sat_sat_edges]

    nx.draw_networkx_edges(G, pos, edgelist=sat_wq_edges, ax=ax,
                           edge_color="#1565C0", width=2, alpha=0.7,
                           arrows=True, arrowsize=15)
    nx.draw_networkx_edges(G, pos, edgelist=wq_wq_edges, ax=ax,
                           edge_color="#E65100", width=2, alpha=0.7,
                           arrows=True, arrowsize=15)
    nx.draw_networkx_edges(G, pos, edgelist=sat_sat_edges, ax=ax,
                           edge_color="#757575", width=1, alpha=0.4,
                           arrows=True, arrowsize=10)
    nx.draw_networkx_edges(G, pos, edgelist=other_edges, ax=ax,
                           edge_color="#9E9E9E", width=1, alpha=0.4,
                           arrows=True, arrowsize=10)

    ax.set_title("Learned Bayesian Network Structure\n"
                 "(Blue=Satellite → Orange=WQ, Blue arrows=Sat→WQ, Orange arrows=WQ→WQ)",
                 fontsize=14, fontweight="bold")
    ax.axis("off")
    save_fig(fig, "09_bayesian_network_dag", tight=False)

    # Simplified DAG — only sat→wq edges
    fig2, ax2 = plt.subplots(figsize=(16, 12))
    G_simple = nx.DiGraph(sat_wq_edges)
    pos2 = nx.bipartite_layout(G_simple,
                                [n for n in G_simple.nodes() if n in features],
                                align="horizontal")
    color_simple = ["#4FC3F7" if n in features else "#FF8A65" for n in G_simple.nodes()]
    label_simple = {n: label_map.get(n, n) for n in G_simple.nodes()}
    nx.draw_networkx_nodes(G_simple, pos2, ax=ax2, node_color=color_simple,
                           node_size=1000, edgecolors="black", linewidths=1)
    nx.draw_networkx_labels(G_simple, pos2, labels=label_simple, ax=ax2,
                            font_size=9, font_weight="bold")
    nx.draw_networkx_edges(G_simple, pos2, ax=ax2, edge_color="#1565C0",
                           width=2, alpha=0.7, arrows=True, arrowsize=15)
    ax2.set_title("Satellite → Water Quality Causal Pathways (Learned by BN)",
                  fontsize=14, fontweight="bold")
    ax2.axis("off")
    save_fig(fig2, "10_bn_sat_wq_pathways", tight=False)


def _analyze_edges(bn: BayesianNetwork, features: List[str]) -> pd.DataFrame:
    print("\n  --- Edge Analysis ---")
    edges_df = pd.DataFrame(list(bn.edges()), columns=["parent", "child"])
    edges_df["type"] = edges_df.apply(
        lambda r: "Satellite→WQ" if r["parent"] in features and r["child"] in WQ_TARGETS
        else "WQ→WQ" if r["parent"] in WQ_TARGETS and r["child"] in WQ_TARGETS
        else "Satellite→Satellite" if r["parent"] in features and r["child"] in features
        else "Other", axis=1
    )
    edges_df.to_csv(TABLES_DIR / "bn_edges.csv", index=False)
    type_counts = edges_df["type"].value_counts()
    for t, c in type_counts.items():
        print(f"    {t}: {c} edges")

    # Which satellite features influence each WQ parameter
    influence = {}
    for wq in WQ_TARGETS:
        parents = [e[0] for e in bn.edges() if e[1] == wq and e[0] in features]
        influence[short(wq)] = parents
        if parents:
            print(f"    {short(wq)} ← {parents}")
    with (TABLES_DIR / "bn_satellite_influence.json").open("w") as fp:
        json.dump(influence, fp, indent=2)

    return edges_df


def _bn_sensitivity_and_inference(
    bn, disc_data, disc_df, discretizer, features, n_bins, bin_labels
) -> Dict:
    print("\n  --- BN Inference & Cross-Validation ---")
    inference = VariableElimination(bn)
    bn_metrics = {}

    # Test on a subset (BN inference is slower)
    test_size = min(200, len(disc_data))
    test_indices = np.random.choice(disc_data.index, size=test_size, replace=False)

    for target in PRIORITY_TARGETS:
        tname = short(target)
        if target not in bn.nodes():
            print(f"    {tname}: not in BN — skipping")
            continue

        parents = list(bn.get_parents(target))
        if not parents:
            print(f"    {tname}: no parents in BN — skipping")
            continue

        correct = 0
        total = 0
        for idx in test_indices:
            evidence = {}
            skip = False
            for p in parents:
                val = disc_data.loc[idx, p]
                if pd.isna(val):
                    skip = True
                    break
                evidence[p] = val
            if skip or not evidence:
                continue
            try:
                query = inference.query([target], evidence=evidence, show_progress=False)
                predicted_bin = query.values.argmax()
                predicted_label = bin_labels[predicted_bin]
                actual = disc_data.loc[idx, target]
                if predicted_label == actual:
                    correct += 1
                total += 1
            except Exception:
                continue

        acc = correct / total if total > 0 else 0
        bn_metrics[target] = {"accuracy": acc, "total_tested": total}
        print(f"    {tname}: BN classification accuracy = {acc:.3f} ({total} samples)")

    return bn_metrics


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5: COMPREHENSIVE EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_all(
    df: pd.DataFrame,
    features: List[str],
    ml_results: Dict,
    bn_metrics: Dict,
    trained_models: Dict,
):
    print("\n" + "=" * 70)
    print("PHASE 5: COMPREHENSIVE EVALUATION")
    print("=" * 70)

    # ── Comparison table ──
    print("\n  --- Model Comparison Table ---")
    rows_list = []
    for target in WQ_TARGETS:
        tname = short(target)
        for mname, metrics in ml_results[target].items():
            rows_list.append({
                "Target": tname,
                "Model": mname,
                "R²": round(metrics["R2"], 4),
                "RMSE": round(metrics["RMSE"], 4),
                "MAE": round(metrics["MAE"], 4),
            })
        if target in bn_metrics:
            rows_list.append({
                "Target": tname,
                "Model": "Bayesian Network",
                "R²": "—",
                "RMSE": "—",
                "MAE": f"Acc={bn_metrics[target]['accuracy']:.3f}",
            })

    comp_df = pd.DataFrame(rows_list)
    comp_df.to_csv(TABLES_DIR / "model_comparison.csv", index=False)
    print(comp_df.to_string(index=False))

    # ── Best model per target ──
    print("\n  --- Best Model per Target (by R²) ---")
    best_rows = []
    for target in WQ_TARGETS:
        tname = short(target)
        best_m = max(ml_results[target].items(), key=lambda x: x[1]["R2"])
        best_rows.append({
            "Target": tname,
            "Best Model": best_m[0],
            "R²": round(best_m[1]["R2"], 4),
            "RMSE": round(best_m[1]["RMSE"], 4),
        })
        print(f"    {tname:6s} → {best_m[0]:25s} R²={best_m[1]['R2']:.4f}")
    pd.DataFrame(best_rows).to_csv(TABLES_DIR / "best_model_per_target.csv", index=False)

    # ── Feature importance (from RF) ──
    _plot_feature_importance(trained_models, features)

    # ── R² comparison bar chart (now includes BNN) ──
    _plot_model_comparison(ml_results)

    # ── Actual vs Predicted scatter plots ──
    _plot_actual_vs_predicted(df, features, trained_models)

    # ── Residual analysis ──
    _plot_residuals(df, features, trained_models)

    # ── LaTeX table ──
    _export_latex_table(ml_results, bn_metrics)

    # ── Temporal split validation ──
    _temporal_split_validation(df, features)


def _plot_feature_importance(trained_models: Dict, features: List[str]):
    print("\n  --- Feature Importance (Random Forest) ---")
    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    for i, target in enumerate(PRIORITY_TARGETS):
        ax = axes.ravel()[i]
        rf = trained_models[target].get("Random Forest")
        if rf is None:
            continue
        importances = rf.feature_importances_
        indices = np.argsort(importances)[-15:]  # top 15
        feat_names = [features[j] for j in indices]
        ax.barh(range(len(indices)), importances[indices], color="steelblue")
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels(feat_names, fontsize=8)
        ax.set_title(f"{short(target)} — RF Feature Importance", fontweight="bold")
        ax.set_xlabel("Importance")
    fig.suptitle("Top-15 Features per Water Quality Parameter (Random Forest)",
                 fontsize=14, fontweight="bold", y=1.01)
    save_fig(fig, "11_feature_importance_rf")

    # XGBoost importance
    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    for i, target in enumerate(PRIORITY_TARGETS):
        ax = axes.ravel()[i]
        xgb_model = trained_models[target].get("XGBoost")
        if xgb_model is None:
            continue
        importances = xgb_model.feature_importances_
        indices = np.argsort(importances)[-15:]
        feat_names = [features[j] for j in indices]
        ax.barh(range(len(indices)), importances[indices], color="coral")
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels(feat_names, fontsize=8)
        ax.set_title(f"{short(target)} — XGBoost Feature Importance", fontweight="bold")
        ax.set_xlabel("Importance")
    fig.suptitle("Top-15 Features per Water Quality Parameter (XGBoost)",
                 fontsize=14, fontweight="bold", y=1.01)
    save_fig(fig, "12_feature_importance_xgb")


def _plot_model_comparison(ml_results: Dict):
    print("\n  --- Model Comparison Chart ---")
    models_list = ["Linear Regression", "Random Forest", "XGBoost", "BNN"]
    r2_data = {m: [] for m in models_list}
    targets_short = []
    for target in WQ_TARGETS:
        targets_short.append(short(target))
        for m in models_list:
            r2_data[m].append(ml_results[target].get(m, {}).get("R2", 0))

    x = np.arange(len(targets_short))
    width = 0.2
    fig, ax = plt.subplots(figsize=(18, 7))
    colors = ["#90CAF9", "#4CAF50", "#FF7043", "#AB47BC"]
    for i, (m, col) in enumerate(zip(models_list, colors)):
        ax.bar(x + i * width, r2_data[m], width, label=m, color=col, edgecolor="white")
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(targets_short, fontsize=10, fontweight="bold")
    ax.set_ylabel("R² Score", fontsize=12)
    ax.set_title("Model Comparison: R² Across Water Quality Parameters\n"
                 "(Spatial Cross-Validation — includes Bayesian Neural Network)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, "13_model_comparison_r2")


def _plot_actual_vs_predicted(df: pd.DataFrame, features: List[str], trained_models: Dict):
    print("\n  --- Actual vs Predicted Plots ---")
    X = df[features].values
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    X_scaled = scaler.transform(X)

    # Use the best model (XGBoost typically)
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    for i, target in enumerate(PRIORITY_TARGETS):
        ax = axes.ravel()[i]
        y_true = df[target].values
        mask = ~np.isnan(y_true)

        # Pick best model
        best_name = "XGBoost"
        model = trained_models[target][best_name]
        y_pred = model.predict(X_scaled[mask])
        r2 = r2_score(y_true[mask], y_pred)

        ax.scatter(y_true[mask], y_pred, alpha=0.3, s=10, c="steelblue")
        lims = [min(y_true[mask].min(), y_pred.min()),
                max(y_true[mask].max(), y_pred.max())]
        ax.plot(lims, lims, "r--", linewidth=2, label="Perfect fit")
        ax.set_xlabel(f"Actual {short(target)}", fontsize=10)
        ax.set_ylabel(f"Predicted {short(target)}", fontsize=10)
        ax.set_title(f"{short(target)} (R²={r2:.3f}, {best_name})", fontweight="bold")
        ax.legend(fontsize=9)
    fig.suptitle("Actual vs Predicted — Best Model (Full Training Data)",
                 fontsize=14, fontweight="bold", y=1.01)
    save_fig(fig, "14_actual_vs_predicted")


def _plot_residuals(df: pd.DataFrame, features: List[str], trained_models: Dict):
    print("\n  --- Residual Analysis ---")
    X = df[features].values
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    X_scaled = scaler.transform(X)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    for i, target in enumerate(PRIORITY_TARGETS):
        ax = axes.ravel()[i]
        y_true = df[target].values
        mask = ~np.isnan(y_true)
        model = trained_models[target]["XGBoost"]
        y_pred = model.predict(X_scaled[mask])
        residuals = y_true[mask] - y_pred

        ax.scatter(y_pred, residuals, alpha=0.3, s=10, c="steelblue")
        ax.axhline(y=0, color="red", linestyle="--")
        ax.set_xlabel(f"Predicted {short(target)}")
        ax.set_ylabel("Residual")
        ax.set_title(f"{short(target)} Residuals", fontweight="bold")
    fig.suptitle("Residual Analysis (XGBoost)", fontsize=14, fontweight="bold", y=1.01)
    save_fig(fig, "15_residual_analysis")


def _export_latex_table(ml_results: Dict, bn_metrics: Dict):
    print("\n  --- LaTeX Table ---")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Model Performance Comparison (Spatial Cross-Validation)}",
        r"\label{tab:model_comparison}",
        r"\begin{tabular}{llrrr}",
        r"\hline",
        r"\textbf{Parameter} & \textbf{Model} & \textbf{R²} & \textbf{RMSE} & \textbf{MAE} \\",
        r"\hline",
    ]
    all_model_names = ["Linear Regression", "Random Forest", "XGBoost", "BNN"]
    for target in WQ_TARGETS:
        tname = short(target).replace("⁻", "$^-$").replace("₃", "$_3$")
        for mname in all_model_names:
            metrics = ml_results[target].get(mname)
            if metrics:
                lines.append(
                    f"  {tname} & {mname} & {metrics['R2']:.4f} & {metrics['RMSE']:.4f} & {metrics['MAE']:.4f} \\\\"
                )
        lines.append(r"\hline")
    lines += [r"\end{tabular}", r"\end{table}"]

    latex_path = TABLES_DIR / "model_comparison.tex"
    latex_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  LaTeX table saved: {latex_path.name}")


def _temporal_split_validation(df: pd.DataFrame, features: List[str]):
    print("\n  --- Temporal Split Validation ---")
    df_sorted = df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(df_sorted) * 0.8)
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[features])
    X_test = scaler.transform(test_df[features])

    temporal_results = {}
    for target in PRIORITY_TARGETS:
        tname = short(target)
        y_train = train_df[target].values
        y_test = test_df[target].values
        mask_tr = ~np.isnan(y_train)
        mask_te = ~np.isnan(y_test)

        rf = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5,
                                   n_jobs=-1, random_state=SEED)
        rf.fit(X_train[mask_tr], y_train[mask_tr])
        y_pred = rf.predict(X_test[mask_te])
        r2 = r2_score(y_test[mask_te], y_pred)
        rmse = np.sqrt(mean_squared_error(y_test[mask_te], y_pred))
        temporal_results[tname] = {"R2": r2, "RMSE": rmse}
        print(f"    {tname:6s}  R²={r2:.4f}  RMSE={rmse:.4f}  "
              f"(train={sum(mask_tr)}, test={sum(mask_te)})")

    pd.DataFrame(temporal_results).T.to_csv(TABLES_DIR / "temporal_validation.csv")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    setup_dirs()

    # Phase 1
    df = load_data()
    eda_statistical_profiling(df)
    corr_df = eda_correlation_analysis(df)
    eda_spatial_analysis(df)
    eda_temporal_analysis(df)

    # Phase 2
    df, features = engineer_features(df)

    # Phase 3
    ml_results, trained_models = train_all_models(df, features)

    # Phase 3B — Bayesian Neural Network
    bnn_results, bnn_models = train_bnn(df, features, ml_results)

    # Phase 4
    bn, bn_metrics = train_bayesian_network(df, features)

    # Phase 5
    evaluate_all(df, features, ml_results, bn_metrics, trained_models)

    print("\n" + "=" * 70)
    print("ALL PHASES COMPLETE")
    print("=" * 70)
    print(f"Results: {RESULTS_DIR}")
    print(f"Figures: {FIGURES_DIR}")
    print(f"Models:  {MODELS_DIR}")
    print(f"Tables:  {TABLES_DIR}")


if __name__ == "__main__":
    main()
