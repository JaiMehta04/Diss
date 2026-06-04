"""
generate_ppt.py
===============
Generate a PowerPoint presentation summarising all dissertation work:
  - Data collection (Sentinel-2 via GEE)
  - EDA & feature engineering
  - Bayesian Neural Network results
  - Hybrid CNN + Tabular results
  - Model comparison
  - Conclusions & next steps

Usage
-----
  pip install python-pptx   (if not already installed)
  python generate_ppt.py
"""

import csv
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.chart import XL_CHART_TYPE
from pptx.chart.data import CategoryChartData

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parents[2]
BNN_RESULTS = BASE / "GangaIndices_All" / "results"
CNN_RESULTS = BASE / "GangaIndices_All" / "results_cnn"
BNN_FIGS = BNN_RESULTS / "figures"
CNN_FIGS = CNN_RESULTS / "figures"
BNN_CSV = BNN_RESULTS / "tables" / "bnn_multioutput_results.csv"
CNN_CSV = CNN_RESULTS / "tables" / "cnn_results.csv"
OUTPUT_PPT = BASE / "Dissertation_Progress.pptx"

# ── Colours ───────────────────────────────────────────────────────────────────
DARK_BG   = RGBColor(0x1B, 0x1B, 0x2F)
ACCENT    = RGBColor(0x00, 0x96, 0xD6)
WHITE     = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GREY = RGBColor(0xCC, 0xCC, 0xCC)
GREEN     = RGBColor(0x4C, 0xAF, 0x50)
ORANGE    = RGBColor(0xFF, 0x98, 0x00)
RED       = RGBColor(0xF4, 0x43, 0x36)

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)


# ── Helpers ───────────────────────────────────────────────────────────────────
def set_slide_bg(slide, color):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_textbox(slide, left, top, width, height, text,
                font_size=18, bold=False, color=WHITE, align=PP_ALIGN.LEFT):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.alignment = align
    return tf


def add_bullet_slide(slide, left, top, width, height, bullets,
                     font_size=16, color=WHITE):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, b in enumerate(bullets):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = b
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.space_after = Pt(6)
        p.level = 0
    return tf


def add_image_safe(slide, img_path, left, top, width=None, height=None):
    """Add image to slide only if file exists."""
    if Path(img_path).exists():
        kwargs = {"left": left, "top": top}
        if width:
            kwargs["width"] = width
        if height:
            kwargs["height"] = height
        slide.shapes.add_picture(str(img_path), **kwargs)
        return True
    return False


def load_csv_results(path):
    """Load results CSV → list of dicts."""
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def add_results_table(slide, rows, left, top, has_uncertainty=False):
    """Add a formatted results table to the slide."""
    cols = 4 if not has_uncertainty else 5
    n_rows = len(rows) + 1  # +1 header
    table_shape = slide.shapes.add_table(
        n_rows, cols, left, top,
        Inches(6.5) if not has_uncertainty else Inches(8),
        Inches(0.3 * n_rows),
    )
    table = table_shape.table

    # Header
    headers = ["Parameter", "R²", "RMSE", "MAE"]
    if has_uncertainty:
        headers.append("Uncertainty")
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = h
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(12)
            p.font.bold = True
            p.font.color.rgb = WHITE
            p.alignment = PP_ALIGN.CENTER
        cell.fill.solid()
        cell.fill.fore_color.rgb = ACCENT

    # Data rows
    for i, row in enumerate(rows, 1):
        r2_val = float(row.get("R2", 0))
        row_color = GREEN if r2_val > 0.5 else ORANGE if r2_val > 0 else RED

        values = [
            row.get("Parameter", ""),
            row.get("R2", ""),
            row.get("RMSE", ""),
            row.get("MAE", ""),
        ]
        if has_uncertainty:
            values.append(row.get("Mean_Uncertainty", ""))

        for j, v in enumerate(values):
            cell = table.cell(i, j)
            cell.text = str(v)
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(11)
                p.font.color.rgb = WHITE if j > 0 else LIGHT_GREY
                p.alignment = PP_ALIGN.CENTER
            if j == 1:  # R² column colored
                cell.fill.solid()
                cell.fill.fore_color.rgb = row_color
            else:
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(0x2A, 0x2A, 0x45)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE BUILDERS
# ══════════════════════════════════════════════════════════════════════════════
def slide_title(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    set_slide_bg(slide, DARK_BG)
    add_textbox(slide, Inches(1), Inches(1.8), Inches(11), Inches(1.5),
                "Remote Sensing of Water Quality\nin the Ganga River Basin",
                font_size=36, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_textbox(slide, Inches(1), Inches(3.8), Inches(11), Inches(0.8),
                "Using Sentinel-2 Satellite Imagery, Bayesian Neural Networks,\n"
                "and Hybrid CNN Models",
                font_size=20, color=ACCENT, align=PP_ALIGN.CENTER)
    add_textbox(slide, Inches(1), Inches(5.5), Inches(11), Inches(0.6),
                "Dissertation Progress Presentation",
                font_size=16, color=LIGHT_GREY, align=PP_ALIGN.CENTER)


def slide_outline(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, DARK_BG)
    add_textbox(slide, Inches(0.8), Inches(0.4), Inches(11), Inches(0.8),
                "Outline", font_size=30, bold=True, color=ACCENT)
    add_bullet_slide(slide, Inches(1), Inches(1.4), Inches(10), Inches(5), [
        "1.  Problem Statement & Research Objectives",
        "2.  Study Area — 37 CPCB Stations along the Ganga",
        "3.  Data Collection — Sentinel-2 via Google Earth Engine",
        "4.  Exploratory Data Analysis & Feature Engineering",
        "5.  Model 1: Bayesian Neural Network (BNN) — Tabular Features",
        "6.  Model 2: Hybrid CNN + Tabular — Image Patches",
        "7.  Model Comparison & Discussion",
        "8.  Conclusions & Next Steps",
    ], font_size=18, color=WHITE)


def slide_problem(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, DARK_BG)
    add_textbox(slide, Inches(0.8), Inches(0.4), Inches(11), Inches(0.8),
                "Problem Statement", font_size=30, bold=True, color=ACCENT)
    add_bullet_slide(slide, Inches(1), Inches(1.5), Inches(10.5), Inches(5), [
        "• The Ganga supports ~500 million people — continuous WQ monitoring is critical",
        "• In-situ sampling is expensive, sparse, and infrequent",
        "• Satellite remote sensing offers synoptic, repeatable, cost-effective monitoring",
        "• Challenge: Can we predict 12 water quality parameters from satellite imagery?",
        "",
        "Research Objectives:",
        "   1. Extract Sentinel-2 spectral features for 37 CPCB stations",
        "   2. Train a Bayesian Neural Network (BNN) on extracted band values",
        "   3. Train a Hybrid CNN on raw image patches (640m × 640m)",
        "   4. Compare approaches and quantify predictive uncertainty",
    ], font_size=16, color=WHITE)


def slide_study_area(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, DARK_BG)
    add_textbox(slide, Inches(0.8), Inches(0.4), Inches(11), Inches(0.8),
                "Study Area — 37 CPCB Monitoring Stations", font_size=30, bold=True, color=ACCENT)
    add_bullet_slide(slide, Inches(0.8), Inches(1.5), Inches(5.5), Inches(5), [
        "• 37 stations spanning Gangotri to the Bay of Bengal",
        "• Data period: ~100 days per station (CPCB daily WQ)",
        "• 75,801 raw measurements → 3,690 unique station-days",
        "• 12 WQ parameters monitored:",
        "   BOD, COD, Chloride, Conductivity, Depth, DO,",
        "   Nitrate, TOC, Water Level, Temperature, Turbidity, pH",
        "",
        "• Coordinates from CPCB station registry",
        "• Aligned with Sentinel-2 revisit cadence (5-day repeat)",
    ], font_size=15, color=WHITE)
    # Add spatial variation plot if available
    add_image_safe(slide, BNN_FIGS / "06_spatial_variation.png",
                   Inches(7), Inches(1.3), width=Inches(5.8))


def slide_data_collection(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, DARK_BG)
    add_textbox(slide, Inches(0.8), Inches(0.4), Inches(11), Inches(0.8),
                "Data Collection — Sentinel-2 via Google Earth Engine",
                font_size=28, bold=True, color=ACCENT)
    add_bullet_slide(slide, Inches(0.8), Inches(1.5), Inches(11), Inches(5.5), [
        "Tabular Extraction (download_all_indices.py):",
        "   • Collection: COPERNICUS/S2_SR_HARMONIZED (Level-2A Surface Reflectance)",
        "   • 10 spectral bands: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12",
        "   • 8 WQ-relevant indices: NDWI, MNDWI, NDTI, NDVI, NDCI, MCI, FAI, SABI",
        "   • Cascading search windows: ±3d/20% cloud → ±10d/60% → ±30d/100%",
        "   • 3,557 of 3,690 station-days successfully matched (96.4%)",
        "   • 4 parallel GEE workers, 39.2 min total runtime",
        "",
        "Image Patch Extraction (download_patches.py):",
        "   • 64×64 pixel patches (640m × 640m at 10m resolution)",
        "   • Same cascading search windows as tabular",
        "   • Saved as .npy files — automatic skip if already downloaded",
        "   • sampleRectangle() API — direct numpy download, no Google Drive",
    ], font_size=14, color=WHITE)


def slide_eda_distributions(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, DARK_BG)
    add_textbox(slide, Inches(0.8), Inches(0.4), Inches(11), Inches(0.8),
                "EDA — Feature & Target Distributions", font_size=28, bold=True, color=ACCENT)
    add_image_safe(slide, BNN_FIGS / "01_satellite_distributions.png",
                   Inches(0.3), Inches(1.3), width=Inches(6.3))
    add_image_safe(slide, BNN_FIGS / "02_wq_distributions.png",
                   Inches(6.8), Inches(1.3), width=Inches(6.3))


def slide_eda_correlation(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, DARK_BG)
    add_textbox(slide, Inches(0.8), Inches(0.4), Inches(11), Inches(0.8),
                "EDA — Satellite ↔ Water Quality Correlations",
                font_size=28, bold=True, color=ACCENT)
    add_image_safe(slide, BNN_FIGS / "03_sat_wq_correlation_heatmap.png",
                   Inches(0.3), Inches(1.3), width=Inches(6.3))
    add_image_safe(slide, BNN_FIGS / "04_sat_wq_spearman_heatmap.png",
                   Inches(6.8), Inches(1.3), width=Inches(6.3))


def slide_eda_seasonal(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, DARK_BG)
    add_textbox(slide, Inches(0.8), Inches(0.4), Inches(11), Inches(0.8),
                "EDA — Seasonal & Spatial Variation", font_size=28, bold=True, color=ACCENT)
    add_image_safe(slide, BNN_FIGS / "07_seasonal_variation.png",
                   Inches(0.3), Inches(1.3), width=Inches(6.3))
    add_image_safe(slide, BNN_FIGS / "08_monthly_index_trends.png",
                   Inches(6.8), Inches(1.3), width=Inches(6.3))


def slide_feature_engineering(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, DARK_BG)
    add_textbox(slide, Inches(0.8), Inches(0.4), Inches(11), Inches(0.8),
                "Feature Engineering", font_size=30, bold=True, color=ACCENT)
    add_bullet_slide(slide, Inches(0.8), Inches(1.5), Inches(5.5), Inches(5), [
        "Satellite Features (18):",
        "   • 10 spectral bands (B2–B12)",
        "   • 8 water quality indices",
        "",
        "Engineered Features (9):",
        "   • month_sin, month_cos (temporal cyclicity)",
        "   • dist_from_source_km (Gangotri reference)",
        "   • Band ratios: B5/B4, B3/B2, B4/B3,",
        "     B8/B4, B11/B8, B5/B6",
        "",
        "Post-cleaning: 27 features",
        "   (dropped >0.95 inter-correlated)",
        "",
        "Target preparation:",
        "   • KNN imputation (k=5) for missing WQ",
        "   • Z-score normalisation per target",
        "   • Log-transform on BOD, COD, EC, TOC, Turb (CNN)",
    ], font_size=14, color=WHITE)
    add_image_safe(slide, BNN_FIGS / "05_feature_intercorrelation.png",
                   Inches(7), Inches(1.3), width=Inches(5.8))


def slide_bnn_architecture(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, DARK_BG)
    add_textbox(slide, Inches(0.8), Inches(0.4), Inches(11), Inches(0.8),
                "Model 1: Bayesian Neural Network (BNN)",
                font_size=28, bold=True, color=ACCENT)
    add_bullet_slide(slide, Inches(0.8), Inches(1.5), Inches(5.5), Inches(5.5), [
        "Architecture: Bayes by Backprop",
        "   • Gaussian posteriors on all weights: q(w) = N(μ, σ²)",
        "   • Local reparameterization trick (lower variance)",
        "   • ELBO = NLL + (kl_weight / N) · KL(q || prior)",
        "",
        "Network:",
        "   Input(27) → BayesLinear(512) → BN → LeakyReLU → Drop(0.05)",
        "            → BayesLinear(256) → BN → LeakyReLU → Drop(0.05)",
        "            → BayesLinear(128) → BN → LeakyReLU → Drop(0.05)",
        "            → BayesLinear(64)  → BN → LeakyReLU",
        "            → BayesLinear(12)  ← ALL 12 WQ outputs",
        "",
        "Training:",
        "   • KL annealing warm-up: 0→1 over 100 epochs",
        "   • Cosine annealing LR (5e-4 → 1e-6)",
        "   • Early stopping on NLL (not ELBO), patience=80",
        "   • 50 MC forward passes for uncertainty estimation",
    ], font_size=13, color=WHITE)
    add_bullet_slide(slide, Inches(7), Inches(1.5), Inches(5.5), Inches(3), [
        "Validation:",
        "   • Spatial GroupKFold (leave-station-out, 10 folds)",
        "   • Prevents spatial data leakage",
        "",
        "Uncertainty:",
        "   • Epistemic: weight distributions",
        "   • MC dropout at inference time",
        "   • Per-sample credible intervals",
    ], font_size=13, color=WHITE)


def slide_bnn_results(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, DARK_BG)
    add_textbox(slide, Inches(0.8), Inches(0.4), Inches(11), Inches(0.8),
                "BNN — Cross-Validation Results", font_size=28, bold=True, color=ACCENT)
    rows = load_csv_results(BNN_CSV)
    if rows:
        add_results_table(slide, rows, Inches(0.5), Inches(1.4), has_uncertainty=True)
    add_image_safe(slide, BNN_FIGS / "18_bnn_r2_uncertainty.png",
                   Inches(9), Inches(1.3), width=Inches(4))


def slide_bnn_scatter(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, DARK_BG)
    add_textbox(slide, Inches(0.8), Inches(0.4), Inches(11), Inches(0.8),
                "BNN — Actual vs Predicted", font_size=28, bold=True, color=ACCENT)
    add_image_safe(slide, BNN_FIGS / "16_bnn_actual_vs_predicted.png",
                   Inches(0.5), Inches(1.2), width=Inches(12))


def slide_bnn_uncertainty(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, DARK_BG)
    add_textbox(slide, Inches(0.8), Inches(0.4), Inches(11), Inches(0.8),
                "BNN — Uncertainty Bands (95% CI)", font_size=28, bold=True, color=ACCENT)
    add_image_safe(slide, BNN_FIGS / "17_bnn_uncertainty_bands.png",
                   Inches(0.5), Inches(1.2), width=Inches(12))


def slide_cnn_architecture(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, DARK_BG)
    add_textbox(slide, Inches(0.8), Inches(0.4), Inches(11), Inches(0.8),
                "Model 2: Hybrid CNN + Tabular", font_size=28, bold=True, color=ACCENT)
    add_bullet_slide(slide, Inches(0.8), Inches(1.5), Inches(5.5), Inches(5.5), [
        "Input: Raw 64×64 Sentinel-2 Patches (10 bands)",
        "   + 5 on-the-fly spectral indices (NDWI, NDVI, NDCI, NDTI, MNDWI)",
        "   = 15-channel input (15 × 64 × 64)",
        "",
        "CNN Branch:",
        "   Conv(15,32,3) → BN → ReLU → MaxPool",
        "   Conv(32,48,3) → BN → ReLU → MaxPool",
        "   Conv(48,64,3) → BN → ReLU → AdaptiveAvgPool(1) → (64,)",
        "",
        "Tabular Branch:",
        "   [lat, lon, month_sin, month_cos, dist_from_source]",
        "   Linear(5,32) → BN → ReLU → (32,)",
        "",
        "Merged Head:",
        "   Concat(64+32=96) → Linear(96,64) → BN → ReLU",
        "   → Dropout(0.25) → Linear(64,12)",
    ], font_size=13, color=WHITE)
    add_bullet_slide(slide, Inches(7), Inches(1.5), Inches(5.5), Inches(3.5), [
        "Key Improvements over v1 (pure CNN):",
        "   • Hybrid: spatial + tabular context",
        "   • 15 channels vs 10 (spectral indices)",
        "   • 10× lighter model (~45K vs ~500K params)",
        "   • Huber loss (robust to outliers)",
        "   • Log-transform on skewed targets",
        "   • Gaussian noise augmentation (σ=0.02)",
        "",
        "Training:",
        "   • 250 epochs, lr=5e-4, patience=40",
        "   • Cosine annealing LR, AdamW",
        "   • Spatial GroupKFold (10 folds)",
        "   • GPU accelerated (Quadro T2000, CUDA 12.8)",
    ], font_size=13, color=WHITE)


def slide_cnn_results(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, DARK_BG)
    add_textbox(slide, Inches(0.8), Inches(0.4), Inches(11), Inches(0.8),
                "Hybrid CNN — Cross-Validation Results", font_size=28, bold=True, color=ACCENT)
    rows = load_csv_results(CNN_CSV)
    if rows:
        add_results_table(slide, rows, Inches(0.5), Inches(1.4), has_uncertainty=False)
    add_image_safe(slide, CNN_FIGS / "cnn_r2_bars.png",
                   Inches(7.5), Inches(1.3), width=Inches(5.5))


def slide_cnn_scatter(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, DARK_BG)
    add_textbox(slide, Inches(0.8), Inches(0.4), Inches(11), Inches(0.8),
                "Hybrid CNN — Actual vs Predicted", font_size=28, bold=True, color=ACCENT)
    add_image_safe(slide, CNN_FIGS / "cnn_actual_vs_predicted.png",
                   Inches(0.5), Inches(1.2), width=Inches(12))


def slide_comparison(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, DARK_BG)
    add_textbox(slide, Inches(0.8), Inches(0.4), Inches(11), Inches(0.8),
                "Model Comparison — BNN vs Hybrid CNN",
                font_size=28, bold=True, color=ACCENT)

    bnn_rows = load_csv_results(BNN_CSV)
    cnn_rows = load_csv_results(CNN_CSV)

    if bnn_rows and cnn_rows:
        # Build comparison table
        n_rows = len(bnn_rows) + 1
        table_shape = slide.shapes.add_table(
            n_rows, 5, Inches(0.5), Inches(1.4),
            Inches(8), Inches(0.3 * n_rows),
        )
        table = table_shape.table

        headers = ["Parameter", "BNN R²", "CNN R²", "BNN RMSE", "CNN RMSE"]
        for j, h in enumerate(headers):
            cell = table.cell(0, j)
            cell.text = h
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(12)
                p.font.bold = True
                p.font.color.rgb = WHITE
                p.alignment = PP_ALIGN.CENTER
            cell.fill.solid()
            cell.fill.fore_color.rgb = ACCENT

        for i, (bnn, cnn) in enumerate(zip(bnn_rows, cnn_rows), 1):
            bnn_r2 = float(bnn.get("R2", 0))
            cnn_r2 = float(cnn.get("R2", 0))
            better_r2 = "bnn" if bnn_r2 >= cnn_r2 else "cnn"

            bnn_rmse = float(bnn.get("RMSE", 999))
            cnn_rmse = float(cnn.get("RMSE", 999))
            better_rmse = "bnn" if bnn_rmse <= cnn_rmse else "cnn"

            values = [
                bnn.get("Parameter", ""),
                f"{bnn_r2:.4f}",
                f"{cnn_r2:.4f}",
                f"{bnn_rmse:.2f}",
                f"{cnn_rmse:.2f}",
            ]
            for j, v in enumerate(values):
                cell = table.cell(i, j)
                cell.text = v
                for p in cell.text_frame.paragraphs:
                    p.font.size = Pt(11)
                    p.font.color.rgb = WHITE
                    p.alignment = PP_ALIGN.CENTER
                cell.fill.solid()
                if j == 1:
                    cell.fill.fore_color.rgb = GREEN if better_r2 == "bnn" else RGBColor(0x2A, 0x2A, 0x45)
                elif j == 2:
                    cell.fill.fore_color.rgb = GREEN if better_r2 == "cnn" else RGBColor(0x2A, 0x2A, 0x45)
                elif j == 3:
                    cell.fill.fore_color.rgb = GREEN if better_rmse == "bnn" else RGBColor(0x2A, 0x2A, 0x45)
                elif j == 4:
                    cell.fill.fore_color.rgb = GREEN if better_rmse == "cnn" else RGBColor(0x2A, 0x2A, 0x45)
                else:
                    cell.fill.fore_color.rgb = RGBColor(0x2A, 0x2A, 0x45)

    add_bullet_slide(slide, Inches(9), Inches(1.5), Inches(4), Inches(5), [
        "Key Observations:",
        "• BNN excels at Water Level (R²=0.75)",
        "  driven by dist_from_source",
        "• BNN better at EC (R²=0.27 vs −0.08)",
        "• CNN captures spatial texture",
        "  but limited by ~3,500 samples",
        "• Both struggle with BOD, COD, TOC",
        "  (weak spectral signatures)",
        "• BNN provides uncertainty estimates",
        "  — critical for decision-making",
    ], font_size=13, color=WHITE)


def slide_conclusions(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, DARK_BG)
    add_textbox(slide, Inches(0.8), Inches(0.4), Inches(11), Inches(0.8),
                "Conclusions & Next Steps", font_size=30, bold=True, color=ACCENT)
    add_bullet_slide(slide, Inches(0.8), Inches(1.5), Inches(5.5), Inches(5.5), [
        "What Worked:",
        "   ✓ Automated Sentinel-2 extraction for 3,557 station-days",
        "   ✓ BNN captures Water Level (R²=0.75) and EC (R²=0.27)",
        "   ✓ Uncertainty quantification via Bayesian inference",
        "   ✓ Spatial CV prevents over-optimistic estimates",
        "   ✓ GPU-accelerated training pipeline (CUDA 12.8)",
        "",
        "Challenges:",
        "   ✗ Most WQ parameters have weak spectral signal",
        "   ✗ ~3,500 samples is small for deep learning",
        "   ✗ Cloud cover limits temporal match quality",
        "   ✗ 500m buffer may mix land/water pixels",
    ], font_size=14, color=WHITE)
    add_bullet_slide(slide, Inches(7), Inches(1.5), Inches(5.5), Inches(5.5), [
        "Next Steps:",
        "   → Test on a different river (transferability)",
        "   → Add Sentinel-1 SAR data (cloud-independent)",
        "   → Multi-temporal stacking (time-series input)",
        "   → Attention mechanism in CNN for band selection",
        "   → Ensemble: BNN + CNN + XGBoost fusion",
        "   → Larger dataset: extend temporal coverage",
        "",
        "Dissertation Deliverables:",
        "   • Full ML pipeline (data → models → prediction)",
        "   • Transferable predict_new_river.py script",
        "   • Comparative analysis: BNN vs CNN vs traditional ML",
        "   • Publication-ready figures and LaTeX tables",
    ], font_size=14, color=WHITE)


def slide_thank_you(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, DARK_BG)
    add_textbox(slide, Inches(1), Inches(2.5), Inches(11), Inches(1.5),
                "Thank You", font_size=44, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)
    add_textbox(slide, Inches(1), Inches(4.2), Inches(11), Inches(0.8),
                "Questions & Discussion", font_size=22, color=LIGHT_GREY, align=PP_ALIGN.CENTER)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H

    # Build all slides
    slide_title(prs)
    slide_outline(prs)
    slide_problem(prs)
    slide_study_area(prs)
    slide_data_collection(prs)
    slide_eda_distributions(prs)
    slide_eda_correlation(prs)
    slide_eda_seasonal(prs)
    slide_feature_engineering(prs)
    slide_bnn_architecture(prs)
    slide_bnn_results(prs)
    slide_bnn_scatter(prs)
    slide_bnn_uncertainty(prs)
    slide_cnn_architecture(prs)
    slide_cnn_results(prs)
    slide_cnn_scatter(prs)
    slide_comparison(prs)
    slide_conclusions(prs)
    slide_thank_you(prs)

    prs.save(str(OUTPUT_PPT))
    print(f"Presentation saved → {OUTPUT_PPT}")
    print(f"  {len(prs.slides)} slides")


if __name__ == "__main__":
    main()
