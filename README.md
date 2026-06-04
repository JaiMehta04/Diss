# 🌊 Ganga River Water Quality Prediction using Sentinel-2 Satellite Imagery

> **M.Tech Dissertation** — SVNIT, Surat | Department of Computer Science & Engineering  
> A deep learning framework for predicting **12 water quality parameters** from Sentinel-2 multispectral satellite data across **37 CPCB monitoring stations** (Gangotri → Bay of Bengal).

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-API-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 Problem Statement

Traditional water quality monitoring of the Ganga River relies on sparse, expensive, and delayed lab sampling at fixed stations. This project builds an end-to-end pipeline that uses **freely available Sentinel-2 satellite imagery** to predict water quality parameters in near real-time at any point along the river.

---

## 🏗️ Research Pipeline (Step by Step)

The project follows a **4-phase pipeline** — each phase builds on the previous:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        RESEARCH WORKFLOW                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Phase 1                Phase 2               Phase 3          Phase 4   │
│  ┌──────────┐          ┌──────────┐          ┌─────────┐     ┌───────┐ │
│  │ Data     │  ──────► │ Satellite│  ──────► │ Model   │ ──► │Deploy │ │
│  │ Fusion   │          │ Features │          │ Training│     │& Infer│ │
│  └──────────┘          └──────────┘          └─────────┘     └───────┘ │
│   ✅ Complete            ✅ Complete           ✅ Complete      ⬜ Next   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Phase 1: Multi-Source Data Integration

**Goal**: Fuse water quality records from 3 heterogeneous government sources into a single, quality-controlled dataset.

### Data Sources

| Source | Type | Stations | Parameters | Period | Trust |
|--------|------|----------|------------|--------|-------|
| CPCB RTWQMS | Real-time sensors (CSV) | 37 | 10 | 2025 | High |
| data.gov.in | Historical API (6 CSVs) | 496 | 23 | 1963–2022 | Medium |
| NWMP Reports | PDF annual reports (7 docs) | 170 | 7 | 2015–2024 | Medium |
| **Fused Output** | **Unified dataset** | **696** | **24** | **1963–2025** | — |

### 7-Stage Pipeline

```
Raw Sources (1.09M records)
    │
    ├─[1] Ingestion ─── CSV parsers + PDF table extraction (pdfplumber)
    ├─[2] Standardization ─── Schema alignment, unit conversion → 950,896
    ├─[3] Spatial Validation ─── Ganga basin bounding box filter → 950,771
    ├─[4] QA/QC ─── Physical range checks + outlier detection → 944,979
    ├─[5] Consistency ─── Cross-parameter rules (BOD≤COD, TDS/EC) → flagged
    ├─[6] Fusion ─── Trust-weighted dedup, conflict resolution → 175,270
    └─[7] Output ─── Wide (14,270 × 35) + Long (175,270 × 13) CSVs
```

### Quality Results

| Metric | Value |
|--------|-------|
| Records passing all QA/QC | 80.8% (141,639 / 175,270) |
| Spatial coverage (with coords) | 93.6% |
| BOD ≤ COD physical consistency | 99.8% |
| pH within [5.5, 10.0] | 99.1% |
| ML-ready subset (Sentinel-2 era) | 5,737 station-days |

### Run It

```bash
python run_full_pipeline.py          # Execute full 7-stage pipeline
python validate_phase1.py            # Quick sanity checks
python integration_audit.py          # Generate audit figures
```

---

## 🛰️ Phase 2: Satellite Feature Extraction

**Goal**: Extract Sentinel-2 spectral bands and water quality indices matched to ground-truth station-days.

### What's Extracted

| Feature Type | Count | Description |
|-------------|-------|-------------|
| Spectral Bands | 10 | B2–B12 (10m–20m resolution) |
| Water Indices | 8 | NDWI, MNDWI, NDTI, NDVI, NDCI, MCI, FAI, SABI |
| Image Patches | 3,690 | 64×64 pixel crops around each station |

### Output

| File | Description | Size |
|------|-------------|------|
| `GangaIndices_All/satellite_indices.csv` | Tabular indices per station-day | 3,557 rows × 18 cols |
| `GangaIndices_All/water_quality_daily.csv` | Matched ground-truth WQ | 3,690 rows |
| `GangaIndices_All/merged_bn_dataset.csv` | **ML-ready joined dataset** | 3,557 × 35 |
| `SentinelPatches/` | Raw 64×64 `.npy` image patches | ~3,690 files |

### Run It

```bash
python src/data_collection/download_all_indices.py   # Extract spectral indices via GEE
python src/data_collection/download_patches.py       # Download 64×64 image patches
```

---

## 🧠 Phase 3: Model Training

**Goal**: Predict 12 WQ parameters (pH, DO, BOD, COD, Turbidity, etc.) from satellite features.

### Model A: Bayesian Neural Network (BNN)

| Aspect | Detail |
|--------|--------|
| Architecture | 512 → 256 → 128 → 64 → 12 (BayesianLinear layers) |
| Training | Bayes by Backprop, KL annealing, spatial cross-validation |
| Inference | 50 Monte Carlo forward passes → mean + epistemic uncertainty |
| Key Benefit | Quantifies prediction uncertainty per parameter |

### Model B: Hybrid CNN + Tabular

| Aspect | Detail |
|--------|--------|
| CNN Branch | Conv(15→32→48→64) on 64×64 Sentinel-2 patches |
| Tabular Branch | Station lat/lon, month, distance from source |
| Fusion | Concatenated → 96 → 64 → 12 with Huber loss |
| Key Benefit | Leverages spatial texture from raw imagery |

### Run It

```bash
python src/training/train_bnn.py --epochs 300                 # Train BNN
python src/training/train_cnn_wq.py --epochs 200 --augment    # Train CNN+Tabular
```

### Results

Model outputs (predictions, figures, metrics) are saved in:
- `GangaIndices_All/results/` — BNN outputs
- `GangaIndices_All/results_cnn/` — CNN outputs

---

## 📂 Repository Structure

```
.
├── run_full_pipeline.py              # ★ Phase 1: Main data integration pipeline
├── integration_audit.py              # Phase 1: Validation & audit figures
├── validate_phase1.py                # Phase 1: Quick sanity checks
│
├── src/                              # ★ Core source code
│   ├── data_collection/              #   GEE extraction scripts
│   │   ├── download_all_indices.py   #     Extract spectral indices
│   │   ├── download_patches.py       #     Download 64×64 image crops
│   │   └── export_indices_from_csv.py
│   ├── preprocessing/                #   Data preparation
│   │   ├── add_station_coords.py     #     Geocode stations
│   │   ├── map_station_coordinates.py
│   │   └── merge_hourly_data.py      #     Temporal aggregation
│   ├── training/                     #   Model training
│   │   ├── train_bnn.py              #     Bayesian Neural Network
│   │   ├── train_cnn_wq.py           #     Hybrid CNN + Tabular
│   │   ├── train_generic_wq.py       #     Generic WQ model
│   │   └── bayesian_wq_model.py      #     BNN model definition
│   ├── inference/                    #   Prediction on new data
│   │   └── predict_new_river.py      #     Generalize to other rivers
│   ├── analysis/                     #   Visualization & stats
│   │   ├── analyze_band_stats.py
│   │   ├── analyze_exports.py
│   │   └── visualize_exports.py
│   ├── utils/                        #   Utility helpers
│   └── build_training_dataset.py     #   Dataset builder
│
├── scripts/                          # Utility & generation scripts
│   ├── generate_ieee_paper.py        #   Auto-generate paper draft
│   ├── generate_presentation.py      #   Generate PPTX slides
│   ├── generate_research_paper.py    #   Research paper builder
│   ├── dashboard.py                  #   Interactive dashboard
│   ├── check_progress.py             #   Progress tracker
│   ├── extract_nwmp_pdfs.py          #   NWMP PDF table extraction
│   ├── extract_narmada_data.py       #   Narmada river data
│   ├── extract_tapi_data.py          #   Tapi river data
│   ├── download_nwdp_cpcb.py         #   NWDP download helper
│   ├── assemble_and_integrate.py     #   Assembly helper
│   └── parse_nwmp_html.py            #   HTML parser
│
├── water_quality_agent/              # Data download automation
│   ├── download_data.py              #   CLI: fetch from data.gov.in
│   ├── main.py                       #   Agent entry point
│   ├── downloaders/                  #   Source-specific scrapers
│   ├── pipeline/                     #   Processing pipelines
│   └── tests/                        #   Unit tests
│
├── cpcb-data-scraper/                # CPCB RTWQMS data scraper
│   ├── cpcb_scraper.py              #   Real-time station scraper
│   └── rtwqms_backfill.py           #   Historical backfill
│
├── eda/                              # Exploratory Data Analysis
│   ├── 01_dataset_eda.py            #   Dataset statistics & plots
│   ├── 02_rtwqms_station_map.py     #   Interactive station map
│   └── figures/                     #   Generated EDA plots
│
├── data/                             # Data directory
│   ├── raw/                         #   Raw source files
│   ├── final_v2/                    #   ★ Pipeline output (final dataset)
│   └── *.csv                        #   Intermediate processed files
│
├── GangaIndices_All/                 # ★ ML-ready satellite + WQ data
│   ├── merged_bn_dataset.csv        #   Training data (3,557 × 35)
│   ├── results/                     #   BNN model outputs & figures
│   └── results_cnn/                 #   CNN model outputs & figures
│
├── training_dataset/                 # Training dataset builder output
├── integration_validation/           # Audit figures for Phase 1
├── paper_figures/                    # Publication-ready figures
├── docs/                             # Architecture diagrams (.drawio)
├── compat/                           # Windows GEE compatibility stubs
├── requirements.txt                  # Python dependencies
└── LICENSE                           # MIT License
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [Google Earth Engine account](https://earthengine.google.com/signup/) (for Phase 2)
- CUDA GPU recommended for Phase 3 (optional — CPU works too)

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/ganga-water-quality.git
cd ganga-water-quality

# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# For PyTorch with CUDA (optional):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Reproduce the Full Pipeline

```bash
# Step 1: Data Integration (Phase 1)
python run_full_pipeline.py

# Step 2: Validate integration quality
python validate_phase1.py
python integration_audit.py

# Step 3: Extract satellite features (Phase 2) — requires GEE auth
earthengine authenticate
python src/data_collection/download_all_indices.py
python src/data_collection/download_patches.py

# Step 4: Train models (Phase 3)
python src/training/train_bnn.py --epochs 300
python src/training/train_cnn_wq.py --epochs 200 --augment

# Step 5: Run inference on new locations
python src/inference/predict_new_river.py
```

---

## 📊 Key Results

### Phase 1 — Data Integration

- **696 stations** across 16 states of the Ganga basin
- **175,270 records** spanning 1963–2025
- **24 water quality parameters** unified from 3 sources

### Phase 2 — Satellite Extraction

- **3,557 matched station-day** samples with Sentinel-2 data
- **18 spectral features** (10 bands + 8 indices)
- **3,690 image patches** (64×64 px) for CNN training

### Phase 3 — Model Performance

See `GangaIndices_All/results/` and `GangaIndices_All/results_cnn/` for:
- Actual vs. Predicted scatter plots
- R² scores per parameter
- Uncertainty bands (BNN)
- Residual analysis

---

## 📖 Citation

If you use this work, please cite:

```bibtex
@mastersthesis{ganga_wq_2026,
  title   = {Water Quality Prediction of Ganga River using Sentinel-2 Satellite Imagery and Deep Learning},
  author  = {<Your Name>},
  school  = {Sardar Vallabhbhai National Institute of Technology, Surat},
  year    = {2026},
  type    = {M.Tech Dissertation}
}
```

---

## 📝 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- **CPCB** — Real-Time Water Quality Monitoring System (RTWQMS)
- **data.gov.in** — Open Government Data Platform
- **NWMP** — National Water Monitoring Programme reports
- **Google Earth Engine** — Sentinel-2 satellite imagery access
- **SVNIT, Surat** — Institutional support
