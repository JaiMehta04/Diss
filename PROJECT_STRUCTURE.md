# Project Structure

## Repository: JaiMehta04/Diss

Ganga River Water Quality Monitoring – Dissertation Project

---

## Tracked Files (Pushed to GitHub)

```
.gitignore
info.txt
integration_audit.py
LICENSE
README.md
requirements.txt
run_full_pipeline.py
validate_phase1.py

compat/
├── fcntl.py
├── StringIO.py
└── termios.py

cpcb-data-scraper/
├── .github/workflows/
│   ├── backfill.yml
│   └── scraper.yml
├── cpcb_scraper.py
├── requirements.txt
└── rtwqms_backfill.py

data/
├── README.md
├── consolidated/
│   └── consolidated_ganga_downloads.csv
├── cpcb_station_locations.csv
├── final/
│   ├── ganga_water_quality_final.csv
│   ├── ganga_water_quality_long.csv
│   └── pipeline_report.json
├── final_v2/
│   ├── ganga_water_quality_final.csv
│   ├── ganga_water_quality_long.csv
│   └── pipeline_report.json
└── raw/
    ├── datagov/
    │   ├── ganga_river_ganga_water_quality_20112015.csv
    │   ├── ganga_statewise_river_ganga_wq_median_20172022.csv
    │   ├── ganga_stationwise_river_ganga_wq_20182020.csv
    │   ├── ganga_stationwise_river_ganga_wq_2021.csv
    │   ├── ganga_water_quality_of_river_ganga__2012.csv
    │   └── surface_surface_water_quality_march_2018_cpcb.csv
    └── download_report.json

docs/
├── Phase2.drawio
└── proposed_framework.drawio

eda/
├── 01_dataset_eda.py
├── 02_rtwqms_station_map.py
├── check_missing_stations.py
├── missing_sentinel_audit.csv
├── outlier_analysis.csv
├── region_month_counts.csv
├── region_season_counts.csv
├── region_summary.csv
├── rtwqms_station_river_mapping.csv
├── station_detail.csv
├── station_season_counts.csv
├── station_trimming_survival.csv
└── figures/
    ├── 01_region_season_distribution.png
    ├── 02_monthly_distribution_by_region.png
    ├── 03_target_boxplots_by_region.png
    ├── 04_missing_data_heatmap.png
    ├── 05_temporal_coverage.png
    ├── 06_correlation_heatmaps_by_region.png
    ├── 07_seasonal_wq_trends.png
    ├── 08_station_map.png
    ├── 09_trimming_impact.png
    ├── 10_station_survival.png
    ├── rtwqms_all_stations_map.png
    └── rtwqms_station_map.html

GangaIndices_All/
├── results/
│   ├── figures/
│   │   ├── 01_satellite_distributions.png
│   │   ├── 02_wq_distributions.png
│   │   ├── 03_sat_wq_correlation_heatmap.png
│   │   ├── 04_sat_wq_spearman_heatmap.png
│   │   ├── 05_feature_intercorrelation.png
│   │   ├── 06_spatial_variation.png
│   │   ├── 07_seasonal_variation.png
│   │   ├── 08_monthly_index_trends.png
│   │   ├── 16_bnn_actual_vs_predicted.png
│   │   ├── 17_bnn_uncertainty_bands.png
│   │   ├── 18_bnn_r2_uncertainty.png
│   │   └── 19_bnn_residuals.png
│   └── tables/
│       ├── bnn_multioutput.tex
│       ├── bnn_multioutput_results.csv
│       ├── sat_wq_pearson_correlation.csv
│       ├── sat_wq_spearman_correlation.csv
│       └── statistical_summary.csv
└── results_cnn/
    ├── figures/
    │   ├── cnn_actual_vs_predicted.png
    │   ├── cnn_r2_bars.png
    │   └── cnn_residuals.png
    └── tables/
        ├── cnn_results.csv
        └── cnn_results.tex

integration_validation/
├── fig1_source_quality.png
├── fig2_temporal_coverage.png
├── fig3_parameter_completeness.png
├── fig4_station_map.png
├── fig5_pipeline_stages.png
├── fig6_parameter_distributions.png
├── ieee_section_iv_text.py
├── integration_audit_report.json
└── integration_summary_table.csv

paper_figures/
├── fig1_source_distribution.png
├── fig2_parameter_coverage.png
├── fig3_temporal_coverage.png
├── fig4_quality_flags.png
├── fig5_state_distribution.png
└── fig6_trust_distribution.png

scripts/
├── README.md
├── assemble_and_integrate.py
├── check_progress.py
├── dashboard.py
├── download_nwdp_cpcb.py
├── extract_narmada_data.py
├── extract_nwmp_pdfs.py
├── extract_tapi_data.py
├── generate_ieee_paper.py
├── generate_presentation.py
├── generate_research_paper.py
└── parse_nwmp_html.py

src/
├── analysis/
│   ├── analyze_band_stats.py
│   ├── analyze_exports.py
│   └── visualize_exports.py
├── build_training_dataset.py
├── build_training_dataset_v2.py
├── data_collection/
│   ├── download_all_indices.py
│   ├── download_patches.py
│   ├── export_indices_from_csv.py
│   └── ganga_water_quality.py
├── inference/
│   └── predict_new_river.py
├── preprocessing/
│   ├── add_station_coords.py
│   ├── map_station_coordinates.py
│   └── merge_hourly_data.py
├── training/
│   ├── bayesian_wq_model.py
│   ├── train_bnn.py
│   ├── train_cnn_wq.py
│   └── train_generic_wq.py
└── utils/
    ├── generate_ppt.py
    └── sample_csv.py

water_quality_agent/
├── __init__.py
├── download_data.py
├── main.py
├── requirements.txt
├── adapters/
│   ├── __init__.py
│   ├── base.py
│   ├── cpcb.py
│   ├── cwc.py
│   ├── data_gov.py
│   └── gemstat.py
├── agents/
│   ├── __init__.py
│   ├── conflict_resolver.py
│   ├── quality_auditor.py
│   └── schema_mapper.py
├── config/
│   ├── __init__.py
│   ├── bounds.py
│   ├── parameters.py
│   ├── schema.py
│   └── units.py
├── downloaders/
│   ├── __init__.py
│   ├── cpcb_realtime.py
│   ├── cpcb_scraper.py
│   ├── datagov.py
│   ├── kaggle_dl.py
│   ├── nwmp_extractor.py
│   └── pdf_extractor.py
├── multi_source_data/
│   ├── README.md
│   ├── 01_cpcb_realtime/
│   │   ├── station_data.csv
│   │   └── station_locations.csv
│   ├── 02_nwdp_cpcb/
│   │   ├── biological_ganga.csv
│   │   ├── chemical_ganga.csv
│   │   └── physical_ganga.csv
│   ├── 03_nwmp_html/
│   │   ├── nwmp_ganga_all_years.csv
│   │   └── nwmp_pdf_ganga_all_years.csv
│   ├── 04_datagov/
│   │   ├── ganga_wq_2011_2015.csv
│   │   ├── ganga_wq_2012.csv
│   │   ├── ganga_wq_median_2017_2022.csv
│   │   ├── ganga_wq_stationwise_2018_2020.csv
│   │   ├── ganga_wq_stationwise_2021.csv
│   │   └── surface_wq_march_2018_cpcb.csv
│   └── integrated/
│       ├── ganga_integrated.csv
│       ├── ganga_integrated_v2.csv
│       ├── source_summary.csv
│       └── validation_report.json
├── pipeline/
│   ├── __init__.py
│   ├── deduplicator.py
│   ├── normalizer.py
│   ├── orchestrator.py
│   └── validator.py
└── utils/
    ├── __init__.py
    ├── date_parser.py
    ├── geo.py
    ├── llm_client.py
    └── logging_config.py
```

---

## Excluded Files & Directories (via .gitignore)

The following were excluded from the repository to keep it within GitHub's size limits.
Total workspace size: ~4.1 GB | Pushed size: ~23.5 MB

### Virtual Environments (~2 GB)
| Directory | Size | Reason |
|-----------|------|--------|
| `.venv/` | ~1,136 MB | Python virtual environment (recreate with `pip install -r requirements.txt`) |
| `ganga/` | ~903 MB | Alternate virtual environment |

### Large Data & Satellite Files (~1.8 GB)
| Directory/Pattern | Size | Reason |
|-------------------|------|--------|
| `SentinelPatches/` | ~578 MB | ~3500+ Sentinel-2 `.npy` image patches (too large for Git) |
| `training_dataset/patches/` | ~271 MB | Training image patches (binary, regenerated by pipeline) |
| `data/Hourly_data/` | — | Hourly CSV dumps (regenerated) |
| `data/Hourly_data_processed/` | — | Processed hourly data |
| `data/rtwqms_historical/` | — | Historical RTWQMS data |
| `data/web_extracted/` | — | Web-scraped intermediate data |
| `data/raw/cpcb/nwmp_pdfs/` | — | Raw PDF files from CPCB |
| `data/raw/cpcb/discovered_files/` | — | Discovered raw files |
| `data/raw/pdf_extracted/` | — | 349 CSVs extracted from PDFs (regenerated) |
| `data/combined_station_data.csv` | — | Intermediate CSV |
| `data/combined_station_data_with_coords.csv` | — | Intermediate CSV |
| `NWMP data/` | ~37 MB | Raw HTML data files |
| `GangaIndices/` | ~10 MB | GeoTIFF exports from Google Earth Engine |

### Model Weights
| Pattern | Reason |
|---------|--------|
| `*.pt`, `*.pth` | PyTorch model checkpoints |
| `*.onnx` | ONNX model exports |
| `*.h5` | Keras/TensorFlow model weights |
| `*.joblib`, `*.pkl` | Serialized sklearn models |

### Generated Outputs & Documents
| Pattern/Directory | Reason |
|-------------------|--------|
| `outputs/` | Regenerated by scripts |
| `docs/*.pptx`, `docs/*.pdf` | Binary documents |
| `docs/generated/` | Auto-generated documentation |
| `*.docx`, `*.pptx` | Office documents |
| `*.zip` | Archive files |
| `GangaIndices_All/results/models/` | Trained model files |
| `GangaIndices_All/results_cnn/models/` | CNN model files |
| `training_dataset/results*/` | Training results |
| `water_quality_agent/output/` | Agent output |

### IDE, Cache & OS Files
| Pattern | Reason |
|---------|--------|
| `.vscode/`, `.idea/` | Editor settings |
| `__pycache__/`, `*.py[cod]` | Python bytecode |
| `.ipynb_checkpoints/`, `*.ipynb` | Jupyter artifacts |
| `.DS_Store`, `Thumbs.db`, `desktop.ini` | OS metadata |
| `*.log`, `*.tmp`, `*.swp`, `*.swo` | Temporary files |

---

## How to Recreate Excluded Data

1. **Virtual environment**: `python -m venv .venv && pip install -r requirements.txt`
2. **Sentinel patches**: Run `src/data_collection/download_patches.py`
3. **Training dataset**: Run `src/build_training_dataset.py`
4. **Hourly data**: Run `scripts/download_nwdp_cpcb.py`
5. **Model weights**: Run training scripts in `src/training/`
