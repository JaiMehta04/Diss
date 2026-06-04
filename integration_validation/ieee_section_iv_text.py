"""
IEEE Paper Section: Data Integration Results & Validation
=========================================================
Paste this text into Section IV of your IEEE paper.
"""

SECTION_IV_TEXT = """
IV. DATA INTEGRATION RESULTS AND VALIDATION

A. Dataset Characteristics

The proposed multi-source integration framework successfully fused water
quality data from three heterogeneous sources into a unified dataset
comprising 175,270 quality-controlled records across 696 monitoring
stations in 16 states of the Ganga basin, spanning 1963–2025 (Table II).
The wide-format dataset contains 14,270 station-date observations with
24 water quality parameters.

TABLE II: Source Contribution Summary
┌─────────────┬──────────┬────────┬─────────┬───────────┬───────────┐
│ Source       │ Stations │ Params │ Records │ Year Span │ Trust Avg │
├─────────────┼──────────┼────────┼─────────┼───────────┼───────────┤
│ CPCB        │       37 │     10 │  36,600 │ 2025      │ 0.849     │
│ data.gov.in │      496 │     23 │ 137,938 │ 1963–2022 │ 0.642     │
│ NWMP PDFs   │      170 │      7 │     732 │ 2015–2024 │ 0.688     │
├─────────────┼──────────┼────────┼─────────┼───────────┼───────────┤
│ INTEGRATED  │      696 │     24 │ 175,270 │ 1963–2025 │ 0.685     │
└─────────────┴──────────┴────────┴─────────┴───────────┴───────────┘

B. Quality Control Effectiveness

The seven-stage QA/QC pipeline processed 1,090,673 raw records through
progressive filtering stages:

  (1) Schema standardization removed 139,777 (12.8%) malformed or
      unmappable records with missing station names or unrecognized
      parameter columns.

  (2) Spatial validation using a Ganga basin bounding box (21.0–31.5°N,
      73.0–90.0°E) rejected 125 records with coordinates clearly
      outside the study area.

  (3) Physical range checks (e.g., pH ∈ [0, 14], DO ∈ [0, 25 mg/L])
      removed 5,792 physically impossible measurements.

  (4) Statistical outlier detection using IQR-based methods flagged
      34,926 records (3.7%) as suspect_outlier without removal.

  (5) Cross-parameter consistency checks (BOD ≤ 2×COD, TDS/EC ratio
      ∈ [0.2, 1.5]) flagged 39,856 records (4.2%) as
      suspect_consistency.

  (6) Trust-weighted deduplication merged 769,709 overlapping records
      into 175,270 unique observations, resolving 81,117 conflicts
      using source-reliability-weighted averaging.

The final dataset achieves an 80.8% validity rate, with remaining
flagged records retained but marked for downstream sensitivity analysis.

C. Data Plausibility Validation

Parameter distributions in the integrated dataset were validated against
known physical constraints for Ganga River water quality:

  • pH: 99.1% within [5.5, 10.0], median = 8.0 (consistent with
    slightly alkaline Gangetic waters)
  • Dissolved Oxygen: 99.9% within [0, 20 mg/L], median = 7.1 mg/L
  • BOD: 99.9% within [0, 100 mg/L], median = 0.5 mg/L
  • BOD ≤ COD consistency: 99.8% (6,247/6,259 paired observations)
  • Temperature: 100% within [0, 45°C], median = 26.5°C

These distributions align with published studies on Ganga water quality
[28]–[30], confirming the plausibility of integrated measurements.

D. Spatial and Temporal Coverage

The integrated dataset provides 93.6% spatial coverage (13,358/14,270
observations with valid coordinates across 278 georeferenced stations).
Coordinates span from Gangotri (30.95°N, 78.94°E) to the Bay of Bengal
(22.61°N, 88.75°E), covering the full 2,525 km course of the Ganga.

Temporal density peaks in 2021 and 2025 due to CPCB's expanded real-time
monitoring network. Historical coverage from data.gov.in extends to 1963,
enabling long-term trend analysis.

E. ML-Readiness Assessment

For the downstream satellite-based prediction model (requiring post-2015
data with coordinates for Sentinel-2 matching), the integration produces:

  • 5,737 station-date pairs with coordinates in the Sentinel-2 era
  • 268 unique stations available for satellite matching
  • Key parameter availability: pH (98.9%), DO (94.9%), BOD (94.2%),
    COD (89.9%), Temperature (97.8%), Conductivity (97.5%)
  • An existing matched dataset of 3,557 samples with 10 Sentinel-2
    bands + 8 spectral indices already extracted via Google Earth Engine

F. Limitations and Future Work

  (1) Multi-source overlap is limited to 7 stations, constraining
      inter-source cross-validation opportunities.
  (2) CPCB real-time data is concentrated in 2025, creating temporal
      asymmetry with the historical data.gov.in records.
  (3) 424 stations (from NWMP/data.gov.in) lack precise coordinates,
      limiting their use in satellite-based models.
  (4) Sparse parameters (arsenic: 1.6%, TSS: 0.8%) may require
      imputation or exclusion in downstream modeling.

Despite these limitations, the framework achieves its primary objective:
creating a reproducible, quality-controlled, multi-source dataset that
substantially expands the training corpus available for satellite-based
water quality prediction along the Ganga River.
"""

if __name__ == "__main__":
    print(SECTION_IV_TEXT)
