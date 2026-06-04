# Multi-Source Data Directory

This folder consolidates **all** water quality data sources used in the
Ganga River integration pipeline.

## Sources

| # | Source | Period | Records | Parameters | Origin |
|---|--------|--------|---------|------------|--------|
| 1 | CPCB Real-Time (local) | Jul–Nov 2025 | 75,801 | 12 | CPCB RTWQMS sensors |
| 2 | NWDP-CPCB Biological | 1961–2025 | 16,932 | BOD, COD, Fecal/Total Coliform | nwdp.nwic.gov.in |
| 3 | NWDP-CPCB Chemical | 1961–2025 | 10,024 | DO, pH, Chloride, Nitrate, metals | nwdp.nwic.gov.in |
| 4 | NWDP-CPCB Physical | 1961–2025 | 14,013 | Conductivity, Turbidity, Temp | nwdp.nwic.gov.in |
| 5 | NWMP HTML (CPCB ENVIS) | 2012–2014 | 155 | 8 (annual Min/Max/Mean) | cpcb.nic.in |
| 6 | data.gov.in | 2011–2022 | ~20,000 | varies | api.data.gov.in |

## File Layout

```
multi_source_data/
├── README.md
├── 01_cpcb_realtime/          ← CPCB real-time 2025 sensor data
│   ├── station_data.csv
│   └── station_locations.csv
├── 02_nwdp_cpcb/              ← NWDP portal CPCB downloads
│   ├── biological_ganga.csv
│   ├── chemical_ganga.csv
│   └── physical_ganga.csv
├── 03_nwmp_html/              ← NWMP annual reports (parsed HTML)
│   └── nwmp_ganga_all_years.csv
├── 04_datagov/                ← data.gov.in API downloads
│   ├── ganga_wq_2011_2015.csv
│   ├── ganga_wq_median_2017_2022.csv
│   ├── ganga_wq_stationwise_2018_2020.csv
│   ├── ganga_wq_stationwise_2021.csv
│   ├── ganga_wq_2012.csv
│   └── surface_wq_march_2018_cpcb.csv
└── integrated/                ← Pipeline output
    ├── ganga_integrated.csv
    └── validation_report.json
```
