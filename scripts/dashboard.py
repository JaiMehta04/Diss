"""
Ganga River Water Quality Dashboard
====================================
Interactive Streamlit dashboard for exploring multi-source integrated
water quality data across the Ganga river basin.

Features:
  - Overview KPIs & source breakdown
  - Interactive station map (Folium)
  - Parameter time-series explorer
  - Cross-source validation summary
  - SQL-like query interface
  - Data download

Launch:
    streamlit run dashboard.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Configuration ─────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "water_quality_agent" / "multi_source_data" / "integrated"
# Pick the newest integrated CSV available
_candidates = [DATA_DIR / "ganga_integrated_v2.csv",
               DATA_DIR / "ganga_integrated_tmp.csv",
               DATA_DIR / "ganga_integrated.csv"]
INTEGRATED_CSV = next((c for c in _candidates if c.exists()), _candidates[-1])
VALIDATION_JSON = DATA_DIR / "validation_report.json"
SUMMARY_CSV = DATA_DIR / "source_summary.csv"

# NWMP PDF data (may or may not exist yet)
NWMP_PDF_CSV = ROOT / "data" / "web_extracted" / "nwmp" / "nwmp_pdf_ganga_all_years.csv"

PARAMS = [
    "Dissolved_Oxygen_mg_l", "BOD_mg_l", "COD_mg_l", "pH",
    "Conductivity_uS_cm", "Temperature_C", "Turbidity_NTU",
    "Nitrate_mg_l", "Chloride_mg_l", "TOC_mg_l",
    "Fecal_Coliform_MPN_100ml", "Total_Coliform_MPN_100ml",
    "TDS_mg_l", "Ammonia_mg_l",
]

PARAM_LABELS = {
    "Dissolved_Oxygen_mg_l": "Dissolved Oxygen (mg/L)",
    "BOD_mg_l": "BOD (mg/L)",
    "COD_mg_l": "COD (mg/L)",
    "pH": "pH",
    "Conductivity_uS_cm": "Conductivity (µS/cm)",
    "Temperature_C": "Temperature (°C)",
    "Turbidity_NTU": "Turbidity (NTU)",
    "Nitrate_mg_l": "Nitrate (mg/L)",
    "Chloride_mg_l": "Chloride (mg/L)",
    "TOC_mg_l": "TOC (mg/L)",
    "Fecal_Coliform_MPN_100ml": "Fecal Coliform (MPN/100ml)",
    "Total_Coliform_MPN_100ml": "Total Coliform (MPN/100ml)",
    "TDS_mg_l": "TDS (mg/L)",
    "Ammonia_mg_l": "Ammonia (mg/L)",
}

# CPCB Water Quality Standards (Class B — bathing)
WQ_STANDARDS = {
    "Dissolved_Oxygen_mg_l": {"min": 3.0, "label": "DO ≥ 3 mg/L (Class C)"},
    "BOD_mg_l": {"max": 3.0, "label": "BOD ≤ 3 mg/L (Class B)"},
    "pH": {"min": 6.5, "max": 8.5, "label": "pH 6.5–8.5"},
    "Fecal_Coliform_MPN_100ml": {"max": 500, "label": "FC ≤ 500 MPN (Class B)"},
    "Total_Coliform_MPN_100ml": {"max": 5000, "label": "TC ≤ 5000 MPN (Class B)"},
}

SOURCE_COLORS = {
    "cpcb_realtime": "#1f77b4",
    "datagov": "#ff7f0e",
    "nwdp_cpcb": "#2ca02c",
    "nwmp_html": "#d62728",
    "nwmp_pdf": "#9467bd",
}

SOURCE_LABELS = {
    "cpcb_realtime": "CPCB Real-Time Sensors",
    "datagov": "data.gov.in API",
    "nwdp_cpcb": "NWDP Portal (CPCB)",
    "nwmp_html": "NWMP HTML Reports",
    "nwmp_pdf": "NWMP PDF Reports",
}

# ── Data Loading ──────────────────────────────────────────────────────────────


@st.cache_data(ttl=600)
def load_data() -> pd.DataFrame:
    """Load the integrated dataset."""
    df = pd.read_csv(INTEGRATED_CSV, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    # If NWMP PDF data exists, append it
    if NWMP_PDF_CSV.exists():
        pdf_df = pd.read_csv(NWMP_PDF_CSV, low_memory=False)
        if "date" not in pdf_df.columns:
            pdf_df["date"] = pd.to_datetime(
                pdf_df["year"].astype(str) + "-07-01", errors="coerce"
            )
        else:
            pdf_df["date"] = pd.to_datetime(pdf_df["date"], errors="coerce")
        if "source" not in pdf_df.columns:
            pdf_df["source"] = "nwmp_pdf"
        pdf_df["year"] = pd.to_numeric(pdf_df.get("year", pd.Series()), errors="coerce")
        # Align columns
        for col in df.columns:
            if col not in pdf_df.columns:
                pdf_df[col] = np.nan
        pdf_df = pdf_df[[c for c in df.columns if c in pdf_df.columns]]
        df = pd.concat([df, pdf_df], ignore_index=True)

    for p in PARAMS:
        if p in df.columns:
            df[p] = pd.to_numeric(df[p], errors="coerce")
    return df


@st.cache_data(ttl=600)
def load_validation():
    """Load validation report."""
    if VALIDATION_JSON.exists():
        with open(VALIDATION_JSON) as f:
            return json.load(f)
    return {}


# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Ganga Water Quality Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("🌊 Ganga WQ Dashboard")

df = load_data()
validation = load_validation()

# Sidebar filters
st.sidebar.header("Filters")

sources = sorted(df["source"].dropna().unique())
selected_sources = st.sidebar.multiselect(
    "Data Sources",
    sources,
    default=sources,
    format_func=lambda x: SOURCE_LABELS.get(x, x),
)

year_min, year_max = int(df["year"].min()), int(df["year"].max())
year_range = st.sidebar.slider("Year Range", year_min, year_max, (year_min, year_max))

stations = sorted(df["station_name"].dropna().astype(str).unique())
selected_stations = st.sidebar.multiselect(
    "Stations (leave empty for all)",
    stations,
    default=[],
)

# Apply filters
mask = (
    df["source"].isin(selected_sources)
    & df["year"].between(year_range[0], year_range[1])
)
if selected_stations:
    mask &= df["station_name"].astype(str).isin(selected_stations)
fdf = df[mask].copy()

st.sidebar.markdown("---")
st.sidebar.metric("Filtered Rows", f"{len(fdf):,}")
st.sidebar.metric("Total Rows", f"{len(df):,}")

# ── Page Navigation ──────────────────────────────────────────────────────────

page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview", "🗺️ Station Map", "📈 Parameter Explorer",
     "✅ Validation Summary", "🔍 Query Data"],
)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

if page == "📊 Overview":
    st.title("Ganga River Water Quality — Multi-Source Overview")

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Records", f"{len(fdf):,}")
    c2.metric("Stations", f"{fdf['station_name'].nunique():,}")
    c3.metric("Sources", f"{fdf['source'].nunique()}")
    c4.metric("Year Range", f"{int(fdf['year'].min())}–{int(fdf['year'].max())}")
    params_with_data = sum(1 for p in PARAMS if p in fdf.columns and fdf[p].notna().any())
    c5.metric("Parameters", f"{params_with_data}")

    st.markdown("---")

    # Source breakdown
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Records by Source")
        src_counts = fdf["source"].value_counts().reset_index()
        src_counts.columns = ["Source", "Count"]
        src_counts["Label"] = src_counts["Source"].map(SOURCE_LABELS).fillna(src_counts["Source"])
        fig = px.pie(
            src_counts, values="Count", names="Label",
            color="Source",
            color_discrete_map=SOURCE_COLORS,
            hole=0.4,
        )
        fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Records by Year")
        year_counts = fdf.groupby(["year", "source"]).size().reset_index(name="count")
        fig = px.bar(
            year_counts, x="year", y="count", color="source",
            color_discrete_map=SOURCE_COLORS,
            labels={"year": "Year", "count": "Records", "source": "Source"},
        )
        fig.update_layout(margin=dict(t=20, b=20), height=350, barmode="stack")
        st.plotly_chart(fig, use_container_width=True)

    # Parameter coverage heatmap
    st.subheader("Parameter Coverage by Source")
    coverage_data = []
    for src in fdf["source"].unique():
        src_df = fdf[fdf["source"] == src]
        for p in PARAMS:
            if p in src_df.columns:
                pct = src_df[p].notna().mean() * 100
                coverage_data.append({
                    "Source": SOURCE_LABELS.get(src, src),
                    "Parameter": PARAM_LABELS.get(p, p),
                    "Coverage (%)": round(pct, 1),
                })
    if coverage_data:
        cov_df = pd.DataFrame(coverage_data)
        pivot = cov_df.pivot(index="Source", columns="Parameter", values="Coverage (%)")
        fig = px.imshow(
            pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            color_continuous_scale="YlGn",
            aspect="auto",
            text_auto=".0f",
            labels={"color": "Coverage %"},
        )
        fig.update_layout(height=300, margin=dict(t=20, b=20))
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    # Key statistics table
    st.subheader("Parameter Statistics")
    stats_rows = []
    for p in PARAMS:
        if p not in fdf.columns:
            continue
        vals = fdf[p].dropna()
        if len(vals) == 0:
            continue
        stats_rows.append({
            "Parameter": PARAM_LABELS.get(p, p),
            "Count": len(vals),
            "Mean": round(vals.mean(), 2),
            "Median": round(vals.median(), 2),
            "Std": round(vals.std(), 2),
            "Min": round(vals.min(), 2),
            "Max": round(vals.max(), 2),
        })
    if stats_rows:
        st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2: STATION MAP
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🗺️ Station Map":
    st.title("Ganga Basin — Monitoring Stations")

    # Prepare station data with coordinates
    geo = fdf.dropna(subset=["latitude", "longitude"])
    geo = geo[(geo["latitude"].between(20, 35)) & (geo["longitude"].between(73, 92))]

    if len(geo) == 0:
        st.warning("No stations with valid coordinates in the current filter.")
    else:
        # Aggregate per station
        station_agg = (
            geo.groupby(["station_name", "latitude", "longitude", "source"])
            .agg(
                records=("date", "size"),
                first_date=("date", "min"),
                last_date=("date", "max"),
                mean_DO=("Dissolved_Oxygen_mg_l", "mean"),
                mean_BOD=("BOD_mg_l", "mean"),
                mean_pH=("pH", "mean"),
            )
            .reset_index()
        )
        station_agg["mean_DO"] = station_agg["mean_DO"].round(2)
        station_agg["mean_BOD"] = station_agg["mean_BOD"].round(2)
        station_agg["mean_pH"] = station_agg["mean_pH"].round(2)

        # Color by parameter
        map_param = st.selectbox(
            "Color stations by",
            ["mean_DO", "mean_BOD", "mean_pH", "records"],
            format_func=lambda x: {
                "mean_DO": "Dissolved Oxygen (mg/L)",
                "mean_BOD": "BOD (mg/L)",
                "mean_pH": "pH",
                "records": "Number of Records",
            }.get(x, x),
        )

        col_map, col_rev = "RdYlGn", False
        if map_param == "mean_BOD":
            col_rev = True  # Higher BOD = worse

        # Plotly mapbox scatter
        fig = px.scatter_mapbox(
            station_agg,
            lat="latitude",
            lon="longitude",
            color=map_param,
            size="records",
            size_max=18,
            hover_name="station_name",
            hover_data={
                "source": True,
                "records": True,
                "mean_DO": ":.2f",
                "mean_BOD": ":.2f",
                "mean_pH": ":.2f",
                "latitude": ":.4f",
                "longitude": ":.4f",
            },
            color_continuous_scale=col_map,
            mapbox_style="open-street-map",
            zoom=5,
            center={"lat": 26.8, "lon": 81.0},
            height=650,
        )
        if col_rev:
            fig.update_coloraxes(reversescale=True)
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)

        # Station table
        with st.expander("📋 Station Details Table"):
            display_cols = ["station_name", "latitude", "longitude", "source",
                            "records", "first_date", "last_date",
                            "mean_DO", "mean_BOD", "mean_pH"]
            st.dataframe(
                station_agg[display_cols].sort_values("records", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

        st.caption(
            f"Showing {station_agg['station_name'].nunique()} stations "
            f"with coordinates ({len(geo):,} records)"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3: PARAMETER EXPLORER
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📈 Parameter Explorer":
    st.title("Parameter Explorer")

    available_params = [p for p in PARAMS if p in fdf.columns and fdf[p].notna().any()]
    if not available_params:
        st.warning("No parameter data available for the current filter.")
    else:
        selected_param = st.selectbox(
            "Select Parameter",
            available_params,
            format_func=lambda x: PARAM_LABELS.get(x, x),
        )

        tab1, tab2, tab3 = st.tabs(["Time Series", "Distribution", "Source Comparison"])

        with tab1:
            st.subheader(f"{PARAM_LABELS.get(selected_param, selected_param)} — Time Series")

            # Aggregate by month per source
            ts = fdf[["date", "source", selected_param]].dropna(subset=[selected_param])
            if len(ts) > 0:
                ts["month"] = ts["date"].dt.to_period("M").astype(str)
                monthly = ts.groupby(["month", "source"])[selected_param].mean().reset_index()

                fig = px.line(
                    monthly, x="month", y=selected_param, color="source",
                    color_discrete_map=SOURCE_COLORS,
                    labels={
                        selected_param: PARAM_LABELS.get(selected_param, selected_param),
                        "month": "Month",
                    },
                )

                # Add standard threshold lines
                if selected_param in WQ_STANDARDS:
                    std = WQ_STANDARDS[selected_param]
                    if "max" in std:
                        fig.add_hline(
                            y=std["max"], line_dash="dash", line_color="red",
                            annotation_text=std["label"],
                        )
                    if "min" in std:
                        fig.add_hline(
                            y=std["min"], line_dash="dash", line_color="red",
                            annotation_text=std["label"],
                        )

                fig.update_layout(height=450, margin=dict(t=30, b=30))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No time-series data available.")

        with tab2:
            st.subheader(f"{PARAM_LABELS.get(selected_param, selected_param)} — Distribution")
            dist = fdf[["source", selected_param]].dropna(subset=[selected_param])
            if len(dist) > 0:
                fig = px.histogram(
                    dist, x=selected_param, color="source",
                    color_discrete_map=SOURCE_COLORS,
                    nbins=50, barmode="overlay", opacity=0.7,
                    labels={selected_param: PARAM_LABELS.get(selected_param, selected_param)},
                )
                if selected_param in WQ_STANDARDS:
                    std = WQ_STANDARDS[selected_param]
                    if "max" in std:
                        fig.add_vline(x=std["max"], line_dash="dash", line_color="red")
                    if "min" in std:
                        fig.add_vline(x=std["min"], line_dash="dash", line_color="red")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.subheader(f"{PARAM_LABELS.get(selected_param, selected_param)} — Source Comparison")
            box_data = fdf[["source", selected_param]].dropna(subset=[selected_param])
            if len(box_data) > 0:
                fig = px.box(
                    box_data, x="source", y=selected_param, color="source",
                    color_discrete_map=SOURCE_COLORS,
                    labels={
                        selected_param: PARAM_LABELS.get(selected_param, selected_param),
                        "source": "Source",
                    },
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

                # Table of stats per source
                src_stats = box_data.groupby("source")[selected_param].describe().round(2)
                st.dataframe(src_stats, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4: VALIDATION SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

elif page == "✅ Validation Summary":
    st.title("Cross-Source Validation Summary")

    if not validation:
        st.warning("No validation report found.")
    else:
        # Overview
        st.subheader("Dataset Overview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Rows", f"{validation.get('total_rows', 0):,}")
        c2.metric("Total Stations", f"{validation.get('total_stations', 0):,}")

        sources_info = validation.get("sources", {})
        n_src = len(sources_info)
        c3.metric("Data Sources", f"{n_src}")

        # Source details table
        st.subheader("Source Details")
        src_table = []
        for src_name, info in sources_info.items():
            src_table.append({
                "Source": SOURCE_LABELS.get(src_name, src_name),
                "Rows": info.get("rows", 0),
                "Stations": info.get("stations", 0),
                "Date Range": f"{info.get('date_range', ['?', '?'])[0][:10]} → {info.get('date_range', ['?', '?'])[1][:10]}",
                "Years": f"{info.get('year_range', ['?', '?'])[0]}–{info.get('year_range', ['?', '?'])[1]}",
            })
        st.dataframe(pd.DataFrame(src_table), use_container_width=True, hide_index=True)

        # Cross-source comparison
        st.subheader("Cross-Source Parameter Comparison")
        comparisons = validation.get("cross_source_comparison", {})
        if comparisons:
            comp_rows = []
            for key, comp in comparisons.items():
                comp_rows.append({
                    "Parameter": PARAM_LABELS.get(comp.get("param", ""), comp.get("param", "")),
                    "Source 1": SOURCE_LABELS.get(comp.get("source_1", ""), comp.get("source_1", "")),
                    "Mean 1": comp.get("mean_1"),
                    "Source 2": SOURCE_LABELS.get(comp.get("source_2", ""), comp.get("source_2", "")),
                    "Mean 2": comp.get("mean_2"),
                    "Diff %": comp.get("pct_diff"),
                })
            comp_df = pd.DataFrame(comp_rows)
            comp_df["Diff %"] = pd.to_numeric(comp_df["Diff %"], errors="coerce").round(1)
            comp_df["Mean 1"] = pd.to_numeric(comp_df["Mean 1"], errors="coerce").round(2)
            comp_df["Mean 2"] = pd.to_numeric(comp_df["Mean 2"], errors="coerce").round(2)

            # Color code by difference
            st.dataframe(
                comp_df.style.background_gradient(
                    subset=["Diff %"], cmap="RdYlGn_r", vmin=0, vmax=50
                ),
                use_container_width=True,
                hide_index=True,
            )

            # Visualization
            st.subheader("Mean Parameter Values by Source")
            # Build a pivot of mean values per source per param
            mean_data = []
            for src_name in sources_info:
                src_df = fdf[fdf["source"] == src_name]
                for p in PARAMS:
                    if p in src_df.columns:
                        m = src_df[p].mean()
                        if pd.notna(m):
                            mean_data.append({
                                "Source": SOURCE_LABELS.get(src_name, src_name),
                                "Parameter": PARAM_LABELS.get(p, p),
                                "Mean": round(m, 2),
                            })
            if mean_data:
                mean_df = pd.DataFrame(mean_data)
                fig = px.bar(
                    mean_df, x="Parameter", y="Mean", color="Source",
                    barmode="group",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig.update_layout(height=450, margin=dict(t=30))
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)

        # QA/QC summary
        st.subheader("QA/QC Summary")
        qa_info = validation.get("qa_qc", {})
        if qa_info:
            col1, col2, col3 = st.columns(3)
            col1.metric("Bounds Violations", f"{qa_info.get('bounds_violations', 0):,}")
            col2.metric("BOD > 2×COD Flags", f"{qa_info.get('bod_gt_2cod', 0):,}")
            col3.metric("Duplicates Removed", f"{qa_info.get('duplicates_removed', 0):,}")

        # Water quality compliance
        st.subheader("CPCB Standards Compliance")
        compliance_rows = []
        for p, std in WQ_STANDARDS.items():
            if p not in fdf.columns:
                continue
            vals = fdf[p].dropna()
            if len(vals) == 0:
                continue
            if "max" in std:
                violations = (vals > std["max"]).sum()
            elif "min" in std:
                violations = (vals < std["min"]).sum()
            else:
                violations = ((vals < std.get("min", -999)) | (vals > std.get("max", 999999))).sum()
            compliance_pct = (1 - violations / len(vals)) * 100
            compliance_rows.append({
                "Standard": std["label"],
                "Total Samples": len(vals),
                "Violations": violations,
                "Compliance %": round(compliance_pct, 1),
            })
        if compliance_rows:
            comp_df = pd.DataFrame(compliance_rows)
            st.dataframe(
                comp_df.style.background_gradient(
                    subset=["Compliance %"], cmap="RdYlGn", vmin=50, vmax=100
                ),
                use_container_width=True,
                hide_index=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 5: QUERY DATA
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Query Data":
    st.title("Query & Explore Data")

    st.markdown("""
    Use the filters below to query the integrated dataset. You can also write
    **pandas query expressions** for advanced filtering.
    """)

    # Quick filters
    col1, col2, col3 = st.columns(3)

    with col1:
        q_param = st.selectbox(
            "Parameter",
            ["All"] + [p for p in PARAMS if p in fdf.columns and fdf[p].notna().any()],
            format_func=lambda x: PARAM_LABELS.get(x, x) if x != "All" else "All Parameters",
        )

    with col2:
        q_source = st.selectbox(
            "Source",
            ["All"] + list(fdf["source"].unique()),
            format_func=lambda x: SOURCE_LABELS.get(x, x) if x != "All" else "All Sources",
        )

    with col3:
        q_limit = st.number_input("Max rows to display", 10, 10000, 500, step=100)

    # Advanced query
    st.subheader("Advanced Query (Pandas Expression)")
    query_expr = st.text_input(
        "Enter a pandas query expression",
        placeholder='e.g., pH > 8.5 & source == "cpcb_realtime"',
        help="Use column names like pH, BOD_mg_l, source, year, station_name. "
             "Operators: >, <, ==, !=, &, |",
    )

    # Apply query
    result = fdf.copy()

    if q_param != "All":
        result = result.dropna(subset=[q_param])
    if q_source != "All":
        result = result[result["source"] == q_source]

    if query_expr:
        try:
            result = result.query(query_expr)
            st.success(f"Query returned {len(result):,} rows")
        except Exception as e:
            st.error(f"Query error: {e}")

    # Display
    st.subheader(f"Results ({len(result):,} rows)")

    # Select columns to display
    display_cols = ["station_name", "date", "source", "year"]
    if q_param != "All":
        display_cols.append(q_param)
    else:
        display_cols.extend([p for p in PARAMS if p in result.columns and result[p].notna().any()])

    st.dataframe(
        result[display_cols].head(q_limit),
        use_container_width=True,
        hide_index=True,
    )

    # Download button
    csv_data = result.to_csv(index=False)
    st.download_button(
        "📥 Download Filtered Data (CSV)",
        csv_data,
        file_name="ganga_wq_query_result.csv",
        mime="text/csv",
    )

    # Quick statistics for query result
    if q_param != "All" and q_param in result.columns:
        st.subheader(f"Statistics for {PARAM_LABELS.get(q_param, q_param)}")
        col1, col2, col3, col4 = st.columns(4)
        vals = result[q_param].dropna()
        col1.metric("Mean", f"{vals.mean():.2f}")
        col2.metric("Median", f"{vals.median():.2f}")
        col3.metric("Min", f"{vals.min():.2f}")
        col4.metric("Max", f"{vals.max():.2f}")


# ── Footer ────────────────────────────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.caption(
    "Ganga Water Quality Dashboard\n"
    "Multi-source integrated dataset\n"
    f"Data: {len(df):,} records, {df['source'].nunique()} sources"
)
