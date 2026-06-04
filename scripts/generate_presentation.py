"""
Generate a presentation (.pptx) from the IEEE research paper content
on the Ganga River water quality data integration framework.
"""

import json
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt, Emu

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
FINAL_DIR = BASE / "data" / "final"
WIDE_CSV = FINAL_DIR / "ganga_water_quality_final.csv"
LONG_CSV = FINAL_DIR / "ganga_water_quality_long.csv"
REPORT_JSON = FINAL_DIR / "pipeline_report.json"
FIGURES_DIR = BASE / "paper_figures"
OUTPUT_PPTX = BASE / "Presentation_Ganga_WQ_Integration.pptx"

# ── Color palette ─────────────────────────────────────────────────────────────
DARK_BLUE = RGBColor(0x1F, 0x4E, 0x79)
MED_BLUE = RGBColor(0x2E, 0x75, 0xB6)
LIGHT_BLUE = RGBColor(0xD6, 0xE4, 0xF0)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BLACK = RGBColor(0x00, 0x00, 0x00)
DARK_GRAY = RGBColor(0x33, 0x33, 0x33)
ACCENT_GREEN = RGBColor(0x2C, 0xA0, 0x2C)
ACCENT_ORANGE = RGBColor(0xFF, 0x7F, 0x0E)

SLIDE_WIDTH = Inches(13.333)
SLIDE_HEIGHT = Inches(7.5)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def set_slide_bg(slide, color):
    """Set solid background color on a slide."""
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_text_box(slide, left, top, width, height, text, font_size=18,
                 bold=False, italic=False, color=BLACK, align=PP_ALIGN.LEFT,
                 font_name="Calibri", line_spacing=1.15):
    """Add a text box with a single run."""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    p.space_after = Pt(4)
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    run.font.name = font_name
    if line_spacing != 1.0:
        p.line_spacing = Pt(font_size * line_spacing)
    return txBox


def add_bullet_slide(slide, items, left, top, width, height, font_size=16,
                     color=DARK_GRAY, bullet_color=MED_BLUE):
    """Add a text frame with bullet points."""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True

    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.space_after = Pt(6)
        p.space_before = Pt(2)
        p.level = 0

        # Bullet character
        bullet_run = p.add_run()
        bullet_run.text = "\u25CF  "
        bullet_run.font.size = Pt(font_size - 2)
        bullet_run.font.color.rgb = bullet_color
        bullet_run.font.name = "Calibri"

        # Text
        if isinstance(item, tuple):
            # Bold lead + normal rest
            bold_part, rest_part = item
            r1 = p.add_run()
            r1.text = bold_part
            r1.font.size = Pt(font_size)
            r1.font.bold = True
            r1.font.color.rgb = color
            r1.font.name = "Calibri"
            r2 = p.add_run()
            r2.text = rest_part
            r2.font.size = Pt(font_size)
            r2.font.color.rgb = color
            r2.font.name = "Calibri"
        else:
            run = p.add_run()
            run.text = item
            run.font.size = Pt(font_size)
            run.font.color.rgb = color
            run.font.name = "Calibri"

    return txBox


def add_title_bar(slide, title_text):
    """Add a dark blue title bar at the top of a content slide."""
    from pptx.util import Emu
    # Dark blue rectangle
    shape = slide.shapes.add_shape(
        1,  # rectangle
        Inches(0), Inches(0), SLIDE_WIDTH, Inches(1.1)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = DARK_BLUE
    shape.line.fill.background()

    # Title text
    add_text_box(slide, Inches(0.6), Inches(0.15), Inches(12), Inches(0.8),
                 title_text, font_size=28, bold=True, color=WHITE,
                 align=PP_ALIGN.LEFT)


def add_slide_number(slide, num, total):
    """Add slide number at bottom right."""
    add_text_box(slide, Inches(12.0), Inches(7.0), Inches(1.2), Inches(0.4),
                 f"{num}/{total}", font_size=10, color=RGBColor(0x99, 0x99, 0x99),
                 align=PP_ALIGN.RIGHT)


def add_pptx_table(slide, left, top, width, height, headers, rows,
                   header_color=DARK_BLUE, font_size=12):
    """Add a formatted table to a slide."""
    n_rows = len(rows) + 1
    n_cols = len(headers)
    table_shape = slide.shapes.add_table(n_rows, n_cols, left, top, width, height)
    table = table_shape.table

    # Header
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = header_color
        for paragraph in cell.text_frame.paragraphs:
            paragraph.alignment = PP_ALIGN.CENTER
            for run in paragraph.runs:
                run.font.size = Pt(font_size)
                run.font.bold = True
                run.font.color.rgb = WHITE
                run.font.name = "Calibri"

    # Data
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = table.cell(i + 1, j)
            cell.text = str(val)
            if i % 2 == 1:
                cell.fill.solid()
                cell.fill.fore_color.rgb = LIGHT_BLUE
            for paragraph in cell.text_frame.paragraphs:
                paragraph.alignment = PP_ALIGN.CENTER
                for run in paragraph.runs:
                    run.font.size = Pt(font_size)
                    run.font.color.rgb = DARK_GRAY
                    run.font.name = "Calibri"

    return table_shape


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading data...")
    wide_df = pd.read_csv(WIDE_CSV, low_memory=False)
    long_df = pd.read_csv(LONG_CSV, low_memory=False)
    with open(REPORT_JSON) as f:
        report = json.load(f)

    total_long = report["total_records_long"]
    total_wide = report["total_rows_wide"]
    sources = report["sources"]
    params = report["parameters"]
    n_stations = wide_df["station_name"].nunique()
    n_states = wide_df["state"].nunique()
    year_min = int(wide_df["year"].min())
    year_max = int(wide_df["year"].max())
    n_params = len(params)
    valid_count = int((long_df["quality_flag"] == "valid").sum())
    suspect_con = int((long_df["quality_flag"] == "suspect_consistency").sum())
    suspect_out = int((long_df["quality_flag"] == "suspect_outlier").sum())
    valid_pct = round(valid_count / total_long * 100, 1)

    TOTAL_SLIDES = 16

    print("Building presentation...")
    prs = Presentation()
    prs.slide_width = SLIDE_WIDTH
    prs.slide_height = SLIDE_HEIGHT

    # Use blank layout
    blank_layout = prs.slide_layouts[6]

    # ══════════════════════════════════════════════════════════════════════════
    # SLIDE 1: Title Slide
    # ══════════════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide, DARK_BLUE)

    add_text_box(slide, Inches(1), Inches(1.5), Inches(11.3), Inches(2.5),
                 "A Multi-Source Data Integration\nFramework for Ganga River\nWater Quality Monitoring",
                 font_size=36, bold=True, color=WHITE, align=PP_ALIGN.LEFT,
                 line_spacing=1.3)

    add_text_box(slide, Inches(1), Inches(4.3), Inches(11.3), Inches(0.6),
                 "Harmonizing Heterogeneous Government Datasets",
                 font_size=20, bold=False, color=RGBColor(0xA0, 0xC8, 0xE8),
                 align=PP_ALIGN.LEFT)

    # Divider line
    shape = slide.shapes.add_shape(1, Inches(1), Inches(5.1), Inches(4), Inches(0.03))
    shape.fill.solid()
    shape.fill.fore_color.rgb = ACCENT_ORANGE
    shape.line.fill.background()

    add_text_box(slide, Inches(1), Inches(5.3), Inches(6), Inches(0.4),
                 "Author Name  |  Institution  |  April 2026",
                 font_size=14, color=RGBColor(0x99, 0xBB, 0xDD))
    add_slide_number(slide, 1, TOTAL_SLIDES)

    # ══════════════════════════════════════════════════════════════════════════
    # SLIDE 2: Outline
    # ══════════════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(blank_layout)
    add_title_bar(slide, "Outline")

    outline_items = [
        "Introduction & Motivation",
        "Study Area & Data Sources",
        "Framework Architecture",
        "NWMP PDF Extraction Methodology",
        "Schema Standardization & Ontology Mapping",
        "QA/QC & Trust Scoring",
        "Multi-Source Fusion",
        "Results & Discussion",
        "Limitations & Future Work",
        "Conclusion",
    ]
    add_bullet_slide(slide, outline_items,
                     Inches(1.5), Inches(1.5), Inches(10), Inches(5.5),
                     font_size=20, color=DARK_GRAY, bullet_color=MED_BLUE)
    add_slide_number(slide, 2, TOTAL_SLIDES)

    # ══════════════════════════════════════════════════════════════════════════
    # SLIDE 3: Introduction & Motivation
    # ══════════════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(blank_layout)
    add_title_bar(slide, "Introduction & Motivation")

    items = [
        "Ganga River: 2,525 km, lifeline of 500+ million people across 5 states",
        "Severe water quality degradation from urbanization, industry, and sewage",
        "Three major government data sources exist — but in isolated silos:",
    ]
    add_bullet_slide(slide, items,
                     Inches(0.6), Inches(1.3), Inches(7), Inches(2.5),
                     font_size=16)

    # Sub-bullets for data sources
    sub_items = [
        ("CPCB: ", "Real-time sensors, automated nomenclature, post-2015"),
        ("Data.gov.in: ", "Open portal, diverse schemas, 1963–2022"),
        ("NWMP PDFs: ", "Annual reports, complex table layouts, manual sampling"),
    ]
    add_bullet_slide(slide, sub_items,
                     Inches(1.5), Inches(3.6), Inches(6), Inches(2.0),
                     font_size=14, bullet_color=ACCENT_ORANGE)

    problem_items = [
        "No existing automated framework integrates all three sources",
        "Incompatible formats, naming conventions, temporal resolutions",
        "Research gap: need for systematic, reproducible integration pipeline",
    ]
    add_bullet_slide(slide, problem_items,
                     Inches(0.6), Inches(5.3), Inches(7), Inches(2.0),
                     font_size=15, bullet_color=RGBColor(0xD6, 0x27, 0x28))

    # Key stat box on right
    shape = slide.shapes.add_shape(1, Inches(8.2), Inches(1.5), Inches(4.5), Inches(5.0))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(0xF0, 0xF4, 0xF8)
    shape.line.color.rgb = MED_BLUE

    add_text_box(slide, Inches(8.5), Inches(1.7), Inches(4.0), Inches(0.5),
                 "The Challenge", font_size=18, bold=True, color=DARK_BLUE,
                 align=PP_ALIGN.CENTER)
    stats_text = (
        f"\n{n_stations} stations across {n_states} states\n\n"
        f"3 heterogeneous data sources\n\n"
        f"200+ column name variants\n\n"
        f"PDF reports with year-specific layouts\n\n"
        f"No unified access mechanism"
    )
    add_text_box(slide, Inches(8.5), Inches(2.3), Inches(4.0), Inches(4.0),
                 stats_text, font_size=14, color=DARK_GRAY,
                 align=PP_ALIGN.CENTER, line_spacing=1.4)
    add_slide_number(slide, 3, TOTAL_SLIDES)

    # ══════════════════════════════════════════════════════════════════════════
    # SLIDE 4: Study Area & Data Sources
    # ══════════════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(blank_layout)
    add_title_bar(slide, "Study Area & Data Sources")

    add_text_box(slide, Inches(0.6), Inches(1.3), Inches(12), Inches(0.6),
                 f"Ganga River Basin — 1.08 million km², {n_states} states, {n_stations} monitoring stations",
                 font_size=16, color=DARK_GRAY)

    add_pptx_table(slide,
        Inches(0.6), Inches(2.2), Inches(12), Inches(2.0),
        ["Data Source", "Type", "Records", "Parameters", "Period", "Format", "Trust Score"],
        [
            ["CPCB RTWQMS", "Real-time sensors", f"{sources['cpcb']:,}", "12", "2015–2025", "CSV (API)", "1.00"],
            ["Data.gov.in", "Open data portal", f"{sources['data_gov_in']:,}", "20+", "1963–2022", "CSV (API)", "0.90"],
            ["NWMP Reports", "Annual sampling", f"{sources['nwmp_pdf']:,}", "18", "2015–2024", "PDF tables", "0.85"],
        ],
        font_size=13)

    key_points = [
        f"Total raw records: {sum(sources.values()):,} across all sources",
        f"CPCB: 40 stations with coordinates, continuous monitoring since 2015",
        f"Data.gov.in: 6 curated datasets, highest schema heterogeneity",
        f"NWMP: 7 annual reports (608 pages), complex PDF table layouts",
    ]
    add_bullet_slide(slide, key_points,
                     Inches(0.6), Inches(4.5), Inches(12), Inches(2.5),
                     font_size=15)
    add_slide_number(slide, 4, TOTAL_SLIDES)

    # ══════════════════════════════════════════════════════════════════════════
    # SLIDE 5: Framework Architecture
    # ══════════════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(blank_layout)
    add_title_bar(slide, "Framework Architecture — Six-Stage Pipeline")

    stages = [
        ("1", "Multi-Source\nIngestion", "CSV parsers +\nPDF extractor", MED_BLUE),
        ("2", "Schema\nStandardization", "Ontology mapping\n200+ aliases → 25 params", RGBColor(0x2C, 0xA0, 0x2C)),
        ("3a", "Physical\nRange QA/QC", "Bounds checking\nTrust assignment", RGBColor(0xFF, 0x7F, 0x0E)),
        ("3b", "Outlier\nDetection", "IQR method\n3×IQR threshold", RGBColor(0xFF, 0x7F, 0x0E)),
        ("3c", "Cross-Param\nConsistency", "BOD≤COD\nTDS/EC ratio", RGBColor(0xFF, 0x7F, 0x0E)),
        ("4", "Trust-Weighted\nFusion", "Dedup + conflict\nresolution", RGBColor(0xD6, 0x27, 0x28)),
    ]

    box_w = Inches(1.7)
    box_h = Inches(2.2)
    start_x = Inches(0.5)
    y_top = Inches(1.5)
    gap = Inches(0.3)

    for i, (num, title, desc, color) in enumerate(stages):
        x = start_x + i * (box_w + gap)

        # Box
        shape = slide.shapes.add_shape(1, x, y_top, box_w, box_h)
        shape.fill.solid()
        shape.fill.fore_color.rgb = color
        shape.line.fill.background()

        # Stage number
        add_text_box(slide, x, y_top + Inches(0.1), box_w, Inches(0.35),
                     f"Stage {num}", font_size=11, bold=True, color=WHITE,
                     align=PP_ALIGN.CENTER)
        # Title
        add_text_box(slide, x, y_top + Inches(0.45), box_w, Inches(0.7),
                     title, font_size=13, bold=True, color=WHITE,
                     align=PP_ALIGN.CENTER, line_spacing=1.1)
        # Description
        add_text_box(slide, x, y_top + Inches(1.3), box_w, Inches(0.8),
                     desc, font_size=10, color=RGBColor(0xE8, 0xE8, 0xE8),
                     align=PP_ALIGN.CENTER, line_spacing=1.1)

        # Arrow between boxes
        if i < len(stages) - 1:
            arrow_x = x + box_w
            add_text_box(slide, arrow_x, y_top + Inches(0.8), gap, Inches(0.4),
                         "→", font_size=24, bold=True, color=DARK_GRAY,
                         align=PP_ALIGN.CENTER)

    # Output row
    shape = slide.shapes.add_shape(1, Inches(3.5), Inches(4.2), Inches(6), Inches(0.6))
    shape.fill.solid()
    shape.fill.fore_color.rgb = DARK_BLUE
    shape.line.fill.background()
    add_text_box(slide, Inches(3.5), Inches(4.25), Inches(6), Inches(0.5),
                 "Output: Long CSV  +  Wide CSV  +  Provenance Report",
                 font_size=14, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    # Key design principles
    principles = [
        "Modular & reproducible — each stage independent",
        "Long-format internal representation (one obs per row)",
        "Full provenance tracking through entire pipeline",
        "Quality flags retained, not discarded — downstream filtering",
    ]
    add_bullet_slide(slide, principles,
                     Inches(0.6), Inches(5.2), Inches(12), Inches(2.0),
                     font_size=14)
    add_slide_number(slide, 5, TOTAL_SLIDES)

    # ══════════════════════════════════════════════════════════════════════════
    # SLIDE 6: NWMP PDF Extraction
    # ══════════════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(blank_layout)
    add_title_bar(slide, "NWMP PDF Table Extraction Methodology")

    strategies = [
        ("Page Relevance Filtering: ", "Scan for 18 Ganga system keywords → ~80% page reduction"),
        ("Multi-Strategy Table Detection: ", "3 fallback strategies — line-based → text-based → strict"),
        ("Header Pattern Recognition: ", "21 regex patterns, multi-line header combination"),
        ("Fragmented Table Reassembly: ", "Cross-page tables detected via header similarity"),
        ("Value Parsing: ", "BDL → 0, ranges averaged, NR/NA → null"),
    ]
    add_bullet_slide(slide, strategies,
                     Inches(0.6), Inches(1.3), Inches(6.5), Inches(3.5),
                     font_size=14)

    # NWMP extraction table on right
    add_pptx_table(slide,
        Inches(7.5), Inches(1.3), Inches(5.3), Inches(3.5),
        ["Year", "Pages", "Relevant", "Records"],
        [
            ["2015", "195", "68", "170"],
            ["2018", "56", "10", "4"],
            ["2019", "78", "13", "13"],
            ["2020", "76", "13", "17"],
            ["2021", "70", "14", "17"],
            ["2022", "70", "16", "17"],
            ["2024", "63", "15", "18"],
            ["Total", "608", "149", "256"],
        ],
        font_size=12)

    highlights = [
        "68% tables extracted via line-based strategy",
        "24% required text-based fallback",
        "8% used strict mode for well-formatted tables",
        "256 station records → 732 parameter measurements after long-form conversion",
    ]
    add_bullet_slide(slide, highlights,
                     Inches(0.6), Inches(5.0), Inches(12), Inches(2.0),
                     font_size=14, bullet_color=ACCENT_GREEN)
    add_slide_number(slide, 6, TOTAL_SLIDES)

    # ══════════════════════════════════════════════════════════════════════════
    # SLIDE 7: Schema Standardization
    # ══════════════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(blank_layout)
    add_title_bar(slide, "Schema Standardization & Ontology Mapping")

    items = [
        "200+ column name variants mapped to 25 canonical parameters",
        "Handles: spelling (sulphate/sulfate), abbreviations (DO/dissolved_oxygen)",
        "Unit-embedded names (conductivity_mhos_cm), source-specific (CPCB vs data.gov.in)",
        "State normalization: 15-entry mapping (UTTARANCHAL → Uttarakhand)",
        "NWMP records filtered to Ganga basin states only (removed 247 non-Ganga records)",
    ]
    add_bullet_slide(slide, items,
                     Inches(0.6), Inches(1.3), Inches(6.5), Inches(3.0),
                     font_size=15)

    # Example mapping table
    add_text_box(slide, Inches(7.5), Inches(1.3), Inches(5.3), Inches(0.4),
                 "Example Column Mappings", font_size=14, bold=True,
                 color=DARK_BLUE, align=PP_ALIGN.CENTER)
    add_pptx_table(slide,
        Inches(7.5), Inches(1.8), Inches(5.3), Inches(3.0),
        ["Source Column", "→", "Canonical"],
        [
            ["Biochemical Oxygen Demand", "→", "bod"],
            ["dissolved_oxygen_mg_l", "→", "do"],
            ["conductivity_mhos_cm", "→", "conductivity"],
            ["Water Temperature", "→", "temperature"],
            ["fc_mpn_100ml", "→", "fecal_coliform"],
            ["sulphate / sulfate / so4", "→", "sulphate"],
            ["UTTAR PRADESH", "→", "Uttar Pradesh"],
            ["UTTRAKHAND", "→", "Uttarakhand"],
        ],
        font_size=11)

    add_text_box(slide, Inches(0.6), Inches(4.5), Inches(12), Inches(0.5),
                 f"Result: 1,090,673 raw records → 950,896 after standardization (139,777 dropped — unmapped params, null values, non-Ganga)",
                 font_size=15, bold=True, color=DARK_BLUE)
    add_slide_number(slide, 7, TOTAL_SLIDES)

    # ══════════════════════════════════════════════════════════════════════════
    # SLIDE 8: QA/QC & Trust Scoring
    # ══════════════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(blank_layout)
    add_title_bar(slide, "QA/QC & Trust Scoring — Three-Tier Validation")

    # Three boxes side by side
    tiers = [
        ("Physical Range\nValidation", "5,793 rejected",
         "pH: 0–14\nDO: 0–25 mg/L\nTemp: 0–50°C\nBOD: 0–1000 mg/L",
         RGBColor(0xD6, 0x27, 0x28)),
        ("Statistical Outlier\nDetection", "34,926 flagged",
         "IQR method per param\n3×IQR threshold\nTrust halved (×0.5)\nRecords retained",
         ACCENT_ORANGE),
        ("Cross-Parameter\nConsistency", "40,225 flagged",
         "BOD ≤ COD (×1.2)\nTDS/EC ratio: 0.2–1.5\nTrust reduced (×0.7)\nRecords retained",
         MED_BLUE),
    ]

    for i, (title, stat, desc, color) in enumerate(tiers):
        x = Inches(0.6) + i * Inches(4.2)
        w = Inches(3.8)

        # Box
        shape = slide.shapes.add_shape(1, x, Inches(1.4), w, Inches(3.2))
        shape.fill.solid()
        shape.fill.fore_color.rgb = color
        shape.line.fill.background()

        add_text_box(slide, x, Inches(1.5), w, Inches(0.7),
                     title, font_size=16, bold=True, color=WHITE,
                     align=PP_ALIGN.CENTER, line_spacing=1.1)
        add_text_box(slide, x, Inches(2.2), w, Inches(0.4),
                     stat, font_size=22, bold=True, color=RGBColor(0xFF, 0xFF, 0xCC),
                     align=PP_ALIGN.CENTER)
        add_text_box(slide, x, Inches(2.8), w, Inches(1.5),
                     desc, font_size=12, color=RGBColor(0xE8, 0xE8, 0xE8),
                     align=PP_ALIGN.CENTER, line_spacing=1.3)

    # Trust scoring
    add_text_box(slide, Inches(0.6), Inches(4.9), Inches(12), Inches(0.5),
                 "Trust Score Assignment", font_size=18, bold=True, color=DARK_BLUE)

    add_pptx_table(slide,
        Inches(0.6), Inches(5.5), Inches(8), Inches(1.5),
        ["Source", "Base Trust", "After Outlier Flag", "After Consistency Flag", "Near Bounds"],
        [
            ["CPCB", "1.00", "×0.5 → 0.50", "×0.7 → 0.70", "×0.7"],
            ["Data.gov.in", "0.90", "×0.5 → 0.45", "×0.7 → 0.63", "×0.7"],
            ["NWMP PDF", "0.85", "×0.5 → 0.43", "×0.7 → 0.60", "×0.7"],
        ],
        font_size=12)
    add_slide_number(slide, 8, TOTAL_SLIDES)

    # ══════════════════════════════════════════════════════════════════════════
    # SLIDE 9: Multi-Source Fusion
    # ══════════════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(blank_layout)
    add_title_bar(slide, "Multi-Source Fusion & Conflict Resolution")

    items = [
        "Records grouped by (station_name, date, parameter)",
        "Station name normalization removes noise words (station, monitoring, WQ, CPCB)",
        "Within each group: highest trust score wins",
        "If values agree within 10% → trust boosted by 1.2× (inter-source agreement)",
        "All contributing sources preserved in provenance field",
    ]
    add_bullet_slide(slide, items,
                     Inches(0.6), Inches(1.4), Inches(6.5), Inches(3.0),
                     font_size=16)

    # Stats box
    shape = slide.shapes.add_shape(1, Inches(7.8), Inches(1.4), Inches(5.0), Inches(3.5))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(0xF0, 0xF4, 0xF8)
    shape.line.color.rgb = MED_BLUE

    add_text_box(slide, Inches(8.0), Inches(1.5), Inches(4.6), Inches(0.4),
                 "Fusion Results", font_size=18, bold=True, color=DARK_BLUE,
                 align=PP_ALIGN.CENTER)
    fusion_stats = (
        f"81,158 conflicts resolved\n\n"
        f"73% → CPCB selected (highest trust)\n\n"
        f"18% → Data.gov.in preferred\n"
        f"      (CPCB penalized by QA/QC)\n\n"
        f"9% → NWMP (sole source)"
    )
    add_text_box(slide, Inches(8.2), Inches(2.1), Inches(4.4), Inches(2.5),
                 fusion_stats, font_size=14, color=DARK_GRAY,
                 align=PP_ALIGN.LEFT, line_spacing=1.3)

    # Output format
    add_text_box(slide, Inches(0.6), Inches(4.8), Inches(12), Inches(0.4),
                 "Final Output Formats:", font_size=16, bold=True, color=DARK_BLUE)

    output_items = [
        ("Wide format: ", f"{total_wide:,} rows × 35 columns (one row per station-date, all params as columns)"),
        ("Long format: ", f"{total_long:,} rows × 13 columns (one row per measurement, with quality flags)"),
        ("Provenance report: ", "JSON metadata with pipeline run statistics"),
    ]
    add_bullet_slide(slide, output_items,
                     Inches(0.6), Inches(5.3), Inches(12), Inches(1.8),
                     font_size=14)
    add_slide_number(slide, 9, TOTAL_SLIDES)

    # ══════════════════════════════════════════════════════════════════════════
    # SLIDE 10: Results — Key Numbers
    # ══════════════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(blank_layout)
    add_title_bar(slide, "Results — Key Numbers")

    # Big stat boxes
    big_stats = [
        (f"{total_long:,}", "Total\nMeasurements", MED_BLUE),
        (f"{total_wide:,}", "Station-Date\nObservations", ACCENT_GREEN),
        (f"{n_stations}", "Unique\nStations", ACCENT_ORANGE),
        (f"{n_params}", "Water Quality\nParameters", RGBColor(0xD6, 0x27, 0x28)),
        (f"{year_min}–{year_max}", "Temporal\nSpan", RGBColor(0x94, 0x67, 0xBD)),
        (f"{valid_pct}%", "Records\nValid", RGBColor(0x17, 0xBE, 0xCF)),
    ]

    for i, (val, label, color) in enumerate(big_stats):
        row = i // 3
        col = i % 3
        x = Inches(0.6) + col * Inches(4.2)
        y = Inches(1.4) + row * Inches(2.8)

        shape = slide.shapes.add_shape(1, x, y, Inches(3.8), Inches(2.4))
        shape.fill.solid()
        shape.fill.fore_color.rgb = color
        shape.line.fill.background()

        add_text_box(slide, x, y + Inches(0.3), Inches(3.8), Inches(1.0),
                     val, font_size=36, bold=True, color=WHITE,
                     align=PP_ALIGN.CENTER)
        add_text_box(slide, x, y + Inches(1.4), Inches(3.8), Inches(0.7),
                     label, font_size=16, color=RGBColor(0xE8, 0xE8, 0xE8),
                     align=PP_ALIGN.CENTER, line_spacing=1.1)
    add_slide_number(slide, 10, TOTAL_SLIDES)

    # ══════════════════════════════════════════════════════════════════════════
    # SLIDE 11: Results — Source Distribution & Quality
    # ══════════════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(blank_layout)
    add_title_bar(slide, "Results — Source Distribution & Quality Assessment")

    if (FIGURES_DIR / "fig1_source_dist.png").exists():
        slide.shapes.add_picture(str(FIGURES_DIR / "fig1_source_dist.png"),
                                 Inches(0.4), Inches(1.3), Inches(6.0))
    if (FIGURES_DIR / "fig4_quality.png").exists():
        slide.shapes.add_picture(str(FIGURES_DIR / "fig4_quality.png"),
                                 Inches(6.8), Inches(1.3), Inches(6.0))

    add_text_box(slide, Inches(0.4), Inches(5.0), Inches(6.0), Inches(0.3),
                 "Fig. 1: Record distribution by source", font_size=11,
                 italic=True, color=DARK_GRAY, align=PP_ALIGN.CENTER)
    add_text_box(slide, Inches(6.8), Inches(5.0), Inches(6.0), Inches(0.3),
                 "Fig. 2: Quality flag distribution after QA/QC", font_size=11,
                 italic=True, color=DARK_GRAY, align=PP_ALIGN.CENTER)

    summary_items = [
        f"Data.gov.in: {sources['data_gov_in']:,} records ({sources['data_gov_in']/total_long*100:.1f}%) — largest volume, highest schema diversity",
        f"CPCB: {sources['cpcb']:,} records ({sources['cpcb']/total_long*100:.1f}%) — highest trust, post-2015 only",
        f"NWMP: {sources['nwmp_pdf']:,} records ({sources['nwmp_pdf']/total_long*100:.1f}%) — unique historical & spatial coverage",
    ]
    add_bullet_slide(slide, summary_items,
                     Inches(0.4), Inches(5.5), Inches(12.5), Inches(1.5),
                     font_size=14)
    add_slide_number(slide, 11, TOTAL_SLIDES)

    # ══════════════════════════════════════════════════════════════════════════
    # SLIDE 12: Results — Parameter Coverage
    # ══════════════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(blank_layout)
    add_title_bar(slide, "Results — Parameter Coverage & Temporal Distribution")

    if (FIGURES_DIR / "fig2_param_coverage.png").exists():
        slide.shapes.add_picture(str(FIGURES_DIR / "fig2_param_coverage.png"),
                                 Inches(0.3), Inches(1.3), Inches(6.2))
    if (FIGURES_DIR / "fig3_temporal.png").exists():
        slide.shapes.add_picture(str(FIGURES_DIR / "fig3_temporal.png"),
                                 Inches(6.8), Inches(1.3), Inches(6.2))

    add_text_box(slide, Inches(0.3), Inches(5.3), Inches(6.2), Inches(0.3),
                 "Fig. 3: Parameter fill rates (top 15)", font_size=11,
                 italic=True, color=DARK_GRAY, align=PP_ALIGN.CENTER)
    add_text_box(slide, Inches(6.8), Inches(4.5), Inches(6.2), Inches(0.3),
                 "Fig. 4: Temporal distribution (2000–2025)", font_size=11,
                 italic=True, color=DARK_GRAY, align=PP_ALIGN.CENTER)

    items = [
        "pH highest fill rate (98.3%), followed by conductivity (90.7%) and chloride (89.7%)",
        "Sharp increase in data volume from 2015 — coincides with CPCB RTWQMS deployment",
        "Arsenic (1.6%) and TSS (0.8%) available primarily from NWMP and data.gov.in",
    ]
    add_bullet_slide(slide, items,
                     Inches(0.3), Inches(5.7), Inches(12.5), Inches(1.5),
                     font_size=14)
    add_slide_number(slide, 12, TOTAL_SLIDES)

    # ══════════════════════════════════════════════════════════════════════════
    # SLIDE 13: Results — Spatial & Trust
    # ══════════════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(blank_layout)
    add_title_bar(slide, "Results — Spatial Coverage & Trust Scores")

    if (FIGURES_DIR / "fig5_states.png").exists():
        slide.shapes.add_picture(str(FIGURES_DIR / "fig5_states.png"),
                                 Inches(0.3), Inches(1.3), Inches(6.2))
    if (FIGURES_DIR / "fig6_trust.png").exists():
        slide.shapes.add_picture(str(FIGURES_DIR / "fig6_trust.png"),
                                 Inches(6.8), Inches(1.3), Inches(6.2))

    add_text_box(slide, Inches(0.3), Inches(5.0), Inches(6.2), Inches(0.3),
                 "Fig. 5: State-wise observation distribution", font_size=11,
                 italic=True, color=DARK_GRAY, align=PP_ALIGN.CENTER)
    add_text_box(slide, Inches(6.8), Inches(4.5), Inches(6.2), Inches(0.3),
                 "Fig. 6: Trust score distribution", font_size=11,
                 italic=True, color=DARK_GRAY, align=PP_ALIGN.CENTER)

    items = [
        "Uttar Pradesh contributes largest share — extensive Ganga + Yamuna frontage",
        "Bimodal trust distribution: CPCB cluster near 1.0, data.gov.in near 0.90",
        "Left tail = records penalized by QA/QC stages",
    ]
    add_bullet_slide(slide, items,
                     Inches(0.3), Inches(5.5), Inches(12.5), Inches(1.5),
                     font_size=14)
    add_slide_number(slide, 13, TOTAL_SLIDES)

    # ══════════════════════════════════════════════════════════════════════════
    # SLIDE 14: Pipeline Summary Table
    # ══════════════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(blank_layout)
    add_title_bar(slide, "Pipeline Summary Statistics")

    add_pptx_table(slide,
        Inches(1.5), Inches(1.5), Inches(10), Inches(4.5),
        ["Pipeline Stage", "Metric", "Value"],
        [
            ["Ingestion", "Raw records loaded", f"{sum(sources.values()):,}"],
            ["Standardization", "After schema alignment", "950,896"],
            ["", "Dropped (unmapped/null/non-Ganga)", "139,777"],
            ["QA/QC — Range", "Rejected (out of bounds)", "5,793"],
            ["QA/QC — Outlier", "Flagged (3×IQR)", f"{suspect_out:,}"],
            ["QA/QC — Consistency", "Flagged (BOD>COD, TDS/EC)", f"{suspect_con:,}"],
            ["Fusion", "After deduplication", f"{total_long:,}"],
            ["", "Conflicts resolved", "81,158"],
            ["Output — Wide", "Station-date rows × columns", f"{total_wide:,} × 35"],
            ["Output — Long", "Measurement rows × columns", f"{total_long:,} × 13"],
        ],
        font_size=14)

    add_text_box(slide, Inches(1.5), Inches(6.2), Inches(10), Inches(0.5),
                 f"Final: {valid_pct}% valid  |  {suspect_con/total_long*100:.1f}% suspect (consistency)  |  {suspect_out/total_long*100:.1f}% suspect (outlier)",
                 font_size=16, bold=True, color=DARK_BLUE, align=PP_ALIGN.CENTER)
    add_slide_number(slide, 14, TOTAL_SLIDES)

    # ══════════════════════════════════════════════════════════════════════════
    # SLIDE 15: Limitations & Future Work
    # ══════════════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(blank_layout)
    add_title_bar(slide, "Limitations & Future Work")

    lim_items = [
        "No ground-truth comparison against known-clean reference dataset",
        "Station coordinates available only for CPCB stations (no spatial validation)",
        "Temporal resolution mismatch: hourly CPCB vs. annual NWMP",
        "PDF keyword matching may miss implicit Ganga references",
        "Some parameters from single source only — limits cross-validation",
    ]
    add_text_box(slide, Inches(0.6), Inches(1.3), Inches(5.5), Inches(0.4),
                 "Limitations", font_size=18, bold=True, color=RGBColor(0xD6, 0x27, 0x28))
    add_bullet_slide(slide, lim_items,
                     Inches(0.6), Inches(1.8), Inches(5.5), Inches(3.5),
                     font_size=14, bullet_color=RGBColor(0xD6, 0x27, 0x28))

    future_items = [
        "Integrate satellite-derived water quality indices (Sentinel-2)",
        "Extend to other Indian river systems (Yamuna, Brahmaputra, Godavari)",
        "Develop real-time monitoring dashboard with integrated data",
        "Apply deep learning models (CNN, LSTM) for water quality prediction",
        "Cross-validate with field measurements for ground truth",
    ]
    add_text_box(slide, Inches(7.0), Inches(1.3), Inches(5.8), Inches(0.4),
                 "Future Work", font_size=18, bold=True, color=ACCENT_GREEN)
    add_bullet_slide(slide, future_items,
                     Inches(7.0), Inches(1.8), Inches(5.8), Inches(3.5),
                     font_size=14, bullet_color=ACCENT_GREEN)
    add_slide_number(slide, 15, TOTAL_SLIDES)

    # ══════════════════════════════════════════════════════════════════════════
    # SLIDE 16: Conclusion / Thank You
    # ══════════════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide, DARK_BLUE)

    add_text_box(slide, Inches(1), Inches(0.8), Inches(11.3), Inches(0.8),
                 "Conclusion", font_size=30, bold=True, color=WHITE,
                 align=PP_ALIGN.LEFT)

    conclusion_items = [
        f"Comprehensive 6-stage framework integrating CPCB + data.gov.in + NWMP PDFs",
        f"Produced {total_long:,} measurements across {n_params} parameters from {n_stations} stations ({year_min}–{year_max})",
        f"{valid_pct}% records pass all quality checks with dynamic trust scoring",
        f"81,158 inter-source conflicts resolved via trust-weighted fusion",
        f"Custom PDF extractor handles 7 years of NWMP reports with year-specific layouts",
        f"Most comprehensive publicly-derived Ganga water quality compilation to date",
        f"Framework is reusable and extensible to other river systems",
    ]
    add_bullet_slide(slide, conclusion_items,
                     Inches(1), Inches(1.8), Inches(11.3), Inches(3.5),
                     font_size=17, color=WHITE, bullet_color=ACCENT_ORANGE)

    # Divider
    shape = slide.shapes.add_shape(1, Inches(1), Inches(5.5), Inches(4), Inches(0.03))
    shape.fill.solid()
    shape.fill.fore_color.rgb = ACCENT_ORANGE
    shape.line.fill.background()

    add_text_box(slide, Inches(1), Inches(5.8), Inches(11.3), Inches(0.6),
                 "Thank You", font_size=28, bold=True, color=WHITE,
                 align=PP_ALIGN.LEFT)
    add_text_box(slide, Inches(1), Inches(6.5), Inches(11.3), Inches(0.5),
                 "Questions?  |  Author Name  |  email@example.com",
                 font_size=14, color=RGBColor(0x99, 0xBB, 0xDD))
    add_slide_number(slide, 16, TOTAL_SLIDES)

    # ── Save ──────────────────────────────────────────────────────────────────
    prs.save(str(OUTPUT_PPTX))
    print(f"\nPresentation saved to: {OUTPUT_PPTX}")
    print(f"Total slides: {TOTAL_SLIDES}")


if __name__ == "__main__":
    main()
