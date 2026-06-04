"""
Parse NWMP HTML files (CPCB ENVIS water quality tables) into clean CSVs.

Each HTM file contains one table with columns:
  Station Code | Location | State | Temp(Min/Max/Mean) | DO(Min/Max/Mean) |
  pH(Min/Max/Mean) | Conductivity(Min/Max/Mean) | BOD(Min/Max/Mean) |
  Nitrate-N(Min/Max/Mean) | Fecal Coliform(Min/Max/Mean) | Total Coliform(Min/Max/Mean)

Output: One CSV per year with flat columns, plus a merged all-years CSV.
"""
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup


def parse_nwmp_htm(filepath: Path, year: int) -> pd.DataFrame:
    """Parse a single NWMP HTM file into a DataFrame."""
    with open(filepath, encoding="windows-1252", errors="replace") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    table = soup.find("table")
    if not table:
        raise ValueError(f"No table found in {filepath.name}")

    rows = table.find_all("tr")

    # Parameters in order (each has Min/Max/Mean sub-columns)
    params = [
        "Temperature_C", "DO_mg_l", "pH", "Conductivity_umhos_cm",
        "BOD_mg_l", "Nitrate_N_mg_l", "Fecal_Coliform_MPN_100ml",
        "Total_Coliform_MPN_100ml",
    ]

    # Build column names
    columns = ["Station_Code", "Location", "State"]
    for p in params:
        columns.extend([f"{p}_Min", f"{p}_Max", f"{p}_Mean"])

    records = []
    for row in rows[6:]:  # skip header/criteria rows (0-5)
        cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]

        # Skip empty rows, section headers, criteria rows
        if len(cells) < 10:
            continue
        # Data rows have 27 cells (3 meta + 8 params × 3 stats)
        if len(cells) < 27:
            continue

        # First cell should look like a station code (numeric)
        code = cells[0].strip()
        if not code or not any(c.isdigit() for c in code):
            continue

        # Clean up values
        row_data = {}
        row_data["Station_Code"] = code
        row_data["Location"] = re.sub(r"\s+", " ", cells[1]).strip()
        row_data["State"] = re.sub(r"\s+", " ", cells[2]).strip()

        # Each parameter has 3 values (Min, Max, Mean) starting at index 3
        idx = 3
        for p in params:
            for stat in ["Min", "Max", "Mean"]:
                key = f"{p}_{stat}"
                val = cells[idx].strip() if idx < len(cells) else ""
                # Clean numeric values
                val = val.replace(",", "").replace("*", "").strip()
                if val in ("", "-", "NR", "NA", "N/A", "NM"):
                    row_data[key] = None
                else:
                    try:
                        row_data[key] = float(val)
                    except ValueError:
                        row_data[key] = None
                idx += 1

        records.append(row_data)

    df = pd.DataFrame(records)
    df["Year"] = year
    return df


def main():
    nwmp_dir = Path(__file__).resolve().parent / "NWMP data"
    out_dir = Path(__file__).resolve().parent / "data" / "web_extracted" / "nwmp"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = {
        2012: "RIVER GANGA-2012.htm",
        2013: "RIVERWATER DATA 2013_5.htm",
        2014: "RIVERWATER DATA 2014_5.htm",
    }

    all_dfs = []
    for year, fname in sorted(files.items()):
        filepath = nwmp_dir / fname
        if not filepath.exists():
            print(f"  MISSING: {fname}")
            continue

        df = parse_nwmp_htm(filepath, year)
        outfile = out_dir / f"nwmp_ganga_{year}.csv"
        df.to_csv(outfile, index=False)
        all_dfs.append(df)

        print(f"  {year}: {len(df)} stations, saved → {outfile.name}")
        print(f"    States: {sorted(df['State'].unique())}")
        print(f"    Params with data: ", end="")
        param_cols = [c for c in df.columns if c not in
                      ("Station_Code", "Location", "State", "Year")]
        data_cols = [c for c in param_cols if df[c].notna().any()]
        # Unique param names
        unique_params = sorted(set(c.rsplit("_", 1)[0] for c in data_cols
                                   if c.endswith(("_Min", "_Max", "_Mean"))))
        print(unique_params)

    # Merge all years
    if all_dfs:
        merged = pd.concat(all_dfs, ignore_index=True)
        merged_file = out_dir / "nwmp_ganga_all_years.csv"
        merged.to_csv(merged_file, index=False)
        print(f"\n  Combined: {len(merged)} rows ({merged['Year'].nunique()} years)")
        print(f"  Unique stations: {merged['Station_Code'].nunique()}")
        print(f"  → {merged_file.name}")

        # Summary stats for Mean columns
        print("\n  Parameter summary (Mean values across all years):")
        for p in ["Temperature_C", "DO_mg_l", "pH", "Conductivity_umhos_cm",
                   "BOD_mg_l", "Nitrate_N_mg_l", "Fecal_Coliform_MPN_100ml",
                   "Total_Coliform_MPN_100ml"]:
            col = f"{p}_Mean"
            if col in merged.columns:
                vals = pd.to_numeric(merged[col], errors="coerce")
                n = vals.notna().sum()
                if n > 0:
                    print(f"    {p}: n={n}, mean={vals.mean():.2f}, "
                          f"range=[{vals.min():.1f}, {vals.max():.1f}]")


if __name__ == "__main__":
    main()
