"""CPCB hourly water-quality scraper.

Fetches the latest readings from the CPCB real-time water quality monitoring
endpoint and writes them to a timestamped CSV in the current working directory.

Designed to be invoked by the `Scrape CPCB Data` GitHub Action every hour.
"""
from __future__ import annotations

import os
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning

# CPCB endpoint serves over a self-signed cert chain - silence the warnings.
warnings.simplefilter("ignore", category=InsecureRequestWarning)

CPCB_URL = "https://rtwqmsdb1.cpcb.gov.in/data/internet/layers/10/index.json"

PARAMETER_NAME_MAP = {
    "River Stage": "Water Level",
    "Oxygen, dissolved": "Dissolved Oxygen",
}

UNIT_MAP = {
    "River Stage": "m above MSL",
}


def fetch_data(url: str) -> list:
    """GET the CPCB JSON feed. Returns [] on any error."""
    try:
        response = requests.get(url, verify=False, timeout=60)
    except requests.exceptions.RequestException as exc:
        print(f"Request failed: {exc}")
        return []

    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text[:200]}")
        return []

    try:
        return response.json()
    except ValueError as exc:
        print(f"Could not decode JSON: {exc}")
        return []


def _parse_timestamp(timestamp: str):
    """Return a datetime for either '...%fZ' or '...%SZ' formats, else None."""
    if not timestamp:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(timestamp, fmt)
        except ValueError:
            continue
    print(f"Invalid timestamp format: {timestamp}")
    return None


def process_data(data: list) -> list[dict]:
    """Map the CPCB payload to our flat schema."""
    out: list[dict] = []
    for dat in data:
        long_name = dat.get("stationparameter_longname", "")
        out.append({
            "stationId": dat.get("station_id", ""),
            "timestamp": dat.get("timestamp", ""),
            "timestampDate": _parse_timestamp(dat.get("timestamp", "")),
            "value": dat.get("ts_value", ""),
            "unit": UNIT_MAP.get(long_name, dat.get("ts_unitsymbol", "")),
            "parameterNo": dat.get("stationparameter_no", ""),
            "parameterName": PARAMETER_NAME_MAP.get(long_name, long_name),
        })
    return out
LOG_CSV = "scrape_log.csv"
STATUS_MD = "STATUS.md"


def _update_progress(run_time: datetime, status: str, filename: str,
                     rows: int, df: pd.DataFrame | None) -> None:
    """Append a row to scrape_log.csv and rewrite STATUS.md.

    The log lets you see, at a glance, every hourly run's outcome
    (success / fail), how many rows it pulled, and how many distinct
    stations / parameters were in the snapshot.
    """
    stations = parameters = 0
    earliest = latest = ""
    if df is not None and not df.empty:
        stations = int(df["stationId"].astype(str).nunique())
        parameters = int(df["parameterNo"].astype(str).nunique())
        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        ts = ts.dropna()
        if not ts.empty:
            earliest = ts.min().isoformat()
            latest = ts.max().isoformat()

    log_row = {
        "run_utc": run_time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "filename": filename,
        "rows": rows,
        "stations": stations,
        "parameters": parameters,
        "earliest_obs": earliest,
        "latest_obs": latest,
    }

    log_path = Path(LOG_CSV)
    log_df = pd.DataFrame([log_row])
    if log_path.exists():
        try:
            existing = pd.read_csv(log_path)
            log_df = pd.concat([existing, log_df], ignore_index=True)
        except Exception as exc:                         # noqa: BLE001
            print(f"Could not read existing log, starting fresh: {exc}")
    log_df.to_csv(log_path, index=False)

    # Human-readable progress dashboard
    total_runs = len(log_df)
    successes = int((log_df["status"] == "ok").sum())
    failures = total_runs - successes
    last10 = log_df.tail(10).to_markdown(index=False) if total_runs else ""
    Path(STATUS_MD).write_text(
        "# CPCB Scraper Progress\n\n"
        f"- Last run (UTC): **{log_row['run_utc']}**\n"
        f"- Last status: **{status}**\n"
        f"- Total runs logged: **{total_runs}**\n"
        f"- Successful: **{successes}**  |  Failed: **{failures}**\n"
        f"- Last file: `{filename or '-'}` ({rows} rows, "
        f"{stations} stations, {parameters} parameters)\n\n"
        "## Last 10 runs\n\n"
        f"{last10}\n",
        encoding="utf-8",
    )


def main() -> None:
    run_time = datetime.now(timezone.utc).replace(tzinfo=None)
    raw = fetch_data(CPCB_URL)
    if not raw:
        print("No data fetched. Exiting.")
        _update_progress(run_time, status="fail_no_data", filename="",
                         rows=0, df=None)
        # Non-zero exit so the GitHub Action surfaces the failure
        raise SystemExit(1)

    df = pd.DataFrame(process_data(raw))
    timestamp_str = run_time.strftime("%Y-%m-%d_%H-%M")
    filename = f"water_data_{timestamp_str}.csv"
    df.to_csv(filename, index=False)
    print(f"Data saved successfully as '{filename}' ({len(df)} rows).")

    _update_progress(run_time, status="ok", filename=filename,
                     rows=len(df), df=df)


if __name__ == "__main__":
    main()
