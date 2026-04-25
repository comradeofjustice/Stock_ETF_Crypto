#!/usr/bin/env python3
"""
08_generate_index_excel.py
==========================
Generate the index Excel file TTP_Stock_ETF_Index.xlsx with hyperlinks to reports.

Sheets:
    Sheet_Universe     - All assets with ticker, name, category, marketCap, date range
    Sheet_DataPaths    - Qlib data path, CSV path, features parquet path per asset
    Sheet_ReportLinks  - HTML report paths with HYPERLINK formulas
    Sheet_RunLog       - Script execution log with timing and error counts

Input:
    config/universe.yaml
    data/raw/stocks/*.csv, etfs/*.csv, crypto/*.csv
    data/features/*.parquet
    data/qlib_data/*/
    output/download_errors.log
    reports/*.html, reports/etfs/*.html, reports/crypto/*.html
Output:
    output/TTP_Stock_ETF_Index.xlsx
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import openpyxl
import pandas as pd
import yaml
from openpyxl.utils import get_column_letter

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
QLIB_DIR = PROJECT_ROOT / "data" / "qlib_data"
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUT_DIR = PROJECT_ROOT / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_universe() -> Dict:
    """Load universe classification."""
    yaml_path = CONFIG_DIR / "universe.yaml"
    if not yaml_path.exists():
        return {}
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_date_range_from_csv(csv_path: Path) -> tuple:
    """Get start and end dates from a CSV file."""
    try:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        if not df.empty:
            return df.index.min().strftime("%Y-%m-%d"), df.index.max().strftime("%Y-%m-%d")
    except Exception:
        pass
    return "", ""


def generate_excel(run_log: List[Dict] = None, error_count: int = 0) -> None:
    """Generate the index Excel file."""
    logger.info("Generating index Excel file")

    wb = openpyxl.Workbook()
    universe = load_universe()

    # ── Sheet_Universe ─────────────────────────────────────────────────
    ws_universe = wb.active
    ws_universe.title = "Sheet_Universe"
    headers = ["Ticker", "Name", "Category", "MarketCap(USD)", "DataStart", "DataEnd"]
    ws_universe.append(headers)

    # Stocks
    for cap_key, category in [("large", "large_cap"), ("mid", "mid_cap"), ("small", "small_cap"), ("unclassified", "unclassified")]:
        tickers_info = universe.get(cap_key, [])
        for info in tickers_info:
            ticker = info["ticker"]
            csv_path = RAW_DIR / "stocks" / f"{ticker}.csv"
            start, end = get_date_range_from_csv(csv_path)
            ws_universe.append([
                ticker,
                info.get("name", ""),
                category,
                info.get("marketCap", ""),
                start,
                end,
            ])

    # ETFs
    etf_dir = RAW_DIR / "etfs"
    if etf_dir.exists():
        for csv_file in sorted(etf_dir.glob("*.csv")):
            ticker = csv_file.stem
            start, end = get_date_range_from_csv(csv_file)
            ws_universe.append([ticker, ticker, "etf", "", start, end])

    # Crypto
    crypto_dir = RAW_DIR / "crypto"
    if crypto_dir.exists():
        for csv_file in sorted(crypto_dir.glob("*.csv")):
            symbol = csv_file.stem
            start, end = get_date_range_from_csv(csv_file)
            ws_universe.append([symbol, symbol, "crypto", "", start, end])

    # Auto-width
    for col_idx, col_name in enumerate(headers, 1):
        ws_universe.column_dimensions[get_column_letter(col_idx)].width = max(12, len(col_name) + 4)

    # ── Sheet_DataPaths ────────────────────────────────────────────────
    ws_paths = wb.create_sheet("Sheet_DataPaths")
    headers = ["Ticker", "RawCSVPath", "FeaturesPath", "QlibDataPath"]
    ws_paths.append(headers)

    # Collect all tickers
    all_tickers = []

    # Stocks
    stock_dir = RAW_DIR / "stocks"
    if stock_dir.exists():
        for csv_file in sorted(stock_dir.glob("*.csv")):
            ticker = csv_file.stem
            features_path = FEATURES_DIR / f"{ticker}_features.parquet"
            all_tickers.append({
                "ticker": ticker,
                "raw_csv": str(csv_file.relative_to(PROJECT_ROOT)),
                "features": str(features_path.relative_to(PROJECT_ROOT)) if features_path.exists() else "",
                "qlib": str((QLIB_DIR / "us_stock").relative_to(PROJECT_ROOT)),
            })

    # ETFs
    if etf_dir.exists():
        for csv_file in sorted(etf_dir.glob("*.csv")):
            ticker = csv_file.stem
            features_path = FEATURES_DIR / f"{ticker}_features.parquet"
            all_tickers.append({
                "ticker": ticker,
                "raw_csv": str(csv_file.relative_to(PROJECT_ROOT)),
                "features": str(features_path.relative_to(PROJECT_ROOT)) if features_path.exists() else "",
                "qlib": str((QLIB_DIR / "us_etf").relative_to(PROJECT_ROOT)),
            })

    # Crypto
    if crypto_dir.exists():
        for csv_file in sorted(crypto_dir.glob("*.csv")):
            symbol = csv_file.stem
            features_path = FEATURES_DIR / f"{symbol}_features.parquet"
            all_tickers.append({
                "ticker": symbol,
                "raw_csv": str(csv_file.relative_to(PROJECT_ROOT)),
                "features": str(features_path.relative_to(PROJECT_ROOT)) if features_path.exists() else "",
                "qlib": str((QLIB_DIR / "crypto").relative_to(PROJECT_ROOT)),
            })

    for t in all_tickers:
        ws_paths.append([t["ticker"], t["raw_csv"], t["features"], t["qlib"]])

    for col_idx, col_name in enumerate(headers, 1):
        ws_paths.column_dimensions[get_column_letter(col_idx)].width = max(15, len(col_name) + 8)

    # ── Sheet_ReportLinks ──────────────────────────────────────────────
    ws_reports = wb.create_sheet("Sheet_ReportLinks")
    headers = ["Group/Ticker", "ReportType", "ReportPath", "Hyperlink"]
    ws_reports.append(headers)

    # Cap group reports
    for cap in ["large_cap", "mid_cap", "small_cap"]:
        report_path = REPORTS_DIR / f"{cap}.html"
        if report_path.exists():
            rel_path = str(report_path.relative_to(PROJECT_ROOT))
            formula = f'=HYPERLINK("{rel_path}", "Open Report")'
            ws_reports.append([cap, "group", rel_path, formula])

    # ETF reports
    etf_reports_dir = REPORTS_DIR / "etfs"
    if etf_reports_dir.exists():
        for html_file in sorted(etf_reports_dir.glob("*.html")):
            ticker = html_file.stem
            rel_path = str(html_file.relative_to(PROJECT_ROOT))
            formula = f'=HYPERLINK("{rel_path}", "Open Report")'
            ws_reports.append([ticker, "etf", rel_path, formula])

    # Crypto reports
    crypto_reports_dir = REPORTS_DIR / "crypto"
    if crypto_reports_dir.exists():
        for html_file in sorted(crypto_reports_dir.glob("*.html")):
            symbol = html_file.stem
            rel_path = str(html_file.relative_to(PROJECT_ROOT))
            formula = f'=HYPERLINK("{rel_path}", "Open Report")'
            ws_reports.append([symbol, "crypto", rel_path, formula])

    for col_idx, col_name in enumerate(headers, 1):
        ws_reports.column_dimensions[get_column_letter(col_idx)].width = max(15, len(col_name) + 4)

    # ── Sheet_RunLog ───────────────────────────────────────────────────
    ws_log = wb.create_sheet("Sheet_RunLog")
    headers = ["Script", "StartTime", "EndTime", "Duration(s)", "Status", "Errors"]
    ws_log.append(headers)

    if run_log:
        for entry in run_log:
            ws_log.append([
                entry.get("script", ""),
                entry.get("start_time", ""),
                entry.get("end_time", ""),
                entry.get("duration", ""),
                entry.get("status", ""),
                entry.get("errors", 0),
            ])

    # Add summary row
    ws_log.append(["", "", "", "", "", ""])
    ws_log.append(["Total Errors", "", "", "", "", error_count])
    ws_log.append(["Run Date", "", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "", "", ""])

    for col_idx, col_name in enumerate(headers, 1):
        ws_log.column_dimensions[get_column_letter(col_idx)].width = max(12, len(col_name) + 4)

    # Save
    output_path = OUTPUT_DIR / "TTP_Stock_ETF_Index.xlsx"
    wb.save(str(output_path))
    logger.info(f"Saved index Excel to {output_path}")


def main() -> None:
    generate_excel()
    logger.info("Done!")


if __name__ == "__main__":
    main()
