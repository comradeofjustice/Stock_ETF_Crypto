#!/usr/bin/env python3
"""
03_download_etfs.py
===================
Download 10 years of daily OHLCV data for all ETFs using akshare.
Saves CSV files to data/raw/etfs/<TICKER>.csv.

Input:
    /root/autodl-tmp/CSY/Stock_analysis/list/etfs_full_list_v2.txt
Output:
    data/raw/etfs/<TICKER>.csv  (columns: Date, Open, High, Low, Close, Volume)
    output/download_errors.log
"""

import time
import logging
from pathlib import Path
from typing import List, Tuple

import akshare as ak
import pandas as pd
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ETFS_LIST = Path("/root/autodl-tmp/CSY/Stock_analysis/list/etfs_full_list_v2.txt")
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "etfs"
OUTPUT_DIR = PROJECT_ROOT / "output"

RAW_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ERROR_LOG = OUTPUT_DIR / "download_errors.log"

# ── Date range ─────────────────────────────────────────────────────────────
END_DATE = "20250425"
START_DATE = "20150425"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def read_tickers(path: Path) -> List[str]:
    """Read tickers from file, skipping comments and blank lines."""
    tickers = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                tickers.append(line)
    return tickers


def map_etf_ticker(ticker: str) -> Tuple[str, str]:
    """Map ETF ticker to akshare symbol.
    
    akshare stock_us_daily works for US ETFs too.
    - SPY.ETF -> SPY
    - QQQ.OQ -> QQQ
    - BITO.ETF -> BITO
    """
    # Remove suffix
    for suffix in [".ETF", ".OQ", ".N"]:
        if ticker.endswith(suffix):
            base = ticker[:-len(suffix)]
            return base, "us"
    return ticker, "unknown"


def download_etf(ak_symbol: str, original_ticker: str, max_retries: int = 3) -> Tuple[bool, str]:
    """Download daily OHLCV for a single ETF with retry logic.
    
    Returns:
        (success, error_reason) - error_reason is empty string on success
    """
    for attempt in range(1, max_retries + 1):
        try:
            df = ak.stock_us_daily(symbol=ak_symbol, adjust="")
            if df is None or df.empty:
                return False, "empty_data"

            # Filter by date range
            df["date"] = pd.to_datetime(df["date"])
            df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)]

            if df.empty:
                return False, "no_data_in_range"

            # Set date as index and format
            df = df.set_index("date")
            df.index.name = "Date"
            df = df[["open", "high", "low", "close", "volume"]]
            df = df.sort_index()

            out_path = RAW_DIR / f"{original_ticker}.csv"
            df.to_csv(out_path)
            logger.info(f"[{original_ticker}] Saved {len(df)} rows to {out_path.name}")
            return True, ""

        except (IndexError, SyntaxError, KeyError, ValueError, TypeError) as e:
            err_msg = f"data_error_{type(e).__name__}"
            logger.warning(f"[{original_ticker}] {err_msg}: {e}")
            return False, err_msg
        except Exception as e:
            logger.warning(f"[{original_ticker}] Attempt {attempt} failed: {type(e).__name__}: {e}")
            if attempt < max_retries:
                time.sleep(2)
    return False, "max_retries_exceeded"


def main() -> None:
    logger.info("Reading ETF list from %s", ETFS_LIST)
    tickers = read_tickers(ETFS_LIST)
    logger.info("Found %d ETF tickers", len(tickers))

    errors: List[str] = []
    failed_tickers: List[Tuple[str, str, str]] = []  # (original, ak_symbol, error_reason)
    success_count = 0

    for i, ticker in enumerate(tqdm(tickers, desc="Downloading ETFs")):
        ak_symbol, ex_type = map_etf_ticker(ticker)
        ok, err_reason = download_etf(ak_symbol, ticker)
        if ok:
            success_count += 1
        else:
            errors.append(f"{ticker} (ak={ak_symbol}): {err_reason}")
            failed_tickers.append((ticker, ak_symbol, err_reason))

        # Rate limiting
        if (i + 1) % 10 == 0:
            time.sleep(5)
        else:
            time.sleep(1.5)

    # Write error log
    with open(ERROR_LOG, "w", encoding="utf-8") as f:
        f.write(f"# ETF download errors ({len(errors)} failures)\n")
        for err in errors:
            f.write(f"{err}\n")

    # Write failed tickers for later analysis
    failed_file = OUTPUT_DIR / "failed_etf_tickers.txt"
    with open(failed_file, "w", encoding="utf-8") as f:
        f.write(f"# Failed ETF tickers for later analysis ({len(failed_tickers)} tickers)\n")
        f.write("# Format: original_ticker | akshare_symbol | error_reason\n")
        for orig, ak_sym, reason in failed_tickers:
            f.write(f"{orig} | {ak_sym} | {reason}\n")

    logger.info(f"Done: {success_count}/{len(tickers)} succeeded, {len(errors)} failed")
    logger.info(f"Failed tickers saved to: {failed_file}")


if __name__ == "__main__":
    main()
