#!/usr/bin/env python3
"""
02_download_stocks.py
=====================
Download 10 years of daily OHLCV data for all stocks in stocks_full_list.txt
using akshare. The script maps exchange-suffixed tickers to akshare-compatible symbols.

Input:
    /root/autodl-tmp/CSY/Stock_analysis/list/stocks_full_list.txt
Output:
    data/raw/stocks/<TICKER>.csv  (columns: Date, Open, High, Low, Close, Volume)
    output/download_errors.log
"""

import logging
import time
from pathlib import Path
from typing import List, Tuple

import akshare as ak
import pandas as pd
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
STOCKS_LIST = Path("/root/autodl-tmp/CSY/Stock_analysis/list/stocks_full_list.txt")
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "stocks"
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


def map_to_akshare_symbol(original: str) -> Tuple[str, str]:
    """Map exchange-suffixed ticker to akshare US stock symbol.
    
    akshare stock_us_daily uses plain ticker symbols for US stocks:
        - AAPL.OQ -> AAPL
        - JPM.N -> JPM
        - SIE.DE -> SIE (if available on US)
    
    For non-US stocks, akshare has limited support. We'll try US equivalent or skip.
    
    Returns: (akshare_symbol, exchange_type)
    """
    # Exchange suffix mapping
    exchange_map = {
        ".OQ": ("", "us"),          # NASDAQ -> plain symbol
        ".N": ("", "us"),           # NYSE -> plain symbol
        ".L": (None, "london"),     # London -> not directly supported
        ".DE": (None, "xetra"),     # XETRA -> not directly supported
        ".PA": (None, "paris"),     # Paris -> not directly supported
        ".AS": (None, "amsterdam"), # Amsterdam -> not directly supported
        ".BR": (None, "brussels"),  # Brussels -> not directly supported
        ".MC": (None, "madrid"),    # Madrid -> not directly supported
        ".MI": (None, "milan"),     # Milan -> not directly supported
        ".HE": (None, "helsinki"),  # Helsinki -> not directly supported
    }

    for suffix, (ak_symbol, ex_type) in exchange_map.items():
        if original.endswith(suffix):
            base = original[:-len(suffix)]
            if ex_type == "us":
                return base, ex_type
            else:
                # For non-US, try the base symbol anyway (might work for ADRs)
                return base, ex_type

    return original, "unknown"


def download_stock(ak_symbol: str, original_ticker: str, ex_type: str, max_retries: int = 3) -> Tuple[bool, str]:
    """Download daily OHLCV for a single ticker with retry logic.
    
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
            
            # Sort by date
            df = df.sort_index()

            out_path = RAW_DIR / f"{original_ticker}.csv"
            df.to_csv(out_path)
            logger.info(f"[{original_ticker}] ({ak_symbol}) Saved {len(df)} rows")
            return True, ""

        except (IndexError, SyntaxError, KeyError, ValueError, TypeError) as e:
            # These errors indicate the ticker doesn't exist or has invalid data
            err_msg = f"data_error_{type(e).__name__}"
            logger.warning(f"[{original_ticker}] {err_msg}: {e}")
            return False, err_msg
        except Exception as e:
            err_str = str(e).lower()
            if "rate" in err_str or "429" in err_str:
                wait = min(10 ** attempt, 300)
                logger.warning(f"[{original_ticker}] Rate limited, waiting {wait}s (attempt {attempt}/{max_retries})")
                time.sleep(wait)
            else:
                logger.warning(f"[{original_ticker}] Attempt {attempt} failed: {type(e).__name__}: {e}")
                if attempt < max_retries:
                    time.sleep(2)
    return False, "max_retries_exceeded"


def main() -> None:
    logger.info("Reading stock list from %s", STOCKS_LIST)
    tickers = read_tickers(STOCKS_LIST)
    logger.info("Found %d stock tickers", len(tickers))

    # Map all tickers to akshare format
    logger.info("Mapping tickers to akshare format...")
    ticker_mapping = []
    for t in tickers:
        ak_symbol, ex_type = map_to_akshare_symbol(t)
        ticker_mapping.append((t, ak_symbol, ex_type))

    # Show mapping examples
    for orig, ak_sym, ex in ticker_mapping[:10]:
        logger.info(f"  {orig:20s} -> {ak_sym:15s} ({ex})")

    errors: List[str] = []
    failed_tickers: List[Tuple[str, str, str]] = []  # (original, ak_symbol, error_reason)
    success_count = 0
    skipped_count = 0

    for i, (original, ak_symbol, ex_type) in enumerate(tqdm(ticker_mapping, desc="Downloading stocks")):
        ok, err_reason = download_stock(ak_symbol, original, ex_type)
        if ok:
            success_count += 1
        else:
            errors.append(f"{original} (ak={ak_symbol}, ex={ex_type}): {err_reason}")
            failed_tickers.append((original, ak_symbol, err_reason))

        # Rate limiting
        if (i + 1) % 20 == 0:
            logger.info(f"Progress: {i+1}/{len(tickers)}, success={success_count}, failed={len(errors)}")
            time.sleep(5)
        else:
            time.sleep(1.5)

    # Write error log
    with open(ERROR_LOG, "w", encoding="utf-8") as f:
        f.write(f"# Stock download errors ({len(errors)} failures)\n")
        for err in errors:
            f.write(f"{err}\n")

    # Write failed tickers for later analysis
    failed_file = OUTPUT_DIR / "failed_stock_tickers.txt"
    with open(failed_file, "w", encoding="utf-8") as f:
        f.write(f"# Failed stock tickers for later analysis ({len(failed_tickers)} tickers)\n")
        f.write("# Format: original_ticker | akshare_symbol | error_reason\n")
        for orig, ak_sym, reason in failed_tickers:
            f.write(f"{orig} | {ak_sym} | {reason}\n")

    logger.info(f"Done: {success_count}/{len(tickers)} succeeded, {len(errors)} failed")
    logger.info(f"Error log: {ERROR_LOG}")
    logger.info(f"Failed tickers saved to: {failed_file}")


if __name__ == "__main__":
    main()
