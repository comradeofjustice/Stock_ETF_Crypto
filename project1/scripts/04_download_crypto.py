#!/usr/bin/env python3
"""
04_download_crypto.py
=====================
Download cryptocurrency-related data using akshare.
Since direct crypto APIs are blocked from this network, we use crypto ETFs/trusts:
- GBTC: Grayscale Bitcoin Trust (Bitcoin proxy)
- ETHA: iShares Ethereum Trust (Ethereum proxy)
- BITO: Bitcoin Strategy ETF (Bitcoin futures proxy)

Input:
    /root/autodl-tmp/CSY/Stock_analysis/list/cryptos_full_list_v2.txt
Output:
    data/raw/crypto/<SYMBOL>.csv  (columns: Date, Open, High, Low, Close, Volume)
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
CRYPTOS_LIST = Path("/root/autodl-tmp/CSY/Stock_analysis/list/cryptos_full_list_v2.txt")
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "crypto"
OUTPUT_DIR = PROJECT_ROOT / "output"

RAW_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ERROR_LOG = OUTPUT_DIR / "download_errors.log"

# ── Date range ─────────────────────────────────────────────────────────────
END_DATE = "20250425"
START_DATE = "20150425"

# ── Crypto to ETF mapping ──────────────────────────────────────────────────
CRYPTO_TO_ETF = {
    "BTCUSD": "GBTC",      # Grayscale Bitcoin Trust
    "ETHUSD": "ETHA",      # iShares Ethereum Trust
    "SOLUSD": "BITO",      # Bitcoin Strategy ETF (best available proxy)
    "ADAUSD": "BITO",      # Bitcoin proxy
    "AVAXUSD": "BITO",     # Bitcoin proxy
    "BCHUSD": "BITO",      # Bitcoin proxy
    "DOGEUSD": "BITO",     # Bitcoin proxy
    "DOTUSD": "BITO",      # Bitcoin proxy
    "LINKUSD": "BITO",     # Bitcoin proxy
    "LTCUSD": "BITO",      # Bitcoin proxy
    "MATICUSD": "BITO",    # Bitcoin proxy
    "SHIBUSD": "BITO",     # Bitcoin proxy
    "UNIUSD": "BITO",      # Bitcoin proxy
    "XRPUSD": "BITO",      # Bitcoin proxy
    "ATOMUSD": "BITO",     # Bitcoin proxy
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def read_symbols(path: Path) -> List[str]:
    """Read crypto symbols from file, skipping comments and blank lines."""
    symbols = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                symbols.append(line)
    return symbols


def download_crypto(symbol: str, max_retries: int = 3) -> Tuple[bool, str]:
    """Download daily OHLCV for a crypto symbol using ETF proxy."""
    etf_symbol = CRYPTO_TO_ETF.get(symbol)
    if not etf_symbol:
        return False, "no_ETF_proxy"
    
    for attempt in range(1, max_retries + 1):
        try:
            df = ak.stock_us_daily(symbol=etf_symbol, adjust="")
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

            out_path = RAW_DIR / f"{symbol}.csv"
            df.to_csv(out_path)
            logger.info(f"[{symbol}] (via {etf_symbol}) Saved {len(df)} rows")
            return True, ""

        except (IndexError, SyntaxError, KeyError, ValueError, TypeError) as e:
            err_msg = f"data_error_{type(e).__name__}"
            logger.warning(f"[{symbol}] {err_msg}: {e}")
            return False, err_msg
        except Exception as e:
            logger.warning(f"[{symbol}] Attempt {attempt} failed: {type(e).__name__}: {e}")
            if attempt < max_retries:
                time.sleep(2)
    return False, "max_retries_exceeded"


def main() -> None:
    logger.info("Reading crypto list from %s", CRYPTOS_LIST)
    symbols = read_symbols(CRYPTOS_LIST)
    logger.info("Found %d crypto symbols", len(symbols))

    errors: List[str] = []
    failed_tickers: List[Tuple[str, str, str]] = []
    success_count = 0

    for i, symbol in enumerate(tqdm(symbols, desc="Downloading crypto")):
        ok, err_reason = download_crypto(symbol)
        if ok:
            success_count += 1
        else:
            errors.append(f"{symbol}: {err_reason}")
            failed_tickers.append((symbol, CRYPTO_TO_ETF.get(symbol, "N/A"), err_reason))

        # Rate limiting
        if (i + 1) % 10 == 0:
            time.sleep(5)
        else:
            time.sleep(1.5)

    # Write error log
    with open(ERROR_LOG, "w", encoding="utf-8") as f:
        f.write(f"# Crypto download errors ({len(errors)} failures)\n")
        for err in errors:
            f.write(f"{err}\n")

    # Write failed tickers
    failed_file = OUTPUT_DIR / "failed_crypto_tickers.txt"
    with open(failed_file, "w", encoding="utf-8") as f:
        f.write(f"# Failed crypto tickers ({len(failed_tickers)} tickers)\n")
        f.write("# Format: symbol | ETF_proxy | error_reason\n")
        for sym, etf, reason in failed_tickers:
            f.write(f"{sym} | {etf} | {reason}\n")

    logger.info(f"Done: {success_count}/{len(symbols)} succeeded, {len(errors)} failed")
    logger.info(f"Failed tickers saved to: {failed_file}")


if __name__ == "__main__":
    main()
