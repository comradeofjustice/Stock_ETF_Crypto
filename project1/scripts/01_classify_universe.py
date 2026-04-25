#!/usr/bin/env python3
"""
01_classify_universe.py
=======================
Read stock tickers from stocks_full_list.txt, classify them into
Large / Mid / Small / unclassified caps.

Strategy: Use known stock lists + quick yfinance batch fetch.
No heavy API usage - classification is primarily for report grouping.

Input:
    /root/autodl-tmp/CSY/Stock_analysis/list/stocks_full_list.txt
Output:
    config/universe.yaml          # YAML grouping by cap category
    output/universe_classification.csv  # CSV with ticker, name, category, marketCap
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml
import yfinance as yf
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
STOCKS_LIST = Path("/root/autodl-tmp/CSY/Stock_analysis/list/stocks_full_list.txt")
CONFIG_DIR = PROJECT_ROOT / "config"
OUTPUT_DIR = PROJECT_ROOT / "output"

CONFIG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Market cap thresholds (USD) ────────────────────────────────────────────
LARGE_CAP = 10_000_000_000      # ≥ $10B
MID_CAP = 2_000_000_000         # ≥ $2B
SMALL_CAP = 300_000_000         # ≥ $0.3B

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


def parse_ticker(ticker: str) -> Tuple[str, str]:
    """Parse ticker into (base_symbol, exchange_type)."""
    if ".OQ" in ticker or ".N" in ticker:
        base = ticker.split(".")[0]
        return base, "us"
    else:
        base = ticker.split(".")[0] if "." in ticker else ticker
        return base, "other"


# ── Known large-cap stocks (S&P 500 mega/large cap) ───────────────────────
KNOWN_LARGE_CAP = {
    # US Mega Cap
    "AAPL", "MSFT", "GOOG", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "BRK-B",
    # US Large Cap
    "JPM", "V", "JNJ", "WMT", "PG", "MA", "UNH", "HD", "DIS", "BAC", "XOM",
    "VZ", "ADBE", "CRM", "CSCO", "PFE", "KO", "INTC", "NKE", "ABT", "MRK",
    "TMO", "AVGO", "COST", "MCD", "ACN", "ABNB", "NFLX", "LIN", "AMD",
    "ORCL", "PYPL", "ISRG", "SBUX", "TXN", "MDT", "QCOM", "AMGN", "UPS",
    "LOW", "CVS", "GE", "LLY", "BLK", "SPGI", "SCHW", "CAT", "GS", "MS",
    "C", "BA", "MMM", "HON", "UNP", "DE", "RTX", "NEE", "IBM", "SYK",
    "DHR", "BKNG", "MDLZ", "ADP", "CL", "GIS", "MO", "KHC", "PEP",
    "TMUS", "CMCSA", "CHTR", "LRCX", "KLAC", "MU", "INTU", "WDAY",
    "ZS", "CRWD", "ZM", "UBER", "BABA", "PDD", "JD", "BIDU", "NTES",
    # European Blue Chips
    "SIE", "ASML", "NESN", "NOVN", "ROG", "LVMH", "SAN", "TTE", "AIR",
    "EL", "MC", "OR", "RI", "PUM", "VOW", "BMW", "DAI", "BAS", "BAYN",
    "SAP", "HEN", "ALV", "MRK", "LIN", "BEI", "ADS", "FME", "CON",
    "RWE", "HNR", "MBG", "VOW3", "DTE", "DBK", "MUV2", "FRE", "IFX",
    "1COV", "BNR", "SU", "CS", "SGEN", "GIB", "AI", "PHIA", "UGI",
    "BNPP", "ACA", "SGO", "KER", "CAP", "PUB", "WLN", "RMS", "STLA",
    "ENI", "TIT", "G", "ISP", "PRY", "MONC", "UCG", "SPM", "TEN",
    "ENEL", "FER", "ANA", "BBVA", "ITX", "TEF", "IBE", "ACS", "REP",
    "NDA", "FRES", "BDEV", "CNA", "GSK", "AZN", "DGE", "HLMA", "SMT",
    "ULVR", "GLEN", "BARC", "HSBA", "LLOY", "PRU", "TW", "BP", "SHEL",
    "RDSA", "NXT", "ADM", "CPG", "WEIR", "WPP", "REL", "SDR", "SVS",
    "FLTR", "MNDI", "EXPN", "AHT", "BLND", "PSN", "CCL", "IAG", "BT",
    "EZJ", "SPX", "SSE", "HAL", "MRO", "UU", "WHT", "WPG", "EONGn",
    "FREG", "RRTL", "CBKG", "WOS", "ZAL", "GLE", "WRT1V", "RIGHT",
    "AUTO", "ENT", "BARL", "IHG", "MKS", "WTB", "WCR", "BPI", "JT2",
    "CTEC", "SN", "PFG", "HLE", "PSH", "SRT", "GLD",
}

# ── Known mid-cap hints ────────────────────────────────────────────────────
KNOWN_MID_CAP = {
    "ZTS", "TER", "ETN", "ROL", "FFIV", "DXCM", "REGN", "ILMN",
    "EXC", "ES", "D", "AEP", "SRE", "TGT", "CTLT", "CCEP", "CPRT",
    "EG", "CINF", "ARE", "WBC", "O", "PSA", "AMT", "PLD", "EQIX",
    "CCI", "DLR", "MPC", "VLO", "PSX", "OXY", "SLB", "HAL", "MRO",
    "FANG", "DVN", "EOG", "HES", "APA", "MAR", "HIL", "ROP",
    "FAST", "ZBRA", "VRSN", "ESTC", "CRWD", "SNOW", "DDOG", "NET",
    "SHOP", "SQ", "COIN", "ROKU", "SPOT", "PTON", "LYFT", "RIVN",
    "NIO", "LI", "XPEV", "TCEHY", "KWEB", "CQQQ", "MCHI", "FXI",
    "DOGE", "CGC", "JCI", "RSG", "EWHS", "VICI", "WY", "O",
}


def classify(cap: Optional[float]) -> str:
    """Classify market cap into category."""
    if cap is None or pd.isna(cap):
        return "unclassified"
    cap_val = float(cap)
    if cap_val >= LARGE_CAP:
        return "large"
    if cap_val >= MID_CAP:
        return "mid"
    if cap_val >= SMALL_CAP:
        return "small"
    return "unclassified"


def main() -> None:
    logger.info("Reading stock list from %s", STOCKS_LIST)
    tickers = read_tickers(STOCKS_LIST)
    logger.info("Found %d tickers", len(tickers))

    results: List[Dict] = []

    # Quick yfinance batch fetch for US stocks (non-blocking)
    logger.info("Quick yfinance batch fetch for market caps...")
    us_tickers = [t for t in tickers if ".OQ" in t or ".N" in t]
    other_tickers = [t for t in tickers if t not in us_tickers]

    # Fetch in small batches with timeout
    yf_cache: Dict[str, Tuple[Optional[float], str]] = {}
    
    for ticker in tqdm(us_tickers[:100], desc="Quick yfinance fetch (first 100)"):
        try:
            t = yf.Ticker(ticker)
            info = t.fast_info
            cap = info.get("market_cap")
            name = info.get("name", "")
            base, _ = parse_ticker(ticker)
            yf_cache[base] = (cap, name)
        except Exception:
            pass
        time.sleep(0.3)

    # Classify all tickers
    for ticker in tickers:
        base, ex_type = parse_ticker(ticker)
        
        # Check yfinance cache first
        if base in yf_cache and yf_cache[base][0] is not None:
            cap, name = yf_cache[base]
            cat = classify(cap)
        elif base in KNOWN_LARGE_CAP:
            cap = None
            name = base
            cat = "large"
        elif base in KNOWN_MID_CAP:
            cap = None
            name = base
            cat = "mid"
        else:
            cap = None
            name = base
            cat = "unclassified"

        results.append({
            "ticker": ticker,
            "name": name,
            "category": cat,
            "marketCap": cap,
        })

    df = pd.DataFrame(results)

    # ── Write CSV ──────────────────────────────────────────────────────
    csv_path = OUTPUT_DIR / "universe_classification.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    logger.info("Wrote %s", csv_path)

    # ── Write YAML ─────────────────────────────────────────────────────
    universe: Dict[str, List[Dict]] = {}
    for cat in ["large", "mid", "small", "unclassified"]:
        subset = df[df["category"] == cat]
        universe[cat] = [
            {"ticker": row["ticker"], "name": row["name"], "marketCap": row["marketCap"]}
            for _, row in subset.iterrows()
        ]

    yaml_path = CONFIG_DIR / "universe.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(universe, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    logger.info("Wrote %s", yaml_path)

    # ── Summary ────────────────────────────────────────────────────────
    for cat in ["large", "mid", "small", "unclassified"]:
        count = len(universe[cat])
        logger.info("  %s cap: %d tickers", cat, count)


if __name__ == "__main__":
    main()
