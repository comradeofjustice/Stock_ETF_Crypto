#!/usr/bin/env python3
"""
06_compute_features.py
======================
Compute volatility features for each asset using pandas.
Features computed:
    - range_pct: (high - low) / open        (intraday range)
    - oc_pct: (close - open) / open          (open-close spread)
    - ATR(5), ATR(14), ATR(64), ATR(128):    Average True Range with Wilder smoothing

Input:
    data/raw/stocks/*.csv, etfs/*.csv, crypto/*.csv
    config/universe.yaml (for stock classification)
Output:
    data/features/<TICKER>_features.parquet
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
CONFIG_DIR = PROJECT_ROOT / "config"

FEATURES_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── ATR periods ────────────────────────────────────────────────────────────
ATR_PERIODS = [5, 14, 64, 128]


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Compute ATR(N) using Wilder smoothing (EMA).
    
    TR = max(high - low, abs(high - close_prev), abs(low - close_prev))
    ATR_N = EMA(TR, period)  # Wilder smoothing: alpha = 1/period
    """
    close_prev = df["close"].shift(1)
    
    tr = pd.DataFrame({
        "hl": df["high"] - df["low"],
        "hc": (df["high"] - close_prev).abs(),
        "lc": (df["low"] - close_prev).abs(),
    }).max(axis=1)
    
    # Wilder smoothing: EMA with alpha = 1/period
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    
    return atr


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all volatility features for a single asset."""
    result = pd.DataFrame(index=df.index)
    
    # Basic features
    result["range_pct"] = (df["high"] - df["low"]) / df["open"]
    result["oc_pct"] = (df["close"] - df["open"]) / df["open"]
    
    # ATR features
    for period in ATR_PERIODS:
        result[f"atr_{period}"] = compute_atr(df, period)
    
    return result


def process_csv_file(csv_path: Path, output_path: Path) -> bool:
    """Process a single CSV file and save features."""
    try:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        
        # Ensure required columns exist
        required = ["open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in required):
            logger.warning(f"[{csv_path.name}] Missing required columns")
            return False
        
        # Compute features
        features = compute_features(df)
        
        # Remove rows with NaN (first few rows due to ATR calculation)
        features = features.dropna()
        
        if features.empty:
            logger.warning(f"[{csv_path.name}] No features after dropna")
            return False
        
        # Save
        features.to_parquet(output_path)
        logger.info(f"[{csv_path.stem}] Saved {len(features)} rows to {output_path.name}")
        return True
        
    except Exception as e:
        logger.error(f"[{csv_path.name}] Failed: {e}")
        return False


def main() -> None:
    logger.info("Starting feature computation")

    datasets = [
        ("stocks", "us_stock"),
        ("etfs", "us_etf"),
        ("crypto", "crypto"),
    ]

    total_processed = 0
    total_failed = 0

    for raw_subdir, qlib_subdir in datasets:
        csv_dir = RAW_DIR / raw_subdir
        if not csv_dir.exists():
            logger.warning(f"{csv_dir} does not exist, skipping")
            continue

        csv_files = sorted(csv_dir.glob("*.csv"))
        if not csv_files:
            logger.warning(f"No CSV files in {csv_dir}")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {raw_subdir}: {len(csv_files)} files")
        logger.info(f"{'='*60}")

        for csv_file in tqdm(csv_files, desc=f"{raw_subdir} features"):
            ticker = csv_file.stem
            output_path = FEATURES_DIR / f"{ticker}_features.parquet"
            
            ok = process_csv_file(csv_file, output_path)
            if ok:
                total_processed += 1
            else:
                total_failed += 1

    logger.info(f"\nFeature computation complete: {total_processed} succeeded, {total_failed} failed")


if __name__ == "__main__":
    main()
