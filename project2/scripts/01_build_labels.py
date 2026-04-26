#!/usr/bin/env python3
"""
01_build_labels.py
==================
For each ticker CSV in data/raw/{stocks,etfs,crypto}:
  - Compute rolling realized volatility features: rv_10, rv_20, ..., rv_60
  - Compute forward realized volatility labels:  rv_fwd_10, rv_fwd_20, ..., rv_fwd_60

RV_N(t)     = std(log_ret[t-N+1 : t]) * sqrt(252)   <- feature (past N days)
RV_fwd_N(t) = std(log_ret[t+1 : t+N]) * sqrt(252)   <- label   (future N days)

Output: data/labels/<TICKER>_labels.parquet
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((PROJECT_ROOT / "config/experiment.yaml").read_text())

RAW_DIR    = PROJECT_ROOT / CONFIG["data"]["raw_dir"]
LABELS_DIR = PROJECT_ROOT / CONFIG["data"]["labels_dir"]
LABELS_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS   = CONFIG["horizons"]     # forward targets
RV_WINDOWS = CONFIG["rv_windows"]   # rolling features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def compute_labels(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"].astype(float)
    log_ret = np.log(close / close.shift(1))

    result = pd.DataFrame(index=df.index)

    # Rolling RV features (past N days)
    for n in RV_WINDOWS:
        result[f"rv_{n}"] = log_ret.rolling(n).std() * np.sqrt(252)

    # Forward RV labels (future N days)
    # rv_fwd_N(t) = std(log_ret[t+1] ... log_ret[t+N]) * sqrt(252)
    for n in HORIZONS:
        rv_fwd = log_ret.rolling(n).std().shift(-n) * np.sqrt(252)
        result[f"rv_fwd_{n}"] = rv_fwd
        result[f"log_rv_fwd_{n}"] = np.log(rv_fwd.clip(lower=1e-8))

    return result


def process_file(csv_path: Path) -> bool:
    try:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        if "close" not in df.columns:
            log.warning(f"[{csv_path.name}] missing 'close' column, skip")
            return False

        labels = compute_labels(df)
        labels = labels.dropna()
        if labels.empty:
            log.warning(f"[{csv_path.name}] empty after dropna, skip")
            return False

        out_path = LABELS_DIR / f"{csv_path.stem}_labels.parquet"
        labels.to_parquet(out_path)
        return True

    except Exception as e:
        log.error(f"[{csv_path.name}] {e}")
        return False


def main():
    all_csvs = list(RAW_DIR.rglob("*.csv"))
    log.info(f"Found {len(all_csvs)} CSV files")

    ok = sum(process_file(p) for p in tqdm(all_csvs, desc="building labels"))
    log.info(f"Done: {ok}/{len(all_csvs)} succeeded")


if __name__ == "__main__":
    main()
