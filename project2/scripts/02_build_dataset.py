#!/usr/bin/env python3
"""
02_build_dataset.py
===================
Merge features (from project1) + rolling RV + forward RV labels for every ticker.
Apply time-series split and save a single dataset.parquet plus split index files.

Output columns:
  ticker, date,
  [features]: range_pct, oc_pct, atr_5, atr_14, atr_64, atr_128,
              rv_10, rv_20, rv_30, rv_40, rv_50, rv_60
  [targets]:  rv_fwd_10, rv_fwd_20, rv_fwd_30, rv_fwd_40, rv_fwd_50, rv_fwd_60
  split:      train | val | test
"""

import logging
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((PROJECT_ROOT / "config/experiment.yaml").read_text())

FEATURES_DIR  = PROJECT_ROOT / CONFIG["data"]["features_dir"]
LABELS_DIR    = PROJECT_ROOT / CONFIG["data"]["labels_dir"]
DATASET_PATH  = PROJECT_ROOT / CONFIG["data"]["dataset_path"]

TRAIN_END = pd.Timestamp(CONFIG["split"]["train_end"])
VAL_END   = pd.Timestamp(CONFIG["split"]["val_end"])

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def assign_split(date: pd.DatetimeIndex) -> pd.Series:
    splits = pd.Series("test", index=date)
    splits[date <= TRAIN_END] = "train"
    splits[(date > TRAIN_END) & (date <= VAL_END)] = "val"
    return splits


def main():
    feature_files = sorted(FEATURES_DIR.glob("*_features.parquet"))
    log.info(f"Found {len(feature_files)} feature files")

    chunks = []
    for feat_path in tqdm(feature_files, desc="merging"):
        ticker = feat_path.stem.replace("_features", "")
        label_path = LABELS_DIR / f"{ticker}_labels.parquet"

        if not label_path.exists():
            log.warning(f"[{ticker}] no label file, skip")
            continue

        feat  = pd.read_parquet(feat_path)
        label = pd.read_parquet(label_path)

        merged = feat.join(label, how="inner")
        merged = merged.dropna()
        if merged.empty:
            continue

        merged.index.name = "date"
        merged["ticker"] = ticker
        merged["split"]  = assign_split(merged.index)

        chunks.append(merged.reset_index())

    if not chunks:
        log.error("No data merged — exiting")
        return

    dataset = pd.concat(chunks, ignore_index=True)
    dataset = dataset.sort_values(["ticker", "date"]).reset_index(drop=True)

    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(DATASET_PATH, index=False)

    # Summary
    for split, grp in dataset.groupby("split"):
        log.info(f"  {split}: {len(grp):,} rows, {grp['ticker'].nunique()} tickers, "
                 f"{grp['date'].min().date()} ~ {grp['date'].max().date()}")

    log.info(f"Dataset saved → {DATASET_PATH}")


if __name__ == "__main__":
    main()
