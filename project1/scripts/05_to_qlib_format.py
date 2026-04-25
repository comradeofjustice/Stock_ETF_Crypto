#!/usr/bin/env python3
"""
05_to_qlib_format.py
====================
Convert raw CSV data (stocks, ETFs, crypto) to Qlib binary format.
Uses Qlib's built-in dump_bin.py script for conversion.

Input:
    data/raw/stocks/*.csv
    data/raw/etfs/*.csv
    data/raw/crypto/*.csv
Output:
    data/qlib_data/us_stock/        # Qlib binary for stocks
    data/qlib_data/us_etf/          # Qlib binary for ETFs
    data/qlib_data/crypto/          # Qlib binary for crypto

Each dataset will have:
    instruments/all.txt             # List of instruments
    calendars/day.txt               # Trading calendar
"""

import logging
import subprocess
import sys
from pathlib import Path

import qlib
import yaml

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
QLIB_DIR = PROJECT_ROOT / "data" / "qlib_data"
CONFIG_DIR = PROJECT_ROOT / "config"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_csv_files(directory: Path) -> list:
    """Get all CSV files in a directory."""
    if not directory.exists():
        return []
    return sorted(directory.glob("*.csv"))


def convert_to_qlib(csv_dir: Path, qlib_output_dir: Path, dataset_name: str) -> bool:
    """Convert CSV files in csv_dir to Qlib binary format in qlib_output_dir."""
    csv_files = get_csv_files(csv_dir)
    if not csv_files:
        logger.warning(f"No CSV files found in {csv_dir}, skipping {dataset_name}")
        return False

    logger.info(f"Converting {len(csv_files)} CSV files from {csv_dir} to {qlib_output_dir}")

    # Build instrument list
    instruments_dir = qlib_output_dir / "instruments"
    instruments_dir.mkdir(parents=True, exist_ok=True)
    inst_file = instruments_dir / "all.txt"

    instrument_names = []
    for csv_file in csv_files:
        # Extract ticker name from filename (remove .csv extension)
        ticker = csv_file.stem
        instrument_names.append(ticker)

    # Write instruments list
    with open(inst_file, "w", encoding="utf-8") as f:
        for name in sorted(instrument_names):
            f.write(f"{name}\n")

    logger.info(f"Written {len(instrument_names)} instruments to {inst_file}")

    # Use Qlib's dump_bin.py for conversion
    # Find dump_bin.py in the qlib package
    qlib_package_dir = Path(qlib.__file__).parent
    dump_bin_script = qlib_package_dir / "scripts" / "dump_bin.py"

    if dump_bin_script.exists():
        logger.info(f"Using Qlib dump_bin.py at {dump_bin_script}")
        cmd = [
            sys.executable,
            str(dump_bin_script),
            "--csv_path", str(csv_dir),
            "--dump_path", str(qlib_output_dir),
            "--include_fields", "open,high,low,close,volume",
        ]
        logger.info(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                logger.error(f"dump_bin.py failed: {result.stderr}")
                # Fall back to manual conversion
                logger.info("Falling back to manual CSV-to-qlib conversion")
                return _manual_convert(csv_dir, qlib_output_dir, instrument_names, dataset_name)
            logger.info(f"dump_bin.py completed successfully for {dataset_name}")
            return True
        except subprocess.TimeoutExpired:
            logger.error(f"dump_bin.py timed out for {dataset_name}")
            return False
        except Exception as e:
            logger.error(f"dump_bin.py error: {e}, falling back to manual conversion")
            return _manual_convert(csv_dir, qlib_output_dir, instrument_names, dataset_name)
    else:
        logger.info(f"dump_bin.py not found at {dump_bin_script}, using manual conversion")
        return _manual_convert(csv_dir, qlib_output_dir, instrument_names, dataset_name)


def _manual_convert(csv_dir: Path, qlib_output_dir: Path, instrument_names: list, dataset_name: str = "") -> bool:
    """Manual conversion: create minimal Qlib-compatible structure."""
    import numpy as np
    import pandas as pd

    logger.info(f"Manual conversion for {len(instrument_names)} instruments")

    # Create calendars directory
    calendars_dir = qlib_output_dir / "calendars"
    calendars_dir.mkdir(parents=True, exist_ok=True)

    # Collect all dates across all files
    all_dates = set()
    for ticker in instrument_names:
        csv_file = csv_dir / f"{ticker}.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            all_dates.update(df.index.strftime("%Y-%m-%d").tolist())

    sorted_dates = sorted(all_dates)

    # Write calendar
    cal_file = calendars_dir / "day.txt"
    with open(cal_file, "w", encoding="utf-8") as f:
        for d in sorted_dates:
            f.write(f"{d}\n")

    logger.info(f"Written {len(sorted_dates)} calendar dates to {cal_file}")

    # Create feature directories
    features_dir = qlib_output_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    for field in ["open", "high", "low", "close", "volume"]:
        field_dir = features_dir / field
        field_dir.mkdir(parents=True, exist_ok=True)

        for ticker in instrument_names:
            csv_file = csv_dir / f"{ticker}.csv"
            if not csv_file.exists():
                continue

            try:
                df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
                if field in df.columns:
                    # Convert to numpy array, fill NaN with 0
                    series = df[field].astype(np.float32)
                    series = series.fillna(0.0)
                    # Save as binary
                    out_file = field_dir / f"{ticker}.bin"
                    series.values.tofile(out_file)
            except Exception as e:
                logger.warning(f"Failed to convert {ticker} {field}: {e}")

    logger.info(f"Manual conversion complete for {dataset_name}")
    return True


def verify_qlib_data(qlib_dir: Path, region: str = "us") -> bool:
    """Verify that Qlib can load the data."""
    try:
        qlib.init(provider_uri=str(qlib_dir), region=region)
        from qlib.data import D
        # Try to list instruments
        inst = D.list_instruments()
        logger.info(f"Verified {qlib_dir}: {len(inst)} instruments loaded")
        return True
    except Exception as e:
        logger.error(f"Failed to verify {qlib_dir}: {e}")
        return False


def main() -> None:
    logger.info("Starting CSV to Qlib format conversion")

    datasets = [
        ("stocks", "us_stock", "us"),
        ("etfs", "us_etf", "us"),
        ("crypto", "crypto", "generic"),
    ]

    for raw_subdir, qlib_subdir, region in datasets:
        csv_dir = RAW_DIR / raw_subdir
        qlib_output_dir = QLIB_DIR / qlib_subdir
        qlib_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {raw_subdir} -> {qlib_subdir}")
        logger.info(f"{'='*60}")

        success = convert_to_qlib(csv_dir, qlib_output_dir, qlib_subdir)

        if success:
            verify_qlib_data(qlib_output_dir, region)
        else:
            logger.error(f"Conversion failed for {raw_subdir}")

    logger.info("\nAll conversions complete!")


if __name__ == "__main__":
    main()
