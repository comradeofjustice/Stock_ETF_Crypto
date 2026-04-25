#!/usr/bin/env python3
"""
07_generate_reports.py
======================
Generate interactive HTML reports (Plotly) for volatility analysis.

Reports generated:
    reports/large_cap.html    - Large cap cross-sectional average
    reports/mid_cap.html      - Mid cap cross-sectional average
    reports/small_cap.html    - Small cap cross-sectional average
    reports/etfs/<TICKER>.html - Per-ETF report
    reports/crypto/<SYMBOL>.html - Per-crypto report

Each report contains:
    1. Histogram + fitted normal curve for range_pct (intraday range)
    2. Histogram + fitted normal curve for oc_pct (open-close spread)
    3. Line chart of ATR(5/14/64/128) over time

Cross-sectional average: for grouped reports (large/mid/small cap),
compute the mean of each metric across all stocks in the group for each date.

Input:
    data/features/<TICKER>_features.parquet
    config/universe.yaml
Output:
    reports/*.html
    reports/etfs/*.html
    reports/crypto/*.html
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import yaml
from scipy import stats
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
CONFIG_DIR = PROJECT_ROOT / "config"
REPORTS_DIR = PROJECT_ROOT / "reports"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
(REPORTS_DIR / "etfs").mkdir(parents=True, exist_ok=True)
(REPORTS_DIR / "crypto").mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── ATR periods ────────────────────────────────────────────────────────────
ATR_PERIODS = [5, 14, 64, 128]


def load_universe_config() -> Dict:
    """Load universe classification from YAML."""
    yaml_path = CONFIG_DIR / "universe.yaml"
    if not yaml_path.exists():
        logger.warning(f"Universe config not found at {yaml_path}")
        return {}
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_features(ticker: str) -> Optional[pd.DataFrame]:
    """Load features parquet for a ticker."""
    path = FEATURES_DIR / f"{ticker}_features.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        return df
    except Exception as e:
        logger.warning(f"Failed to load features for {ticker}: {e}")
        return None


def fit_and_plot_distribution(
    data: pd.Series,
    title: str,
    x_label: str,
) -> go.Figure:
    """Create histogram with fitted normal distribution curve and statistics."""
    fig = go.Figure()

    # Remove NaN
    clean_data = data.dropna()

    if len(clean_data) < 10:
        fig.add_annotation(text="Insufficient data", xref="paper", yref="paper", showarrow=False)
        fig.update_layout(title=title)
        return fig

    # Fit normal distribution
    mu, sigma = stats.norm.fit(clean_data)
    skew = stats.skew(clean_data)
    kurtosis = stats.kurtosis(clean_data)

    # Histogram
    fig.add_trace(go.Histogram(
        x=clean_data,
        nbinsx=100,
        name="Data",
        opacity=0.6,
        histnorm="probability density",
        marker_color="rgba(0, 100, 200, 0.5)",
    ))

    # Fitted normal curve
    x_range = np.linspace(clean_data.min(), clean_data.max(), 200)
    y_range = stats.norm.pdf(x_range, mu, sigma)
    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_range,
        mode="lines",
        name=f"Normal fit (μ={mu:.6f}, σ={sigma:.6f})",
        line=dict(color="red", width=2),
    ))

    # Statistics text
    stats_text = (
        f"μ = {mu:.6f}<br>"
        f"σ = {sigma:.6f}<br>"
        f"Skewness = {skew:.4f}<br>"
        f"Kurtosis = {kurtosis:.4f}<br>"
        f"N = {len(clean_data):,}"
    )

    fig.add_annotation(
        x=0.02, y=0.98,
        xref="paper", yref="paper",
        text=stats_text,
        showarrow=False,
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=11),
    )

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title="Density",
        template="plotly_white",
        height=500,
    )

    return fig


def plot_atr_timeseries(
    data: pd.DataFrame,
    title: str,
    is_cross_sectional: bool = False,
) -> go.Figure:
    """Create line chart of ATR(5/14/64/128) over time."""
    fig = go.Figure()

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    labels = ["ATR(5)", "ATR(14)", "ATR(64)", "ATR(128)"]

    for i, period in enumerate(ATR_PERIODS):
        col = f"atr_{period}"
        if col in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[col],
                mode="lines",
                name=labels[i],
                line=dict(color=colors[i], width=1.5),
            ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="ATR",
        template="plotly_white",
        height=400,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    return fig


def create_single_report(
    df: pd.DataFrame,
    ticker: str,
    output_path: Path,
    is_cross_sectional: bool = False,
) -> None:
    """Create a complete HTML report for a single ticker or group."""
    # Range pct distribution
    fig_range = fit_and_plot_distribution(
        df["range_pct"],
        title=f"{ticker} - Intraday Range Distribution",
        x_label="Range % = (High - Low) / Open",
    )

    # OC pct distribution
    fig_oc = fit_and_plot_distribution(
        df["oc_pct"],
        title=f"{ticker} - Open-Close Spread Distribution",
        x_label="OC % = (Close - Open) / Open",
    )

    # ATR timeseries
    fig_atr = plot_atr_timeseries(
        df,
        title=f"{ticker} - ATR Over Time",
        is_cross_sectional=is_cross_sectional,
    )

    # Combine into one HTML
    from plotly.subplots import make_subplots
    combined_fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            "Intraday Range Distribution",
            "Open-Close Spread Distribution",
            "ATR Over Time",
        ),
        vertical_spacing=0.08,
    )

    # Add range distribution
    for trace in fig_range.data:
        trace.update(xaxis="x1", yaxis="y1")
        combined_fig.add_trace(trace, row=1, col=1)

    # Add OC distribution
    for trace in fig_oc.data:
        trace.update(xaxis="x2", yaxis="y2")
        combined_fig.add_trace(trace, row=2, col=1)

    # Add ATR timeseries
    for trace in fig_atr.data:
        trace.update(xaxis="x3", yaxis="y3")
        combined_fig.add_trace(trace, row=3, col=1)

    combined_fig.update_layout(
        height=1200,
        title_text=f"<b>{ticker} - Volatility Analysis Report</b>",
        showlegend=False,
    )

    # Save HTML
    combined_fig.write_html(str(output_path), include_plotlyjs=True)
    logger.info(f"Saved report to {output_path}")


def compute_cross_sectional_avg(tickers: List[str]) -> Optional[pd.DataFrame]:
    """Compute cross-sectional average of features for a group of tickers."""
    all_dfs = []
    for ticker in tickers:
        df = load_features(ticker)
        if df is not None and not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        return None

    # Align all dataframes to common date index
    combined = pd.concat(all_dfs, axis=0, keys=tickers)
    if isinstance(combined.index, pd.MultiIndex):
        # Average across tickers for each date
        avg_df = combined.groupby(level=1).mean()
    else:
        avg_df = combined.mean(axis=1, level=1) if isinstance(combined.columns, pd.MultiIndex) else combined

    return avg_df


def generate_cap_reports(universe: Dict) -> None:
    """Generate reports for large/mid/small cap groups."""
    cap_map = {
        "large": "large_cap",
        "mid": "mid_cap",
        "small": "small_cap",
    }

    for cap_key, report_name in cap_map.items():
        tickers_info = universe.get(cap_key, [])
        if not tickers_info:
            logger.warning(f"No {cap_key} cap stocks found, skipping {report_name}")
            continue

        tickers = [t["ticker"] for t in tickers_info]
        logger.info(f"Generating {report_name} report for {len(tickers)} stocks")

        avg_df = compute_cross_sectional_avg(tickers)
        if avg_df is None or avg_df.empty:
            logger.warning(f"No features available for {report_name}")
            continue

        output_path = REPORTS_DIR / f"{report_name}.html"
        create_single_report(avg_df, f"{cap_key.capitalize()} Cap Average", output_path, is_cross_sectional=True)


def generate_etf_reports() -> None:
    """Generate individual reports for each ETF."""
    raw_dir = RAW_DIR / "etfs"
    if not raw_dir.exists():
        logger.warning("ETF raw data directory not found")
        return

    csv_files = list(raw_dir.glob("*.csv"))
    logger.info(f"Generating {len(csv_files)} ETF reports")

    for csv_file in tqdm(csv_files, desc="ETF reports"):
        ticker = csv_file.stem
        df = load_features(ticker)
        if df is None or df.empty:
            logger.warning(f"No features for {ticker}, skipping")
            continue

        output_path = REPORTS_DIR / "etfs" / f"{ticker}.html"
        create_single_report(df, ticker, output_path)


def generate_crypto_reports() -> None:
    """Generate individual reports for each cryptocurrency."""
    raw_dir = RAW_DIR / "crypto"
    if not raw_dir.exists():
        logger.warning("Crypto raw data directory not found")
        return

    csv_files = list(raw_dir.glob("*.csv"))
    logger.info(f"Generating {len(csv_files)} crypto reports")

    for csv_file in tqdm(csv_files, desc="Crypto reports"):
        symbol = csv_file.stem
        df = load_features(symbol)
        if df is None or df.empty:
            logger.warning(f"No features for {symbol}, skipping")
            continue

        output_path = REPORTS_DIR / "crypto" / f"{symbol}.html"
        create_single_report(df, symbol, output_path)


def main() -> None:
    logger.info("Starting report generation")

    # Load universe config
    universe = load_universe_config()

    # 1. Generate cap group reports
    logger.info("\n--- Generating cap group reports ---")
    generate_cap_reports(universe)

    # 2. Generate ETF reports
    logger.info("\n--- Generating ETF reports ---")
    generate_etf_reports()

    # 3. Generate crypto reports
    logger.info("\n--- Generating crypto reports ---")
    generate_crypto_reports()

    logger.info("\nAll reports generated!")


if __name__ == "__main__":
    main()
