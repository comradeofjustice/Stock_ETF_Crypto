#!/usr/bin/env python3
"""
plot_predictions.py
===================
Plot predicted vs actual RV curves on the test set for HAR-RV and LightGBM.
One figure per horizon (10/20/30/40/50/60d), each showing:
  - Actual RV
  - HAR-RV prediction
  - LightGBM prediction

Also plots a single ticker deep-dive (AAPL by default) across all horizons.

Output: results/plots/pred_vs_actual_<horizon>d.png
        results/plots/ticker_deepdive.png
"""

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import yaml
import lightgbm as lgb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((PROJECT_ROOT / "config/experiment.yaml").read_text())

DATASET_PATH = PROJECT_ROOT / CONFIG["data"]["dataset_path"]
HAR_DIR      = PROJECT_ROOT / "models/har_rv"
LGBM_DIR     = PROJECT_ROOT / "models/lgbm"
PLOTS_DIR    = PROJECT_ROOT / "results/plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = CONFIG["horizons"]
RV_FEATURES = [f"rv_{n}" for n in CONFIG["rv_windows"]]
ALL_FEATURES = (
    ["range_pct", "oc_pct", "atr_5", "atr_14", "atr_64", "atr_128"]
    + RV_FEATURES
)

DEEPDIVE_TICKER = "AAPL.OQ"


def load_models():
    har_models, lgbm_models = {}, {}
    for h in HORIZONS:
        with open(HAR_DIR / f"har_rv_{h}.pkl", "rb") as f:
            har_models[h] = pickle.load(f)
        with open(LGBM_DIR / f"lgbm_{h}.pkl", "rb") as f:
            lgbm_models[h] = pickle.load(f)
    return har_models, lgbm_models


def predict_all(test_df, har_models, lgbm_models):
    preds = {}
    for h in HORIZONS:
        X_har  = test_df[RV_FEATURES].values
        X_lgbm = test_df[ALL_FEATURES].values
        preds[h] = {
            "actual":   test_df[f"rv_fwd_{h}"].values,
            "har_rv":   har_models[h].predict(X_har),
            "lgbm":     lgbm_models[h].predict(X_lgbm),
            "date":     pd.to_datetime(test_df["date"].values),
        }
    return preds


def plot_aggregate(preds):
    """One subplot per horizon, aggregate mean across all tickers per day."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    axes = axes.flatten()

    for i, h in enumerate(HORIZONS):
        p   = preds[h]
        df_tmp = pd.DataFrame({
            "date":   p["date"],
            "actual": p["actual"],
            "har_rv": p["har_rv"],
            "lgbm":   p["lgbm"],
        }).groupby("date").mean().sort_index()

        ax = axes[i]
        ax.plot(df_tmp.index, df_tmp["actual"], color="black",  lw=1.2, label="Actual RV",  alpha=0.85)
        ax.plot(df_tmp.index, df_tmp["har_rv"], color="royalblue", lw=1.0, label="HAR-RV", linestyle="--", alpha=0.8)
        ax.plot(df_tmp.index, df_tmp["lgbm"],   color="tomato",    lw=1.0, label="LightGBM", linestyle="-.", alpha=0.8)

        ax.set_title(f"Forward RV {h}d  (all tickers, daily mean)", fontsize=11)
        ax.set_ylabel("Annualised RV")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Predicted vs Actual Realised Volatility — Test Set (2024)", fontsize=13, y=1.01)
    plt.tight_layout()
    out = PLOTS_DIR / "pred_vs_actual_all_horizons.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


def plot_ticker_deepdive(test_df, har_models, lgbm_models, ticker):
    sub = test_df[test_df["ticker"] == ticker].sort_values("date")
    if sub.empty:
        print(f"Ticker {ticker} not found in test set, skipping deep-dive.")
        return

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    axes = axes.flatten()

    for i, h in enumerate(HORIZONS):
        actual = sub[f"rv_fwd_{h}"].values
        har_p  = har_models[h].predict(sub[RV_FEATURES].values)
        lgb_p  = lgbm_models[h].predict(sub[ALL_FEATURES].values)
        dates  = pd.to_datetime(sub["date"].values)

        ax = axes[i]
        ax.plot(dates, actual, color="black",     lw=1.4, label="Actual RV")
        ax.plot(dates, har_p,  color="royalblue", lw=1.0, label="HAR-RV",   linestyle="--")
        ax.plot(dates, lgb_p,  color="tomato",    lw=1.0, label="LightGBM", linestyle="-.")

        ax.set_title(f"{ticker}  —  Forward RV {h}d", fontsize=11)
        ax.set_ylabel("Annualised RV")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{ticker} — Predicted vs Actual RV across Horizons (Test 2024)", fontsize=13, y=1.01)
    plt.tight_layout()
    out = PLOTS_DIR / f"ticker_deepdive_{ticker.replace('.','_')}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


def main():
    print("Loading dataset and models ...")
    df      = pd.read_parquet(DATASET_PATH)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    print(f"Test rows: {len(test_df):,}  |  Tickers: {test_df['ticker'].nunique()}")

    har_models, lgbm_models = load_models()

    print("Generating aggregate plot ...")
    preds = predict_all(test_df, har_models, lgbm_models)
    plot_aggregate(preds)

    print(f"Generating {DEEPDIVE_TICKER} deep-dive ...")
    plot_ticker_deepdive(test_df, har_models, lgbm_models, DEEPDIVE_TICKER)

    print("Done.")


if __name__ == "__main__":
    main()
