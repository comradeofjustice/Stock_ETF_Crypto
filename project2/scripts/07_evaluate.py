#!/usr/bin/env python3
"""
07_evaluate.py
==============
Aggregate all model metrics into one comparison table and plot.
Outputs:
  results/metrics/all_metrics.csv
  results/plots/rmse_by_horizon.png
  results/plots/dir_acc_by_horizon.png
"""

import logging
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR  = PROJECT_ROOT / "results/metrics"
PLOTS_DIR    = PROJECT_ROOT / "results/plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MODEL_FILES = {
    "HAR-RV":      "har_rv_metrics.csv",
    "LightGBM":    "lgbm_metrics.csv",
    "LSTM":        "lstm_metrics.csv",
    "Transformer": "transformer_metrics.csv",
}


def main():
    chunks = []
    for model, fname in MODEL_FILES.items():
        path = METRICS_DIR / fname
        if not path.exists():
            log.warning(f"Missing: {path}")
            continue
        df = pd.read_csv(path)
        df["model"] = model
        chunks.append(df)

    if not chunks:
        log.error("No metric files found. Run training scripts first.")
        return

    all_metrics = pd.concat(chunks, ignore_index=True)
    out = METRICS_DIR / "all_metrics.csv"
    all_metrics.to_csv(out, index=False)
    log.info(f"Combined metrics → {out}")

    test_metrics = all_metrics[all_metrics["split"] == "test"]

    print("\n=== TEST SET RMSE (lower is better) ===")
    rmse_pivot = test_metrics.pivot_table(index="horizon", columns="model", values="rmse")
    print(rmse_pivot.round(6).to_string())

    print("\n=== TEST SET DIR_ACC (higher is better) ===")
    acc_pivot = test_metrics.pivot_table(index="horizon", columns="model", values="dir_acc")
    print(acc_pivot.round(4).to_string())

    # Identify winner per horizon
    print("\n=== WINNER BY HORIZON (test RMSE) ===")
    winner = rmse_pivot.idxmin(axis=1).rename("best_model")
    print(winner.to_string())

    # Plot RMSE
    horizons = sorted(test_metrics["horizon"].unique())
    models   = test_metrics["model"].unique()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for model in models:
        sub = test_metrics[test_metrics["model"] == model].sort_values("horizon")
        axes[0].plot(sub["horizon"], sub["rmse"],    marker="o", label=model)
        axes[1].plot(sub["horizon"], sub["dir_acc"], marker="o", label=model)

    axes[0].set_title("RMSE by Horizon (test)")
    axes[0].set_xlabel("Forward horizon (days)")
    axes[0].set_ylabel("RMSE")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Directional Accuracy by Horizon (test)")
    axes[1].set_xlabel("Forward horizon (days)")
    axes[1].set_ylabel("Dir Acc")
    axes[1].axhline(0.5, color="grey", linestyle="--", alpha=0.5, label="random")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = PLOTS_DIR / "model_comparison.png"
    plt.savefig(fig_path, dpi=150)
    log.info(f"Plot saved → {fig_path}")
    plt.close()


if __name__ == "__main__":
    main()
