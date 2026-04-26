#!/usr/bin/env python3
"""
03_train_har_rv_log.py
======================
HAR-RV trained on log(RV) targets.  Predictions are converted back to RV
space via exp() for directional-accuracy reporting.

Saves:
  models/har_rv_log/har_rv_log_{horizon}.pkl
  results/metrics/har_rv_log_metrics.csv
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((PROJECT_ROOT / "config/experiment.yaml").read_text())

DATASET_PATH = PROJECT_ROOT / CONFIG["data"]["dataset_path"]
MODEL_DIR    = PROJECT_ROOT / "models/har_rv_log"
METRICS_DIR  = PROJECT_ROOT / "results/metrics"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS    = CONFIG["horizons"]
RV_FEATURES = [f"rv_{n}" for n in CONFIG["rv_windows"]]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def directional_accuracy(y_true, y_pred, y_current):
    true_dir = (y_true > y_current).astype(int)
    pred_dir = (y_pred > y_current).astype(int)
    return (true_dir == pred_dir).mean()


def main():
    df = pd.read_parquet(DATASET_PATH)

    train = df[df["split"] == "train"]
    val   = df[df["split"] == "val"]
    test  = df[df["split"] == "test"]

    records = []

    for h in HORIZONS:
        log_target = f"log_rv_fwd_{h}"
        rv_target  = f"rv_fwd_{h}"
        log.info(f"Training HAR-RV (log) for horizon {h}d ...")

        X_train = train[RV_FEATURES].values
        y_train = train[log_target].values
        X_val   = val[RV_FEATURES].values
        y_val   = val[log_target].values
        X_test  = test[RV_FEATURES].values
        y_test  = test[log_target].values

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr",     LinearRegression()),
        ])
        pipe.fit(X_train, y_train)

        for split_name, X, y_log, df_split in [
            ("val",  X_val,  y_val,  val),
            ("test", X_test, y_test, test),
        ]:
            y_pred_log = pipe.predict(X)
            y_pred_rv  = np.exp(y_pred_log)
            y_true_rv  = df_split[rv_target].values
            ref_rv     = df_split[f"rv_{h}"].values

            records.append({
                "model":        "HAR-RV-log",
                "horizon":      h,
                "split":        split_name,
                "rmse_log":     np.sqrt(mean_squared_error(y_log, y_pred_log)),
                "mae_log":      mean_absolute_error(y_log, y_pred_log),
                "rmse_rv":      np.sqrt(mean_squared_error(y_true_rv, y_pred_rv)),
                "mae_rv":       mean_absolute_error(y_true_rv, y_pred_rv),
                "dir_acc":      directional_accuracy(y_true_rv, y_pred_rv, ref_rv),
            })

        model_path = MODEL_DIR / f"har_rv_log_{h}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(pipe, f)
        log.info(f"  Saved → {model_path}")

    metrics = pd.DataFrame(records)
    out = METRICS_DIR / "har_rv_log_metrics.csv"
    metrics.to_csv(out, index=False)
    log.info(f"\nMetrics saved → {out}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
