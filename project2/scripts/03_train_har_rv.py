#!/usr/bin/env python3
"""
03_train_har_rv.py
==================
HAR-RV baseline (Corsi 2009).
One LinearRegression per horizon; features are the rolling RV windows
(rv_10, rv_20, rv_30, rv_40, rv_50, rv_60) which map to daily/weekly/monthly
components analogous to the original HAR specification.

Saves:
  models/har_rv/har_rv_<horizon>.pkl   (sklearn pipeline per horizon)
  results/metrics/har_rv_metrics.csv
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
MODEL_DIR    = PROJECT_ROOT / "models/har_rv"
METRICS_DIR  = PROJECT_ROOT / "results/metrics"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS    = CONFIG["horizons"]
RV_FEATURES = [f"rv_{n}" for n in CONFIG["rv_windows"]]   # HAR inputs

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                          y_current: np.ndarray) -> float:
    """Fraction of correct up/down predictions relative to current RV."""
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
        target = f"rv_fwd_{h}"
        log.info(f"Training HAR-RV for horizon {h}d ...")

        X_train = train[RV_FEATURES].values
        y_train = train[target].values
        X_val   = val[RV_FEATURES].values
        y_val   = val[target].values
        X_test  = test[RV_FEATURES].values
        y_test  = test[target].values

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr",     LinearRegression()),
        ])
        pipe.fit(X_train, y_train)

        for split_name, X, y, ref_rv in [
            ("val",  X_val,  y_val,  val[f"rv_{h}"].values),
            ("test", X_test, y_test, test[f"rv_{h}"].values),
        ]:
            y_pred = pipe.predict(X)
            records.append({
                "model":   "HAR-RV",
                "horizon": h,
                "split":   split_name,
                "mse":     mean_squared_error(y, y_pred),
                "rmse":    np.sqrt(mean_squared_error(y, y_pred)),
                "mae":     mean_absolute_error(y, y_pred),
                "dir_acc": directional_accuracy(y, y_pred, ref_rv),
            })

        model_path = MODEL_DIR / f"har_rv_{h}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(pipe, f)
        log.info(f"  Saved → {model_path}")

    metrics = pd.DataFrame(records)
    out = METRICS_DIR / "har_rv_metrics.csv"
    metrics.to_csv(out, index=False)
    log.info(f"\nMetrics saved → {out}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
