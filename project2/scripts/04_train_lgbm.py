#!/usr/bin/env python3
"""
04_train_lgbm.py
================
LightGBM multi-output regression for all 6 horizons simultaneously.
Uses full feature set: range_pct, oc_pct, atr_*, rv_*.

Saves:
  models/lgbm/lgbm_<horizon>.pkl
  results/metrics/lgbm_metrics.csv
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((PROJECT_ROOT / "config/experiment.yaml").read_text())

DATASET_PATH = PROJECT_ROOT / CONFIG["data"]["dataset_path"]
MODEL_DIR    = PROJECT_ROOT / "models/lgbm"
METRICS_DIR  = PROJECT_ROOT / "results/metrics"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = CONFIG["horizons"]
FEATURES = (
    ["range_pct", "oc_pct", "atr_5", "atr_14", "atr_64", "atr_128"]
    + [f"rv_{n}" for n in CONFIG["rv_windows"]]
)
LGBM_PARAMS = CONFIG["models"]["lgbm"]

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

    X_train = train[FEATURES].values
    X_val   = val[FEATURES].values
    X_test  = test[FEATURES].values

    records = []

    for h in HORIZONS:
        target = f"rv_fwd_{h}"
        log.info(f"Training LightGBM for horizon {h}d ...")

        y_train = train[target].values
        y_val_h = val[target].values
        y_test_h = test[target].values

        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURES)
        dval   = lgb.Dataset(X_val,   label=y_val_h, reference=dtrain)

        params = {
            "objective":        "regression",
            "metric":           "rmse",
            "learning_rate":    LGBM_PARAMS["learning_rate"],
            "max_depth":        LGBM_PARAMS["max_depth"],
            "subsample":        LGBM_PARAMS["subsample"],
            "colsample_bytree": LGBM_PARAMS["colsample_bytree"],
            "n_jobs":           LGBM_PARAMS["n_jobs"],
            "verbosity":        -1,
            "seed":             CONFIG["training"]["seed"],
        }

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=LGBM_PARAMS["n_estimators"],
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=CONFIG["training"]["patience"],
                                   verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        for split_name, X, y, ref_rv in [
            ("val",  X_val,   y_val_h,  val[f"rv_{h}"].values),
            ("test", X_test,  y_test_h, test[f"rv_{h}"].values),
        ]:
            y_pred = model.predict(X)
            records.append({
                "model":   "LightGBM",
                "horizon": h,
                "split":   split_name,
                "mse":     mean_squared_error(y, y_pred),
                "rmse":    np.sqrt(mean_squared_error(y, y_pred)),
                "mae":     mean_absolute_error(y, y_pred),
                "dir_acc": directional_accuracy(y, y_pred, ref_rv),
            })

        model_path = MODEL_DIR / f"lgbm_{h}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        log.info(f"  Saved → {model_path}  (best iter: {model.best_iteration})")

    metrics = pd.DataFrame(records)
    out = METRICS_DIR / "lgbm_metrics.csv"
    metrics.to_csv(out, index=False)
    log.info(f"\nMetrics saved → {out}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
