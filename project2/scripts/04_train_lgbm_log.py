#!/usr/bin/env python3
"""
04_train_lgbm_log.py
====================
LightGBM trained on log(RV) targets.  Predictions are converted back to RV
space via exp() for directional-accuracy reporting.

Saves:
  models/lgbm_log/lgbm_log_{horizon}.pkl
  results/metrics/lgbm_log_metrics.csv
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
MODEL_DIR    = PROJECT_ROOT / "models/lgbm_log"
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
        log_target = f"log_rv_fwd_{h}"
        rv_target  = f"rv_fwd_{h}"
        log.info(f"Training LightGBM (log) for horizon {h}d ...")

        y_train  = train[log_target].values
        y_val_h  = val[log_target].values
        y_test_h = test[log_target].values

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

        for split_name, X, y_log, df_split in [
            ("val",  X_val,   y_val_h,  val),
            ("test", X_test,  y_test_h, test),
        ]:
            y_pred_log = model.predict(X)
            y_pred_rv  = np.exp(y_pred_log)
            y_true_rv  = df_split[rv_target].values
            ref_rv     = df_split[f"rv_{h}"].values

            records.append({
                "model":    "LightGBM-log",
                "horizon":  h,
                "split":    split_name,
                "rmse_log": np.sqrt(mean_squared_error(y_log, y_pred_log)),
                "mae_log":  mean_absolute_error(y_log, y_pred_log),
                "rmse_rv":  np.sqrt(mean_squared_error(y_true_rv, y_pred_rv)),
                "mae_rv":   mean_absolute_error(y_true_rv, y_pred_rv),
                "dir_acc":  directional_accuracy(y_true_rv, y_pred_rv, ref_rv),
            })

        model_path = MODEL_DIR / f"lgbm_log_{h}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        log.info(f"  Saved → {model_path}  (best iter: {model.best_iteration})")

    metrics = pd.DataFrame(records)
    out = METRICS_DIR / "lgbm_log_metrics.csv"
    metrics.to_csv(out, index=False)
    log.info(f"\nMetrics saved → {out}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
