#!/usr/bin/env python3
"""
05_train_lstm.py
================
PyTorch LSTM for multi-horizon RV prediction.
Input:  sequence of seq_len days × n_features
Output: 6 forward RV values (rv_fwd_10 … rv_fwd_60)

Saves:
  models/lstm/lstm.pt
  results/metrics/lstm_metrics.csv
"""

import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import pickle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((PROJECT_ROOT / "config/experiment.yaml").read_text())

DATASET_PATH = PROJECT_ROOT / CONFIG["data"]["dataset_path"]
MODEL_DIR    = PROJECT_ROOT / "models/lstm"
METRICS_DIR  = PROJECT_ROOT / "results/metrics"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS   = CONFIG["horizons"]
SEQ_LEN    = CONFIG["sequence_len"]
FEATURES   = (
    ["range_pct", "oc_pct", "atr_5", "atr_14", "atr_64", "atr_128"]
    + [f"rv_{n}" for n in CONFIG["rv_windows"]]
)
TARGETS    = [f"rv_fwd_{h}" for h in HORIZONS]
TRAIN_CFG  = CONFIG["training"]
LSTM_CFG   = CONFIG["models"]["lstm"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RVSequenceDataset(Dataset):
    """Build (seq_len, n_feat) → (n_targets,) samples per ticker."""

    def __init__(self, df: pd.DataFrame, scaler_X=None, fit_scaler=False):
        self.samples_X = []
        self.samples_y = []

        X_all = df[FEATURES].values.astype(np.float32)
        y_all = df[TARGETS].values.astype(np.float32)

        if fit_scaler:
            self.scaler_X = StandardScaler().fit(X_all)
        else:
            self.scaler_X = scaler_X

        X_all = self.scaler_X.transform(X_all)

        for ticker, grp in df.groupby("ticker"):
            idx = grp.index
            X_t = X_all[idx]
            y_t = y_all[idx]
            for i in range(SEQ_LEN, len(X_t)):
                self.samples_X.append(X_t[i - SEQ_LEN: i])
                self.samples_y.append(y_t[i])

        self.samples_X = np.array(self.samples_X, dtype=np.float32)
        self.samples_y = np.array(self.samples_y, dtype=np.float32)

    def __len__(self):
        return len(self.samples_X)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.samples_X[idx]),
            torch.from_numpy(self.samples_y[idx]),
        )


class LSTMModel(nn.Module):
    def __init__(self, n_features: int, n_targets: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=LSTM_CFG["hidden_size"],
            num_layers=LSTM_CFG["num_layers"],
            dropout=LSTM_CFG["dropout"] if LSTM_CFG["num_layers"] > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Dropout(LSTM_CFG["dropout"]),
            nn.Linear(LSTM_CFG["hidden_size"], n_targets),
            nn.Softplus(),   # RV is always positive
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def directional_accuracy(y_true, y_pred, y_current):
    true_dir = (y_true > y_current).astype(int)
    pred_dir = (y_pred > y_current).astype(int)
    return (true_dir == pred_dir).mean()


def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            preds.append(model(X_b.to(device)).cpu().numpy())
            trues.append(y_b.numpy())
    return np.vstack(preds), np.vstack(trues)


def main():
    set_seed(TRAIN_CFG["seed"])

    df = pd.read_parquet(DATASET_PATH).reset_index(drop=True)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df   = df[df["split"] == "val"].reset_index(drop=True)
    test_df  = df[df["split"] == "test"].reset_index(drop=True)

    log.info("Building datasets ...")
    train_ds = RVSequenceDataset(train_df, fit_scaler=True)
    val_ds   = RVSequenceDataset(val_df,   scaler_X=train_ds.scaler_X)
    test_ds  = RVSequenceDataset(test_df,  scaler_X=train_ds.scaler_X)

    # Save scaler for inference
    with open(MODEL_DIR / "scaler_X.pkl", "wb") as f:
        pickle.dump(train_ds.scaler_X, f)

    BS = TRAIN_CFG["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BS, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BS, shuffle=False, num_workers=0)

    model = LSTMModel(n_features=len(FEATURES), n_targets=len(TARGETS)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_CFG["lr"])
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    log.info(f"Training LSTM on {DEVICE} ...")
    for epoch in range(1, TRAIN_CFG["epochs"] + 1):
        model.train()
        train_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(X_b)
        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                val_loss += criterion(model(X_b.to(DEVICE)), y_b.to(DEVICE)).item() * len(X_b)
        val_loss /= len(val_ds)

        if epoch % 10 == 0:
            log.info(f"  Epoch {epoch:3d}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= TRAIN_CFG["patience"]:
                log.info(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODEL_DIR / "lstm.pt")
    log.info(f"Saved → {MODEL_DIR}/lstm.pt")

    # Metrics
    records = []
    for split_name, loader, split_df in [
        ("val",  val_loader,  val_df),
        ("test", test_loader, test_df),
    ]:
        # rebuild reference rv per horizon from the split df
        # (samples after SEQ_LEN offset so trim accordingly)
        y_pred, y_true = evaluate(model.to(DEVICE), loader, DEVICE)

        for i, h in enumerate(HORIZONS):
            ref_rv = split_df[f"rv_{h}"].values[SEQ_LEN:]
            min_len = min(len(y_true), len(ref_rv))
            records.append({
                "model":   "LSTM",
                "horizon": h,
                "split":   split_name,
                "mse":     mean_squared_error(y_true[:min_len, i], y_pred[:min_len, i]),
                "rmse":    np.sqrt(mean_squared_error(y_true[:min_len, i], y_pred[:min_len, i])),
                "mae":     mean_absolute_error(y_true[:min_len, i], y_pred[:min_len, i]),
                "dir_acc": directional_accuracy(y_true[:min_len, i], y_pred[:min_len, i],
                                                ref_rv[:min_len]),
            })

    metrics = pd.DataFrame(records)
    out = METRICS_DIR / "lstm_metrics.csv"
    metrics.to_csv(out, index=False)
    log.info(f"\nMetrics saved → {out}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
