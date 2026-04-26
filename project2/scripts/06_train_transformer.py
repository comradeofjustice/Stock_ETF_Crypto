#!/usr/bin/env python3
"""
06_train_transformer.py
=======================
Transformer Encoder for multi-horizon RV prediction.
Same input/output contract as LSTM:
  Input:  (batch, seq_len, n_features)
  Output: (batch, n_targets)  — 6 forward RV horizons

Saves:
  models/transformer/transformer.pt
  models/transformer/scaler_X.pkl
  results/metrics/transformer_metrics.csv
"""

import logging
import math
import pickle
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((PROJECT_ROOT / "config/experiment.yaml").read_text())

DATASET_PATH = PROJECT_ROOT / CONFIG["data"]["dataset_path"]
MODEL_DIR    = PROJECT_ROOT / "models/transformer"
METRICS_DIR  = PROJECT_ROOT / "results/metrics"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS  = CONFIG["horizons"]
SEQ_LEN   = CONFIG["sequence_len"]
FEATURES  = (
    ["range_pct", "oc_pct", "atr_5", "atr_14", "atr_64", "atr_128"]
    + [f"rv_{n}" for n in CONFIG["rv_windows"]]
)
TARGETS   = [f"rv_fwd_{h}" for h in HORIZONS]
TRAIN_CFG = CONFIG["training"]
TF_CFG    = CONFIG["models"]["transformer"]

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

        for _, grp in df.groupby("ticker"):
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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerModel(nn.Module):
    def __init__(self, n_features: int, n_targets: int):
        super().__init__()
        d_model = TF_CFG["d_model"]
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=TF_CFG["dropout"])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=TF_CFG["nhead"],
            dim_feedforward=TF_CFG["dim_feedforward"],
            dropout=TF_CFG["dropout"],
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                             num_layers=TF_CFG["num_encoder_layers"])
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_targets),
            nn.Softplus(),
        )

    def forward(self, x):
        x = self.pos_enc(self.input_proj(x))  # (B, seq, d_model)
        x = self.encoder(x)                    # (B, seq, d_model)
        return self.head(x[:, -1, :])          # last token → targets


def directional_accuracy(y_true, y_pred, y_current):
    true_dir = (y_true > y_current).astype(int)
    pred_dir = (y_pred > y_current).astype(int)
    return (true_dir == pred_dir).mean()


def evaluate(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            preds.append(model(X_b.to(DEVICE)).cpu().numpy())
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

    with open(MODEL_DIR / "scaler_X.pkl", "wb") as f:
        pickle.dump(train_ds.scaler_X, f)

    BS = TRAIN_CFG["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BS, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BS, shuffle=False, num_workers=0)

    model = TransformerModel(n_features=len(FEATURES), n_targets=len(TARGETS)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_CFG["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    log.info(f"Training Transformer on {DEVICE} ...")
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

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                val_loss += criterion(model(X_b.to(DEVICE)), y_b.to(DEVICE)).item() * len(X_b)
        val_loss /= len(val_ds)

        scheduler.step(val_loss)

        if epoch % 10 == 0:
            log.info(f"  Epoch {epoch:3d}  train={train_loss:.6f}  val={val_loss:.6f}")

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
    torch.save(model.state_dict(), MODEL_DIR / "transformer.pt")
    log.info(f"Saved → {MODEL_DIR}/transformer.pt")

    records = []
    for split_name, loader, split_df in [
        ("val",  val_loader,  val_df),
        ("test", test_loader, test_df),
    ]:
        y_pred, y_true = evaluate(model, loader)
        for i, h in enumerate(HORIZONS):
            ref_rv = split_df[f"rv_{h}"].values[SEQ_LEN:]
            min_len = min(len(y_true), len(ref_rv))
            records.append({
                "model":   "Transformer",
                "horizon": h,
                "split":   split_name,
                "mse":     mean_squared_error(y_true[:min_len, i], y_pred[:min_len, i]),
                "rmse":    np.sqrt(mean_squared_error(y_true[:min_len, i], y_pred[:min_len, i])),
                "mae":     mean_absolute_error(y_true[:min_len, i], y_pred[:min_len, i]),
                "dir_acc": directional_accuracy(y_true[:min_len, i], y_pred[:min_len, i],
                                                ref_rv[:min_len]),
            })

    metrics = pd.DataFrame(records)
    out = METRICS_DIR / "transformer_metrics.csv"
    metrics.to_csv(out, index=False)
    log.info(f"\nMetrics saved → {out}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
