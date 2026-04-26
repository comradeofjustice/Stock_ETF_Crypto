#!/usr/bin/env python3
"""
plot_dir_accuracy.py
====================
Plot predicted vs actual RV for all four models on the test set.
Prediction lines are colored segment-by-segment:
  green = direction correct  (sign(pred - rv_current) == sign(actual - rv_current))
  red   = direction wrong

Models: HAR-RV · LightGBM · LSTM · Transformer
Output: results/plots/dir_accuracy_all_horizons.png
"""

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math
import yaml
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((PROJECT_ROOT / "config/experiment.yaml").read_text())

DATASET_PATH     = PROJECT_ROOT / CONFIG["data"]["dataset_path"]
HAR_DIR          = PROJECT_ROOT / "models/har_rv"
LGBM_DIR         = PROJECT_ROOT / "models/lgbm"
LSTM_DIR         = PROJECT_ROOT / "models/lstm"
TRANSFORMER_DIR  = PROJECT_ROOT / "models/transformer"
PLOTS_DIR        = PROJECT_ROOT / "results/plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS     = CONFIG["horizons"]
SEQ_LEN      = CONFIG["sequence_len"]
RV_FEATURES  = [f"rv_{n}" for n in CONFIG["rv_windows"]]
ALL_FEATURES = (
    ["range_pct", "oc_pct", "atr_5", "atr_14", "atr_64", "atr_128"]
    + RV_FEATURES
)
LSTM_CFG = CONFIG["models"]["lstm"]
TF_CFG   = CONFIG["models"]["transformer"]
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COLOR_ACT = "black"

# Per-model: (correct_color, wrong_color, linewidth, alpha)
# Correct = bright/saturated, Wrong = very dark same hue — high contrast
MODEL_STYLES = {
    "har":         ("#48CAE4", "#03045E", 0.9, 0.85),   # bright cyan   / dark navy
    "lgbm":        ("#FB8500", "#370617", 1.2, 0.85),   # bright orange / near-black brown
    "lstm":        ("#52B788", "#1B4332", 2.0, 0.90),   # bright green  / dark forest green
    "transformer": ("#E040FB", "#4A0072", 2.0, 0.90),   # bright magenta/ dark purple
}


# ── Neural network definitions (must match training scripts) ──────────────────

class LSTMModel(nn.Module):
    def __init__(self, n_features, n_targets):
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
            nn.Softplus(),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerModel(nn.Module):
    def __init__(self, n_features, n_targets):
        super().__init__()
        d_model = TF_CFG["d_model"]
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=TF_CFG["dropout"])
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=TF_CFG["nhead"],
            dim_feedforward=TF_CFG["dim_feedforward"],
            dropout=TF_CFG["dropout"], batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                             num_layers=TF_CFG["num_encoder_layers"])
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_targets),
            nn.Softplus(),
        )

    def forward(self, x):
        x = self.pos_enc(self.input_proj(x))
        return self.head(self.encoder(x)[:, -1, :])


# ── Model loading ─────────────────────────────────────────────────────────────

def load_sklearn_models():
    har, lgbm = {}, {}
    for h in HORIZONS:
        with open(HAR_DIR  / f"har_rv_{h}.pkl", "rb") as f: har[h]  = pickle.load(f)
        with open(LGBM_DIR / f"lgbm_{h}.pkl",   "rb") as f: lgbm[h] = pickle.load(f)
    return har, lgbm


def load_neural_models():
    n_feat, n_tgt = len(ALL_FEATURES), len(HORIZONS)

    lstm = LSTMModel(n_feat, n_tgt).to(DEVICE)
    lstm.load_state_dict(torch.load(LSTM_DIR / "lstm.pt", map_location=DEVICE))
    lstm.eval()

    tf = TransformerModel(n_feat, n_tgt).to(DEVICE)
    tf.load_state_dict(torch.load(TRANSFORMER_DIR / "transformer.pt", map_location=DEVICE))
    tf.eval()

    with open(LSTM_DIR        / "scaler_X.pkl", "rb") as f: lstm_scaler = pickle.load(f)
    with open(TRANSFORMER_DIR / "scaler_X.pkl", "rb") as f: tf_scaler   = pickle.load(f)

    return lstm, lstm_scaler, tf, tf_scaler


# ── Sequence prediction helpers ───────────────────────────────────────────────

def seq_predict(model, test_df, scaler, batch_size=2048):
    """Build sequences per ticker, run inference, return per-row predictions
    aligned to test_df index (first SEQ_LEN rows per ticker → NaN)."""
    X_all  = scaler.transform(test_df[ALL_FEATURES].values.astype(np.float32))
    n_tgt  = len(HORIZONS)
    preds  = np.full((len(test_df), n_tgt), np.nan, dtype=np.float32)

    for _, grp in test_df.groupby("ticker"):
        idx  = grp.index.tolist()
        X_t  = X_all[idx]
        seqs, positions = [], []
        for i in range(SEQ_LEN, len(X_t)):
            seqs.append(X_t[i - SEQ_LEN: i])
            positions.append(idx[i])

        if not seqs:
            continue

        seqs_t = torch.from_numpy(np.array(seqs, dtype=np.float32))
        out_batches = []
        with torch.no_grad():
            for start in range(0, len(seqs_t), batch_size):
                out_batches.append(
                    model(seqs_t[start: start + batch_size].to(DEVICE)).cpu().numpy()
                )
        preds[positions] = np.vstack(out_batches)

    return preds  # shape (N, n_targets)


# ── Daily aggregation ─────────────────────────────────────────────────────────

def build_daily_all(test_df, har, lgbm, lstm_preds, tf_preds):
    dates = pd.to_datetime(test_df["date"].values)
    rows  = {"date": dates}
    for j, h in enumerate(HORIZONS):
        rows[f"actual_{h}"]  = test_df[f"rv_fwd_{h}"].values
        rows[f"current_{h}"] = test_df[f"rv_{h}"].values
        rows[f"har_{h}"]     = har[h].predict(test_df[RV_FEATURES].values)
        rows[f"lgbm_{h}"]    = lgbm[h].predict(test_df[ALL_FEATURES].values)
        rows[f"lstm_{h}"]    = lstm_preds[:, j]
        rows[f"tf_{h}"]      = tf_preds[:, j]

    tmp   = pd.DataFrame(rows)
    daily = tmp.groupby("date").mean().sort_index()
    return daily


# ── Plot ──────────────────────────────────────────────────────────────────────

def dir_correct(pred, actual, current):
    return np.sign(pred - current) == np.sign(actual - current)


def colored_line(ax, x, y, mask, col_ok, col_err, lw, alpha):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    pts  = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    cols = [col_ok if c else col_err for c in mask[:-1]]
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=lw, alpha=alpha))


def plot_all(daily):
    MODEL_LABELS = {
        "har":  "HAR-RV",
        "lgbm": "LightGBM",
        "lstm": "LSTM",
        "tf":   "Transformer",
    }
    MODEL_CFG_KEYS = {
        "har":  "har",
        "lgbm": "lgbm",
        "lstm": "lstm",
        "tf":   "transformer",
    }

    fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    axes = axes.flatten()

    for i, h in enumerate(HORIZONS):
        ax = axes[i]
        actual  = daily[f"actual_{h}"].values
        current = daily[f"current_{h}"].values
        dates   = mdates.date2num(daily.index.to_pydatetime())
        y_cap   = np.nanpercentile(actual, 99) * 1.15

        ax.set_xlim(daily.index[0], daily.index[-1])
        ax.plot(daily.index, actual, color=COLOR_ACT, lw=1.8, zorder=5, label="Actual RV")

        legend_elems = [Line2D([0], [0], color=COLOR_ACT, lw=1.8, label="Actual RV")]

        for key, cfg_key in MODEL_CFG_KEYS.items():
            col_ok, col_err, lw, alpha = MODEL_STYLES[cfg_key]
            label = MODEL_LABELS[key]
            pred  = daily[f"{key}_{h}"].values
            valid = ~np.isnan(pred)
            if valid.sum() < 2:
                continue
            ok  = dir_correct(pred[valid], actual[valid], current[valid])
            acc = ok.mean()
            colored_line(ax, dates[valid], pred[valid], ok,
                         col_ok, col_err, lw, alpha)
            legend_elems += [
                Line2D([0], [0], color=col_ok,  lw=lw,
                       label=f"{label} ✓ {acc:.0%}"),
                Line2D([0], [0], color=col_err, lw=lw,
                       label=f"{label} ✗"),
            ]

        ax.autoscale_view()
        ax.set_ylim(0, y_cap)
        ax.set_title(f"Forward RV {h}d  (all tickers, daily mean)", fontsize=11)
        ax.set_ylabel("Annualised RV")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax.grid(True, alpha=0.3)
        ax.legend(handles=legend_elems, fontsize=7.5, ncol=2, loc="upper left")

    fig.suptitle(
        "Predicted vs Actual RV — Test Set (2024)  |  All Models\n"
        "Bright color = direction correct  |  Dark color = direction wrong\n"
        "HAR (cyan)  ·  LightGBM (orange)  ·  LSTM (green, thick)  ·  Transformer (magenta, thick)",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    out = PLOTS_DIR / "dir_accuracy_all_horizons.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading dataset ...")
    df      = pd.read_parquet(DATASET_PATH)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    print(f"Test rows: {len(test_df):,}  |  Tickers: {test_df['ticker'].nunique()}")

    print("Loading models ...")
    har, lgbm             = load_sklearn_models()
    lstm, lstm_sc, tf, tf_sc = load_neural_models()

    print("Running neural inference ...")
    lstm_preds = seq_predict(lstm, test_df, lstm_sc)
    tf_preds   = seq_predict(tf,   test_df, tf_sc)

    print("Aggregating daily means ...")
    daily = build_daily_all(test_df, har, lgbm, lstm_preds, tf_preds)

    print("Plotting ...")
    plot_all(daily)
    print("Done.")


if __name__ == "__main__":
    main()
