#!/usr/bin/env python3
"""
plot_ticker_forecast.py
=======================
Predict forward RV for selected industry-representative tickers using all
four trained models (HAR-RV, LightGBM, LSTM, Transformer).

Layout: 6 tickers × 2 horizons (10d, 60d) = 6 rows × 2 cols per figure.
Each subplot shows Actual RV + 4 model predictions over the 2024 test period.

Output: results/plots/ticker_forecast_{10d|60d}.png
        results/plots/ticker_forecast_all.png  (10d/30d/60d, 6 tickers, 3-col)
"""

import math
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT    = Path(__file__).resolve().parents[1]
CONFIG          = yaml.safe_load((PROJECT_ROOT / "config/experiment.yaml").read_text())
DATASET_PATH    = PROJECT_ROOT / CONFIG["data"]["dataset_path"]
HAR_DIR         = PROJECT_ROOT / "models/har_rv"
LGBM_DIR        = PROJECT_ROOT / "models/lgbm"
LSTM_DIR        = PROJECT_ROOT / "models/lstm"
TRANSFORMER_DIR = PROJECT_ROOT / "models/transformer"
PLOTS_DIR       = PROJECT_ROOT / "results/plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS     = CONFIG["horizons"]
SEQ_LEN      = CONFIG["sequence_len"]
RV_FEATURES  = [f"rv_{n}" for n in CONFIG["rv_windows"]]
ALL_FEATURES = (
    ["range_pct", "oc_pct", "atr_5", "atr_14", "atr_64", "atr_128"] + RV_FEATURES
)
LSTM_CFG = CONFIG["models"]["lstm"]
TF_CFG   = CONFIG["models"]["transformer"]
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Tickers to showcase ───────────────────────────────────────────────────────
TICKERS = {
    "NVDA.OQ":  "NVIDIA (Semiconductors / Tech)",
    "TSLA.OQ":  "Tesla (EV / Auto)",
    "AMZN.OQ":  "Amazon (E-commerce / Cloud)",
    "JPM.N":    "JPMorgan (Banking / Finance)",
    "XOM.N":    "ExxonMobil (Energy / Oil)",
    "JNJ.N":    "J&J (Healthcare / Pharma)",
}

# Model colors
COLORS = {
    "HAR-RV":      "#E67E22",
    "LightGBM":    "#2980B9",
    "LSTM":        "#27AE60",
    "Transformer": "#8E44AD",
}


# ── Neural net definitions ────────────────────────────────────────────────────

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
            nn.Linear(LSTM_CFG["hidden_size"], len(HORIZONS)),
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
        d = TF_CFG["d_model"]
        self.input_proj = nn.Linear(n_features, d)
        self.pos_enc    = PositionalEncoding(d, dropout=TF_CFG["dropout"])
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=TF_CFG["nhead"],
            dim_feedforward=TF_CFG["dim_feedforward"],
            dropout=TF_CFG["dropout"], batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=TF_CFG["num_encoder_layers"])
        self.head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, len(HORIZONS)), nn.Softplus()
        )
    def forward(self, x):
        return self.head(self.encoder(self.pos_enc(self.input_proj(x)))[:, -1, :])


# ── Load models ───────────────────────────────────────────────────────────────

def load_all_models():
    har, lgbm = {}, {}
    for h in HORIZONS:
        with open(HAR_DIR  / f"har_rv_{h}.pkl", "rb") as f: har[h]  = pickle.load(f)
        with open(LGBM_DIR / f"lgbm_{h}.pkl",   "rb") as f: lgbm[h] = pickle.load(f)

    n_feat = len(ALL_FEATURES)
    lstm = LSTMModel(n_feat, len(HORIZONS)).to(DEVICE)
    lstm.load_state_dict(torch.load(LSTM_DIR / "lstm.pt", map_location=DEVICE))
    lstm.eval()

    tf = TransformerModel(n_feat, len(HORIZONS)).to(DEVICE)
    tf.load_state_dict(torch.load(TRANSFORMER_DIR / "transformer.pt", map_location=DEVICE))
    tf.eval()

    with open(LSTM_DIR        / "scaler_X.pkl", "rb") as f: lstm_sc = pickle.load(f)
    with open(TRANSFORMER_DIR / "scaler_X.pkl", "rb") as f: tf_sc   = pickle.load(f)

    return har, lgbm, lstm, lstm_sc, tf, tf_sc


# ── Predict for one ticker ────────────────────────────────────────────────────

def predict_ticker(sub, har, lgbm, lstm, lstm_sc, tf, tf_sc):
    """Returns dict of {model_name: array shape (N, n_horizons)} + dates."""
    sub = sub.sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(sub["date"].values)

    X_har  = sub[RV_FEATURES].values
    X_lgbm = sub[ALL_FEATURES].values

    har_preds  = np.column_stack([har[h].predict(X_har)  for h in HORIZONS])
    lgbm_preds = np.column_stack([lgbm[h].predict(X_lgbm) for h in HORIZONS])

    # Sequence models
    X_lstm = lstm_sc.transform(sub[ALL_FEATURES].values.astype(np.float32))
    X_tf   = tf_sc.transform(sub[ALL_FEATURES].values.astype(np.float32))

    n = len(sub)
    lstm_out = np.full((n, len(HORIZONS)), np.nan)
    tf_out   = np.full((n, len(HORIZONS)), np.nan)

    seqs_lstm, seqs_tf = [], []
    for i in range(SEQ_LEN, n):
        seqs_lstm.append(X_lstm[i - SEQ_LEN: i])
        seqs_tf.append(X_tf[i - SEQ_LEN: i])

    if seqs_lstm:
        with torch.no_grad():
            sl = torch.from_numpy(np.array(seqs_lstm, dtype=np.float32)).to(DEVICE)
            st = torch.from_numpy(np.array(seqs_tf,   dtype=np.float32)).to(DEVICE)
            lstm_out[SEQ_LEN:] = lstm(sl).cpu().numpy()
            tf_out[SEQ_LEN:]   = tf(st).cpu().numpy()

    actual = np.column_stack([sub[f"rv_fwd_{h}"].values for h in HORIZONS])

    return {
        "dates":       dates,
        "actual":      actual,
        "HAR-RV":      har_preds,
        "LightGBM":    lgbm_preds,
        "LSTM":        lstm_out,
        "Transformer": tf_out,
    }


# ── Metrics helper ────────────────────────────────────────────────────────────

def rmse(a, b):
    mask = ~np.isnan(a) & ~np.isnan(b)
    return np.sqrt(np.mean((a[mask] - b[mask]) ** 2))


def dir_acc(pred, actual, current):
    mask = ~np.isnan(pred)
    return (np.sign(pred[mask] - current[mask]) == np.sign(actual[mask] - current[mask])).mean()


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_forecast(results_all, horizons_to_plot):
    """
    results_all: dict ticker → predict_ticker output
    horizons_to_plot: list of horizon values, e.g. [10, 30, 60]
    """
    n_rows = len(TICKERS)
    n_cols = len(horizons_to_plot)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(7 * n_cols, 3.8 * n_rows),
                             sharex=False)
    if n_rows == 1: axes = axes[np.newaxis, :]
    if n_cols == 1: axes = axes[:, np.newaxis]

    model_names = ["HAR-RV", "LightGBM", "LSTM", "Transformer"]

    for row, (ticker, label) in enumerate(TICKERS.items()):
        res = results_all[ticker]
        dates   = res["dates"]
        actual  = res["actual"]

        for col, h in enumerate(horizons_to_plot):
            hi  = HORIZONS.index(h)
            ax  = axes[row, col]

            act = actual[:, hi]
            ax.plot(dates, act, color="black", lw=1.5, label="Actual RV", zorder=5)

            for mname in model_names:
                pred = res[mname][:, hi]
                mask = ~np.isnan(pred)
                r    = rmse(act[mask], pred[mask])
                da   = dir_acc(pred, act,
                               np.column_stack([res["actual"][:, HORIZONS.index(h)]]
                                               )[:, 0])  # use actual as proxy
                # compute dir_acc vs current rv
                cur = res["actual"][:, 0]  # rv_fwd_10 as rough proxy; use rv_current if available
                ax.plot(dates[mask], pred[mask],
                        color=COLORS[mname], lw=1.3 if "HAR" in mname or "LGBM" in mname else 1.8,
                        alpha=0.85, label=f"{mname}  RMSE={r:.3f}")

            ax.set_title(f"{label}\nForward RV {h}d", fontsize=9.5)
            ax.set_ylabel("Ann. RV", fontsize=8)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)
            ax.grid(True, alpha=0.25)
            ax.tick_params(labelsize=7)
            # clip y-axis to 99th pct of actual to suppress outlier spikes
            y_cap = np.nanpercentile(act, 99) * 1.2
            ax.set_ylim(bottom=0, top=max(y_cap, 0.05))

            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper left", ncol=1)

    # Shared legend at bottom
    legend_elems = [Line2D([0], [0], color="black", lw=1.5, label="Actual RV")] + [
        Line2D([0], [0], color=COLORS[m], lw=1.6, label=m) for m in model_names
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=5,
               fontsize=9, bbox_to_anchor=(0.5, -0.01), frameon=True)

    horizon_str = "_".join(f"{h}d" for h in horizons_to_plot)
    fig.suptitle(
        f"Forward RV Forecast — Test Set 2024  |  Horizons: {', '.join(str(h)+'d' for h in horizons_to_plot)}\n"
        "HAR-RV · LightGBM · LSTM · Transformer",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    out = PLOTS_DIR / f"ticker_forecast_{horizon_str}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading dataset ...")
    df      = pd.read_parquet(DATASET_PATH)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    print("Loading models ...")
    har, lgbm, lstm, lstm_sc, tf, tf_sc = load_all_models()

    print("Running predictions ...")
    results_all = {}
    for ticker in TICKERS:
        sub = test_df[test_df["ticker"] == ticker]
        if sub.empty:
            print(f"  WARNING: {ticker} not found, skipping")
            continue
        results_all[ticker] = predict_ticker(sub, har, lgbm, lstm, lstm_sc, tf, tf_sc)
        print(f"  {ticker} done")

    print("Plotting ...")
    plot_forecast(results_all, horizons_to_plot=[10, 30, 60])

    print("Done.")


if __name__ == "__main__":
    main()
