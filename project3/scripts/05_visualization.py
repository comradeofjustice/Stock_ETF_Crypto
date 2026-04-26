"""
05_visualization.py
生成全部交互式 HTML 图表（plotly）：
  1. 波动率相关系数热力图（Pearson 21d）
  2. 波动率协方差矩阵热力图（21d）
  3. BTC / ETH / SOL 滚动波动率时序 vs 代表性 ETF
  4. 层次聚类树（相关性距离）
  5. 散点矩阵：Crypto vs 代表性大盘标的（21d vol）
  6. 全标的平均波动率排名条形图
"""

import pandas as pd
import numpy as np
from pathlib import Path

import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

DATA    = Path(__file__).parent.parent / "data" / "processed"
REPORTS = Path(__file__).parent.parent / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

# ─── 颜色方案 ───────────────────────────────────────────────────────────────
CLASS_COLOR = {"crypto": "#f7931a", "etf": "#1f77b4", "stock": "#2ca02c"}
CRYPTO_COLS = {"BTC": "#f7931a", "ETH": "#627eea", "SOL": "#9945ff"}


def load_meta(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir / "asset_meta.csv").set_index("ticker")


def class_label(ticker: str, meta: pd.DataFrame) -> str:
    return meta["asset_class"].get(ticker, "stock")


# ─── 1. 相关系数热力图 ──────────────────────────────────────────────────────
def plot_corr_heatmap(corr: pd.DataFrame, meta: pd.DataFrame,
                      title: str, filename: str) -> None:
    n = len(corr)
    tickers = corr.index.tolist()

    # 分组颜色条
    class_colors = [CLASS_COLOR.get(class_label(t, meta), "#999") for t in tickers]

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=tickers,
        y=tickers,
        colorscale="RdBu",
        zmid=0,
        zmin=-1, zmax=1,
        colorbar=dict(title="Correlation"),
        hoverongaps=False,
        hovertemplate="x: %{x}<br>y: %{y}<br>r: %{z:.3f}<extra></extra>",
    ))

    # 资产类别分隔线（crypto / etf / stock 边界）
    classes = [class_label(t, meta) for t in tickers]
    boundaries = [i for i in range(1, n) if classes[i] != classes[i - 1]]
    for b in boundaries:
        fig.add_shape(type="line", x0=b - 0.5, x1=b - 0.5, y0=-0.5, y1=n - 0.5,
                      line=dict(color="black", width=1.5))
        fig.add_shape(type="line", x0=-0.5, x1=n - 0.5, y0=b - 0.5, y1=b - 0.5,
                      line=dict(color="black", width=1.5))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        width=max(900, n * 5 + 200),
        height=max(850, n * 5 + 200),
        xaxis=dict(tickfont=dict(size=7), tickangle=90),
        yaxis=dict(tickfont=dict(size=7)),
        margin=dict(l=120, r=80, t=80, b=120),
    )
    path = REPORTS / filename
    fig.write_html(path)
    print(f"  saved → {path.name}")


# ─── 2. 协方差热力图 ────────────────────────────────────────────────────────
def plot_cov_heatmap(cov: pd.DataFrame, meta: pd.DataFrame,
                     title: str, filename: str) -> None:
    tickers = cov.index.tolist()
    n = len(tickers)

    # log scale 协方差值便于可视化（全正，取 log1p）
    z = np.log1p(np.abs(cov.values)) * np.sign(cov.values)

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=tickers,
        y=tickers,
        colorscale="Viridis",
        colorbar=dict(title="log(|cov|)·sign"),
        hoverongaps=False,
        customdata=cov.values,
        hovertemplate="x: %{x}<br>y: %{y}<br>cov: %{customdata:.6f}<extra></extra>",
    ))

    classes = [class_label(t, meta) for t in tickers]
    boundaries = [i for i in range(1, n) if classes[i] != classes[i - 1]]
    for b in boundaries:
        fig.add_shape(type="line", x0=b - 0.5, x1=b - 0.5, y0=-0.5, y1=n - 0.5,
                      line=dict(color="white", width=1.5))
        fig.add_shape(type="line", x0=-0.5, x1=n - 0.5, y0=b - 0.5, y1=b - 0.5,
                      line=dict(color="white", width=1.5))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        width=max(900, n * 5 + 200),
        height=max(850, n * 5 + 200),
        xaxis=dict(tickfont=dict(size=7), tickangle=90),
        yaxis=dict(tickfont=dict(size=7)),
        margin=dict(l=120, r=80, t=80, b=120),
    )
    path = REPORTS / filename
    fig.write_html(path)
    print(f"  saved → {path.name}")


# ─── 3. 滚动波动率时序 ─────────────────────────────────────────────────────
def plot_rolling_vol_timeseries(vol_21: pd.DataFrame, vol_63: pd.DataFrame,
                                meta: pd.DataFrame) -> None:
    # 子图一：BTC/ETH/SOL 的 21d vol
    cryptos = [c for c in ["BTC", "ETH", "SOL"] if c in vol_21.columns]
    # 代表性 ETF
    rep_etf = [c for c in ["SPY", "QQQ", "GLD", "VXX", "BITO"] if c in vol_21.columns]

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["Crypto 21-day Rolling Annualised Volatility",
                        "Representative ETF 21-day Rolling Annualised Volatility"],
        shared_xaxes=True,
        vertical_spacing=0.08,
    )

    for crypto in cryptos:
        fig.add_trace(go.Scatter(
            x=vol_21.index, y=vol_21[crypto],
            name=crypto, line=dict(color=CRYPTO_COLS.get(crypto)),
            hovertemplate="%{x|%Y-%m-%d}<br>vol: %{y:.1%}<extra>" + crypto + "</extra>",
        ), row=1, col=1)

    for etf in rep_etf:
        fig.add_trace(go.Scatter(
            x=vol_21.index, y=vol_21[etf],
            name=etf,
            hovertemplate="%{x|%Y-%m-%d}<br>vol: %{y:.1%}<extra>" + etf + "</extra>",
        ), row=2, col=1)

    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(
        title="Rolling Volatility: Crypto vs Representative ETF",
        height=750, width=1200,
        legend=dict(orientation="v", x=1.01),
        margin=dict(r=140),
    )
    path = REPORTS / "rolling_vol_timeseries.html"
    fig.write_html(path)
    print(f"  saved → {path.name}")

    # 全标的 63d vol（按资产类别着色）
    fig2 = go.Figure()
    for ticker in vol_63.columns:
        cls = class_label(ticker, meta)
        color = CLASS_COLOR.get(cls, "#999")
        opacity = 0.9 if cls == "crypto" else 0.2
        width   = 2.5  if cls == "crypto" else 0.8
        fig2.add_trace(go.Scatter(
            x=vol_63.index, y=vol_63[ticker],
            name=ticker, mode="lines",
            line=dict(color=color, width=width),
            opacity=opacity,
            legendgroup=cls,
            showlegend=(ticker in ["BTC", "ETH", "SOL", "SPY", "QQQ"]),
            hovertemplate="%{x|%Y-%m-%d}<br>vol: %{y:.1%}<extra>" + ticker + "</extra>",
        ))
    fig2.update_yaxes(tickformat=".0%")
    fig2.update_layout(
        title="All Assets 63-day Rolling Annualised Volatility (Crypto highlighted)",
        height=600, width=1300,
        margin=dict(r=160),
    )
    path2 = REPORTS / "rolling_vol_all_63d.html"
    fig2.write_html(path2)
    print(f"  saved → {path2.name}")


# ─── 4. 层次聚类树 ─────────────────────────────────────────────────────────
def plot_dendrogram(corr: pd.DataFrame, meta: pd.DataFrame,
                    title: str, filename: str) -> None:
    # 填充 NaN 为 0 后强制对称
    c = corr.fillna(0).values
    c = (c + c.T) / 2
    np.fill_diagonal(c, 1.0)

    dist = 1 - np.abs(c)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2  # 数值误差保对称
    dist = dist.clip(0)

    tickers = corr.index.tolist()

    dist_condensed = squareform(dist)

    dendro = ff.create_dendrogram(
        c,
        labels=tickers,
        linkagefun=lambda x: hierarchy.linkage(dist_condensed, method="ward"),
        color_threshold=0.7,
    )
    dendro.update_layout(
        title=title,
        height=600,
        width=max(1200, len(tickers) * 12),
        xaxis=dict(tickfont=dict(size=7), tickangle=90),
        margin=dict(b=140),
    )
    path = REPORTS / filename
    dendro.write_html(path)
    print(f"  saved → {path.name}")


# ─── 5. 散点矩阵（Crypto × 代表性大盘，21d vol）──────────────────────────
def plot_scatter_matrix(vol_21: pd.DataFrame, meta: pd.DataFrame) -> None:
    cryptos  = [c for c in ["BTC", "ETH", "SOL"] if c in vol_21.columns]
    rep_trad = [c for c in ["SPY", "QQQ", "GLD", "XLK", "XLF", "IWM", "VXX"]
                if c in vol_21.columns]
    cols = cryptos + rep_trad
    df = vol_21[cols].dropna(how="all").copy()

    color_seq = (
        [CRYPTO_COLS[c] for c in cryptos if c in CRYPTO_COLS] +
        [CLASS_COLOR["etf"]] * len(rep_trad)
    )

    # Plotly scatter matrix expects a class column for coloring
    df_long = df.copy()
    df_long["_date"] = df_long.index

    fig = px.scatter_matrix(
        df_long,
        dimensions=cols,
        labels={c: c for c in cols},
        opacity=0.4,
        title="Scatter Matrix: 21-day Volatility (Crypto vs Representative ETF)",
    )
    fig.update_traces(marker=dict(size=3), diagonal_visible=False)
    fig.update_layout(
        height=900, width=900,
        margin=dict(l=80, r=80, t=80, b=80),
    )
    path = REPORTS / "scatter_matrix_crypto_vs_etf.html"
    fig.write_html(path)
    print(f"  saved → {path.name}")


# ─── 6. 平均波动率排名条形图 ───────────────────────────────────────────────
def plot_mean_vol_bar(mean_vol: pd.DataFrame, meta: pd.DataFrame) -> None:
    df = mean_vol.copy()
    df["asset_class"] = df.index.map(lambda t: class_label(t, meta))
    df = df.sort_values("mean_vol_21d", ascending=True)

    colors = df["asset_class"].map(CLASS_COLOR).fillna("#999")

    fig = go.Figure(go.Bar(
        x=df["mean_vol_21d"],
        y=df.index,
        orientation="h",
        marker_color=colors.tolist(),
        hovertemplate="%{y}<br>mean vol (21d): %{x:.1%}<extra></extra>",
    ))

    # 图例
    for cls, col in CLASS_COLOR.items():
        fig.add_trace(go.Bar(x=[None], y=[None], name=cls,
                             marker_color=col, showlegend=True))

    fig.update_layout(
        title="All Assets: Mean Annualised Volatility (21-day rolling)",
        xaxis=dict(title="Mean Annualised Volatility", tickformat=".0%"),
        height=max(600, len(df) * 14),
        width=1000,
        margin=dict(l=140, r=80, t=80, b=60),
        barmode="overlay",
        legend=dict(orientation="h", y=1.02),
    )
    path = REPORTS / "mean_vol_ranking.html"
    fig.write_html(path)
    print(f"  saved → {path.name}")


# ─── 主流程 ────────────────────────────────────────────────────────────────
def main():
    print("=== 05 Visualization ===")

    meta       = load_meta(DATA)
    vol_21     = pd.read_csv(DATA / "vol_21d.csv",           index_col=0, parse_dates=True)
    vol_63     = pd.read_csv(DATA / "vol_63d.csv",           index_col=0, parse_dates=True)
    mean_vol   = pd.read_csv(DATA / "mean_vol.csv",          index_col=0)
    corr_p21   = pd.read_csv(DATA / "corr_pearson_21d.csv",  index_col=0)
    corr_s21   = pd.read_csv(DATA / "corr_spearman_21d.csv", index_col=0)
    cov_21     = pd.read_csv(DATA / "cov_matrix_21d.csv",    index_col=0)

    print("  [1/6] 相关系数热力图（Pearson 21d）…")
    plot_corr_heatmap(corr_p21, meta,
                      "Volatility Pearson Correlation Matrix (21-day rolling)",
                      "vol_corr_pearson_21d.html")

    print("  [2/6] 相关系数热力图（Spearman 21d）…")
    plot_corr_heatmap(corr_s21, meta,
                      "Volatility Spearman Correlation Matrix (21-day rolling)",
                      "vol_corr_spearman_21d.html")

    print("  [3/6] 协方差矩阵热力图（21d）…")
    plot_cov_heatmap(cov_21, meta,
                     "Volatility Covariance Matrix (21-day rolling)",
                     "vol_covariance_21d.html")

    print("  [4/6] 滚动波动率时序…")
    plot_rolling_vol_timeseries(vol_21, vol_63, meta)

    print("  [5/6] 层次聚类树…")
    plot_dendrogram(corr_p21, meta,
                    "Hierarchical Clustering by Volatility Correlation (Pearson 21d)",
                    "vol_cluster_dendrogram.html")

    print("  [6/6] 散点矩阵（Crypto vs ETF）…")
    plot_scatter_matrix(vol_21, meta)

    print("  [+] 平均波动率排名条形图…")
    plot_mean_vol_bar(mean_vol, meta)

    print(f"\n  全部图表已保存到 {REPORTS}")
    print("=== 05 Done ===")


if __name__ == "__main__":
    main()
