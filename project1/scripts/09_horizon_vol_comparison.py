"""
Compare mean and std of intraday range / return across
daily / weekly / monthly / yearly horizons for:
  - Large Cap (average of all large-cap stocks)
  - Mid Cap   (average of all mid-cap stocks)
  - Selected ETFs: SPY, QQQ, IWM, GLD, XLE, VXX
"""

import os, re, warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────────
BASE     = "/root/autodl-tmp/CSY/project1"
RAW_STK  = os.path.join(BASE, "data/raw/stocks")
RAW_ETF  = os.path.join(BASE, "data/raw/etfs")
UNIV     = os.path.join(BASE, "output/universe_classification.csv")
OUT_HTML = os.path.join(BASE, "reports/horizon_vol_comparison.html")
PLOTLY_JS = "/root/miniconda3/lib/python3.12/site-packages/plotly/package_data/plotly.min.js"

HIGHLIGHT_ETFS = ["SPY.ETF", "QQQ.OQ", "IWM.ETF", "GLD.ETF", "XLE.ETF", "VXX.OQ"]
HORIZONS = {"Daily": "D", "Weekly": "W", "Monthly": "ME", "Yearly": "YE"}


def _fig_to_div(fig) -> str:
    """
    Render a plotly figure to a bare <div>+<script> block with NO external
    dependencies and NO embedded library copy.

    Approach: generate a full standalone HTML (include_plotlyjs=True), then
    strip out the plotly library block and the PlotlyConfig init — the caller
    already puts plotly.js in <head>.
    """
    full = fig.to_html(full_html=True, include_plotlyjs=True)
    m = re.search(r"<body>(.*?)</body>", full, re.DOTALL)
    body = m.group(1).strip() if m else full

    # Remove the big inline library block (starts with /** plotly.js or !function)
    body = re.sub(
        r'<script[^>]*>\s*(?:/\*\*[\s\S]*?plotly\.js|!function)[\s\S]*?</script>',
        "", body, count=1
    )
    # Remove window.PlotlyConfig block emitted by plotly 6.x
    body = re.sub(
        r'<script[^>]*>\s*window\.PlotlyConfig\s*=[\s\S]*?</script>',
        "", body
    )
    return body.strip()


# ── data loading & stats ──────────────────────────────────────────────────────
def load_ohlcv(ticker: str) -> pd.DataFrame | None:
    for subdir in [RAW_STK, RAW_ETF]:
        path = os.path.join(subdir, f"{ticker}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df.columns = [c.lower() for c in df.columns]
            return df[["open", "high", "low", "close"]].dropna()
    return None


def compute_horizon_stats(df: pd.DataFrame, rule: str) -> dict:
    if rule == "D":
        r = df.copy()
    else:
        r = df.resample(rule).agg(
            open=("open", "first"), high=("high", "max"),
            low=("low", "min"),   close=("close", "last"),
        ).dropna()
    r = r[r["open"] > 0]
    range_pct = (r["high"] - r["low"]) / r["open"]
    oc_pct    = (r["close"] - r["open"]) / r["open"]
    return {
        "range_mean": float(range_pct.mean()), "range_std": float(range_pct.std()),
        "oc_mean":    float(oc_pct.mean()),    "oc_std":    float(oc_pct.std()),
        "n": len(r),
    }


def build_group_stats(tickers, label):
    records = {h: [] for h in HORIZONS}
    loaded = 0
    for t in tickers:
        df = load_ohlcv(t)
        if df is None or len(df) < 60:
            continue
        loaded += 1
        for h, rule in HORIZONS.items():
            records[h].append(compute_horizon_stats(df, rule))
    print(f"  {label}: loaded {loaded}/{len(tickers)}")
    result = {}
    for h, lst in records.items():
        if lst:
            result[h] = {k: float(np.mean([s[k] for s in lst])) for k in ("range_mean","range_std","oc_mean","oc_std")}
            result[h]["n"] = int(np.mean([s["n"] for s in lst]))
    return result


# ── load data ─────────────────────────────────────────────────────────────────
univ = pd.read_csv(UNIV)
large_tickers = univ[univ["category"] == "large"]["ticker"].tolist()
mid_tickers   = univ[univ["category"] == "mid"]["ticker"].tolist()

print("Computing Large Cap stats...")
large_stats = build_group_stats(large_tickers, "Large Cap")
print("Computing Mid Cap stats...")
mid_stats   = build_group_stats(mid_tickers,   "Mid Cap")

print("Computing ETF stats...")
etf_stats: dict[str, dict] = {}
for etf in HIGHLIGHT_ETFS:
    df = load_ohlcv(etf)
    if df is None:
        print(f"  {etf}: not found"); continue
    etf_stats[etf] = {h: compute_horizon_stats(df, rule) for h, rule in HORIZONS.items()}
    print(f"  {etf}: ok")

# ── build figures ─────────────────────────────────────────────────────────────
horizon_labels = list(HORIZONS.keys())
groups = {"Large Cap": large_stats, "Mid Cap": mid_stats, **etf_stats}
COLORS = {
    "Large Cap": "#1f77b4", "Mid Cap": "#aec7e8",
    "SPY.ETF": "#2ca02c",   "QQQ.OQ": "#17becf",
    "IWM.ETF": "#ff7f0e",   "GLD.ETF": "#d62728",
    "XLE.ETF": "#9467bd",   "VXX.OQ": "#8c564b",
}


def bar_traces(metric_key):
    traces = []
    for name, stats in groups.items():
        y = [stats.get(h, {}).get(metric_key) for h in horizon_labels]
        traces.append(go.Bar(
            name=name, x=horizon_labels, y=y,
            marker_color=COLORS.get(name, "#999"),
            hovertemplate="<b>" + name + "</b><br>%{x}: %{y:.4f}<extra></extra>",
        ))
    return traces


# ① 4-panel bar chart
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        "振幅均值 range_pct mean", "振幅标准差 range_pct std",
        "收益均值 oc_pct mean",    "收益标准差 oc_pct std",
    ],
    vertical_spacing=0.14, horizontal_spacing=0.08,
)
show_legend = True
for metric_key, row, col in [
    ("range_mean", 1, 1), ("range_std", 1, 2),
    ("oc_mean",    2, 1), ("oc_std",   2, 2),
]:
    for tr in bar_traces(metric_key):
        tr.showlegend = show_legend
        fig.add_trace(tr, row=row, col=col)
    show_legend = False
fig.update_layout(
    title=dict(text="各时间维度波动性对比：大盘股 / 中盘股 / 典型 ETF", font_size=18),
    barmode="group", height=820,
    legend=dict(orientation="h", y=1.06, x=0),
    template="plotly_white",
)

# ② heatmaps
def heatmap_fig(metric_key, title):
    names = list(groups.keys())
    data  = [[groups[n].get(h, {}).get(metric_key, np.nan) for h in horizon_labels] for n in names]
    hm = go.Figure(go.Heatmap(
        z=data, x=horizon_labels, y=names,
        colorscale="Blues",
        text=[[f"{v:.4f}" if not np.isnan(v) else "" for v in row] for row in data],
        texttemplate="%{text}",
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.4f}<extra></extra>",
    ))
    hm.update_layout(title=title, height=360, template="plotly_white",
                     margin=dict(l=130, r=20, t=50, b=40))
    return hm

hm_figs = [
    heatmap_fig("range_mean", "振幅均值 range_pct mean"),
    heatmap_fig("range_std",  "振幅标准差 range_pct std"),
    heatmap_fig("oc_mean",    "收益均值 oc_pct mean"),
    heatmap_fig("oc_std",     "收益标准差 oc_pct std"),
]

# ③ summary table
rows_data = []
for name, stats in groups.items():
    for h in horizon_labels:
        s = stats.get(h, {})
        rows_data.append({
            "Group": name, "Horizon": h,
            "range_mean": round(s.get("range_mean", np.nan), 6),
            "range_std":  round(s.get("range_std",  np.nan), 6),
            "oc_mean":    round(s.get("oc_mean",    np.nan), 6),
            "oc_std":     round(s.get("oc_std",     np.nan), 6),
            "N(avg)":     s.get("n", ""),
        })
tbl_df = pd.DataFrame(rows_data)
tbl_fig = go.Figure(go.Table(
    header=dict(
        values=[f"<b>{c}</b>" for c in tbl_df.columns],
        fill_color="#1f77b4", font=dict(color="white", size=12), align="center",
    ),
    cells=dict(
        values=[tbl_df[c].tolist() for c in tbl_df.columns],
        align=["left","left","right","right","right","right","right"],
        font=dict(size=11), height=24,
    ),
))
tbl_fig.update_layout(
    height=len(rows_data) * 26 + 60,
    margin=dict(l=10, r=10, t=10, b=10),
)

# ── assemble HTML ─────────────────────────────────────────────────────────────
with open(PLOTLY_JS, "r", encoding="utf-8") as f:
    plotly_js_text = f.read()

CSS = """
body{font-family:'Segoe UI',Arial,sans-serif;max-width:1400px;margin:0 auto;padding:20px;background:#f8f9fa}
h1{color:#1f77b4;border-bottom:2px solid #1f77b4;padding-bottom:10px}
h2{color:#333;margin-top:32px}
.info{background:#e8f4fd;border-left:4px solid #1f77b4;padding:12px 16px;border-radius:4px;margin:16px 0;font-size:13px}
.card{background:white;border-radius:8px;padding:12px 16px;margin:16px 0;box-shadow:0 1px 4px rgba(0,0,0,.1)}
"""

sections = [
    f"<!DOCTYPE html><html lang='zh'><head><meta charset='UTF-8'>"
    f"<title>Horizon Vol Comparison</title>"
    f"<style>{CSS}</style>"
    f"<script>{plotly_js_text}</script>"
    f"</head><body>",

    "<h1>各时间维度波动性对比报告</h1>",
    "<div class='info'><b>指标说明：</b><ul>"
    "<li><b>range_pct</b>：(High−Low)/Open，价格振幅（恒正）</li>"
    "<li><b>oc_pct</b>：(Close−Open)/Open，方向性收益（有正有负）</li>"
    "<li>周/月/年数据由日线 OHLCV 聚合（取周期内最高/最低/首开/末收）</li>"
    f"<li>大盘股 {len(large_tickers)} 只均值；中盘股 {len(mid_tickers)} 只均值</li>"
    f"<li>ETF 单独展示：{', '.join(etf_stats.keys())}</li>"
    "</ul></div>",

    "<div class='card'><h2>① 分组柱状图</h2>" + _fig_to_div(fig) + "</div>",

    "<div class='card'><h2>② 热力图</h2>"
    + "".join(_fig_to_div(hm) for hm in hm_figs)
    + "</div>",

    "<div class='card'><h2>③ 完整数据表</h2>" + _fig_to_div(tbl_fig) + "</div>",

    "</body></html>",
]

with open(OUT_HTML, "w", encoding="utf-8") as f:
    f.write("\n".join(sections))

print(f"\nReport saved → {OUT_HTML}")
print(f"File size: {os.path.getsize(OUT_HTML)/1024/1024:.1f} MB")
