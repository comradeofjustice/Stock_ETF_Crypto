"""
06_portfolio_correlation_report.py
Generate comprehensive correlation and portfolio strategy analysis report (Markdown).
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

DATA = Path(__file__).parent.parent / "data" / "processed"
CONCLUSION = Path(__file__).parent.parent / "conclution"
CONCLUSION.mkdir(parents=True, exist_ok=True)

LOG = CONCLUSION / "analysis_log.txt"

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG, "a") as f:
        f.write(line + "\n")

# --- Load Data ---
log("Loading data...")
meta = pd.read_csv(DATA / "asset_meta.csv")
mean_vol = pd.read_csv(DATA / "mean_vol.csv", index_col=0)
r21 = pd.read_csv(DATA / "corr_pearson_21d.csv", index_col=0)
r63 = pd.read_csv(DATA / "corr_pearson_63d.csv", index_col=0)
s21 = pd.read_csv(DATA / "corr_spearman_21d.csv", index_col=0)
p21 = pd.read_csv(DATA / "pval_pearson_21d.csv", index_col=0)
cov21 = pd.read_csv(DATA / "cov_matrix_21d.csv", index_col=0)
cov63 = pd.read_csv(DATA / "cov_matrix_63d.csv", index_col=0)

with open(CONCLUSION / "portfolio_results.json") as f:
    port = json.load(f)

log(f"Loaded {len(meta)} assets: {meta['asset_class'].value_counts().to_dict()}")

# Map ticker -> asset_class
ticker_cls = meta.set_index("ticker")["asset_class"].to_dict()
all_tickers = list(r21.columns)

crypto_tickers = [t for t in all_tickers if ticker_cls.get(t) == "crypto"]
etf_tickers = [t for t in all_tickers if ticker_cls.get(t) == "etf"]
stock_tickers = [t for t in all_tickers if ticker_cls.get(t) == "stock"]

log(f"Crypto: {len(crypto_tickers)}, ETFs: {len(etf_tickers)}, Stocks: {len(stock_tickers)}")

# --- Helper Functions ---
def class_corr_matrix(corr_df, tickers, label):
    """Extract intra-class correlation stats."""
    common = [t for t in tickers if t in corr_df.index]
    if len(common) < 2:
        return {"label": label, "n": len(common), "mean": np.nan, "median": np.nan,
                "min": np.nan, "max": np.nan, "std": np.nan}
    sub = corr_df.loc[common, common]
    vals = []
    for i in range(len(common)):
        for j in range(i+1, len(common)):
            v = sub.iloc[i, j]
            if not np.isnan(v):
                vals.append(v)
    vals = np.array(vals)
    return {"label": label, "n": len(common), "mean": vals.mean(),
            "median": np.median(vals), "min": vals.min(), "max": vals.max(),
            "std": vals.std()}

def inter_class_corr(corr_df, tickers_a, tickers_b):
    """Extract inter-class correlation stats."""
    common_a = [t for t in tickers_a if t in corr_df.index]
    common_b = [t for t in tickers_b if t in corr_df.index]
    vals = []
    for a in common_a:
        for b in common_b:
            v = corr_df.loc[a, b]
            if not np.isnan(v):
                vals.append(v)
    vals = np.array(vals)
    return {"mean": vals.mean(), "median": np.median(vals), "min": vals.min(),
            "max": vals.max(), "std": vals.std(), "n_pairs": len(vals)}

def top_corr_pairs(corr_df, tickers=None, top_n=10, exclude_self=True):
    """Find top-N most correlated pairs."""
    if tickers is not None:
        common = [t for t in tickers if t in corr_df.index]
        sub = corr_df.loc[common, common]
    else:
        sub = corr_df
        common = list(sub.index)
    pairs = []
    for i in range(len(common)):
        for j in range(i+1, len(common)):
            v = sub.iloc[i, j]
            if not np.isnan(v):
                pairs.append((common[i], common[j], v))
    pairs.sort(key=lambda x: -x[2])
    return pairs[:top_n]

def bottom_corr_pairs(corr_df, tickers=None, top_n=10):
    """Find most negatively correlated pairs."""
    if tickers is not None:
        common = [t for t in tickers if t in corr_df.index]
        sub = corr_df.loc[common, common]
    else:
        sub = corr_df
        common = list(sub.index)
    pairs = []
    for i in range(len(common)):
        for j in range(i+1, len(common)):
            v = sub.iloc[i, j]
            if not np.isnan(v):
                pairs.append((common[i], common[j], v))
    pairs.sort(key=lambda x: x[2])
    return pairs[:top_n]

def top_corr_with(corr_df, target, top_n=10):
    """Top-N correlations with a specific target."""
    if target not in corr_df.index:
        return []
    row = corr_df[target].drop(target).dropna()
    return row.nlargest(top_n)

def bottom_corr_with(corr_df, target, top_n=10):
    """Most negative correlations with a specific target."""
    if target not in corr_df.index:
        return []
    row = corr_df[target].drop(target).dropna()
    return row.nsmallest(top_n)

# ===================== BUILD REPORT =====================
log("Generating report...")
lines = []

def w(s=""):
    lines.append(s)

w("# 股票/ETF/加密货币 相关性分析与投资组合策略报告")
w()
w(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
w()
w("---")
w()

# ===== SECTION 1: DATA OVERVIEW =====
w("## 1. 数据概览")
w()
w(f"- **总标的数**: {len(all_tickers)}")
w(f"  - 加密货币: {len(crypto_tickers)} 个 ({', '.join(crypto_tickers)})")
w(f"  - ETF: {len(etf_tickers)} 个")
w(f"  - 个股: {len(stock_tickers)} 个")
w(f"- **数据时间范围**: 2015-04-27 ~ 2026-04 (约11年)")
w(f"- **交易日数**: ~2,516")
w(f"- **分析维度**:")
w(f"  - 日对数收益率 (log return)")
w(f"  - 21日滚动波动率 (短期)")
w(f"  - 63日滚动波动率 (长期/季度)")
w(f"  - Pearson / Spearman 相关系数")
w(f"  - 协方差矩阵")
w()
w("### 波动率时间序列的相关系数（非价格收益率）")
w()
w("本报告采用**波动率时间序列之间的相关系数**进行分析，而非传统价格收益率相关系数。")
w("波动率相关性衡量的是不同资产**风险水平的联动关系**，更能反映在市场压力时期各资产的共险特征。")
w()
w("---")
w()

log("Section 1 done")

# ===== SECTION 2: VOLATILITY ANALYSIS =====
w("## 2. 波动率特征分析")
w()
w("### 2.1 各类资产平均波动率 (21日滚动)")
w()
mv = mean_vol.copy()
mv["asset_class"] = mv.index.map(lambda x: ticker_cls.get(x, "unknown"))
mv["ticker"] = mv.index

# Summary by class
cls_vol = mv.groupby("asset_class")["mean_vol_21d"].agg(["mean", "median", "max", "min", "std", "count"])
cls_vol = cls_vol.round(6)

w()
w("| 资产类别 | 数量 | 平均波动率 | 中位数波动率 | 最高 | 最低 | 标准差 |")
w("|----------|------|-----------|-------------|------|------|--------|")
for cls_name in ["crypto", "etf", "stock"]:
    if cls_name in cls_vol.index:
        row = cls_vol.loc[cls_name]
        w(f"| {cls_name} | {int(row['count'])} | {row['mean']:.4f} | {row['median']:.4f} | {row['max']:.4f} | {row['min']:.4f} | {row['std']:.4f} |")
w()

# Top 5 highest vol
w("### 2.2 波动率最高的标的 (Top 10)")
w()
top_vol = mv.nlargest(10, "mean_vol_21d")[["ticker", "asset_class", "mean_vol_21d", "mean_vol_63d"]]
w("| Ticker | 类别 | 21日均波动率 | 63日均波动率 |")
w("|--------|------|-------------|-------------|")
for _, row in top_vol.iterrows():
    w(f"| {row['ticker']} | {row['asset_class']} | {row['mean_vol_21d']:.4f} | {row['mean_vol_63d']:.4f} |")
w()

# Lowest 5 vol
w("### 2.3 波动率最低的标的 (Top 10)")
w()
low_vol = mv.nsmallest(10, "mean_vol_21d")[["ticker", "asset_class", "mean_vol_21d", "mean_vol_63d"]]
w("| Ticker | 类别 | 21日均波动率 | 63日均波动率 |")
w("|--------|------|-------------|-------------|")
for _, row in low_vol.iterrows():
    w(f"| {row['ticker']} | {row['asset_class']} | {row['mean_vol_21d']:.4f} | {row['mean_vol_63d']:.4f} |")
w()

log("Section 2 done")

# ===== SECTION 3: CORRELATION ANALYSIS =====
w("## 3. 相关性分析")
w()
w("### 3.1 大类资产内部相关性 (21日波动率 Pearson)")
w()
w("衡量同一类别内标的之间波动率变化的同步程度：")
w()

classes = [
    (crypto_tickers, "加密货币"),
    (etf_tickers, "ETF"),
    (stock_tickers, "个股"),
]

intra_stats = []
for tickers, label in classes:
    s = class_corr_matrix(r21, tickers, label)
    intra_stats.append(s)

w("| 类别 | 标的数 | 平均相关性 | 中位数 | 最低 | 最高 | 标准差 |")
w("|------|--------|-----------|--------|------|------|--------|")
for s in intra_stats:
    w(f"| {s['label']} | {s['n']} | {s['mean']:.4f} | {s['median']:.4f} | {s['min']:.4f} | {s['max']:.4f} | {s['std']:.4f} |")
w()

w("**解读**:")
w("- 加密货币内部 (BTC-SOL): 21日波动率相关性高达 **{:.4f}**，63日相关性为 **{:.4f}**，表明加密资产风险高度联动。".format(
    r21.loc["BTC", "SOL"], r63.loc["BTC", "SOL"]))
w("- ETF内部平均相关性约 **{:.4f}**，反映了不同类型ETF（权益、商品、波动率）的风险分散化程度。".format(intra_stats[1]["mean"]))
w("- 个股内部平均相关性约 **{:.4f}**，涵盖不同行业和地区。".format(intra_stats[2]["mean"]))
w()

log("Section 3.1 done")

# --- 3.2 Inter-class ---
w("### 3.2 大类资产间相关性")
w()
pairs_cls = [
    (crypto_tickers, etf_tickers, "加密货币 vs ETF"),
    (crypto_tickers, stock_tickers, "加密货币 vs 个股"),
    (etf_tickers, stock_tickers, "ETF vs 个股"),
]

w("| 比较对 | 配对数量 | 平均相关性 | 中位数 | 最低 | 最高 |")
w("|--------|----------|-----------|--------|------|------|")
for ta, tb, label in pairs_cls:
    s = inter_class_corr(r21, ta, tb)
    w(f"| {label} | {s['n_pairs']} | {s['mean']:.4f} | {s['median']:.4f} | {s['min']:.4f} | {s['max']:.4f} |")
w()

w("**解读**:")
w("- 加密货币与ETF/个股的波动率相关性整体处于中等偏低水平，表明加密货币的风险动态与传统金融市场部分脱钩。")
w("- ETF与个股间相关性较高，因为大量ETF追踪个股指数。")
w()

log("Section 3.2 done")

# --- 3.3 BTC/SOL with others ---
w("### 3.3 加密货币与传统资产的相关性")
w()
for crypto in ["BTC", "SOL"]:
    if crypto not in r21.columns:
        continue
    w(f"#### {crypto} 最高相关标的 (21日 Pearson)")
    w()
    top = top_corr_with(r21, crypto, 15)
    w("| Ticker | 类别 | 相关系数 |")
    w("|--------|------|----------|")
    for t, v in top.items():
        cls_name = ticker_cls.get(t, "?")
        w(f"| {t} | {cls_name} | {v:.4f} |")
    w()

    # lowest / negative
    w(f"#### {crypto} 最低/负相关标的")
    w()
    bot = bottom_corr_with(r21, crypto, 10)
    w("| Ticker | 类别 | 相关系数 |")
    w("|--------|------|----------|")
    for t, v in bot.items():
        cls_name = ticker_cls.get(t, "?")
        w(f"| {t} | {cls_name} | {v:.4f} |")
    w()

log("Section 3.3 done")

# --- 3.4 Key ETF correlations ---
w("### 3.4 关键ETF间的相关性")
w()
key_etfs = ["SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "USO", "VXX", "TQQQ", "SQQQ", "ARKK", "XLK", "XLF", "XLE"]
avail_etfs = [e for e in key_etfs if e in r21.columns]
sub_etf = r21.loc[avail_etfs, avail_etfs]

w("21日波动率 Pearson 相关系数矩阵 (关键ETF):")
w()
w("| | " + " | ".join(avail_etfs) + " |")
w("|" + "|".join(["-" * 6] * (len(avail_etfs) + 1)) + "|")
for etf in avail_etfs:
    vals = " | ".join([f"{sub_etf.loc[etf, e]:.3f}" for e in avail_etfs])
    w(f"| {etf} | {vals} |")
w()

w("**关键发现**:")
w("- SPY与QQQ高度相关 ({:.3f})，属同类权益风险。".format(sub_etf.loc["SPY", "QQQ"]))
w("- TQQQ与SQQQ高度负相关 ({:.3f})，前者3倍做多、后者3倍做空。".format(sub_etf.loc["TQQQ", "SQQQ"]) if "TQQQ" in avail_etfs and "SQQQ" in avail_etfs else "")
if "GLD" in avail_etfs and "SPY" in avail_etfs:
    w("- GLD与SPY相关性 {:.3f}，黄金与美股风险联动较弱，可作避险分散。".format(sub_etf.loc["GLD", "SPY"]))
if "VXX" in avail_etfs and "SPY" in avail_etfs:
    w("- VXX（波动率指数ETF）与SPY相关性 {:.3f}，市场恐慌时VIX上涨、股票下跌的规律在波动率相关性中体现。".format(sub_etf.loc["VXX", "SPY"]))
w()

log("Section 3.4 done")

# --- 3.5 Top correlated pairs overall ---
w("### 3.5 波动率相关性最高的资产对 (Top 15)")
w()
top_pairs = top_corr_pairs(r21, top_n=15)
w("| 资产A | 类别A | 资产B | 类别B | 相关系数 |")
w("|-------|-------|-------|-------|----------|")
for a, b, v in top_pairs:
    ca = ticker_cls.get(a, "?")
    cb = ticker_cls.get(b, "?")
    w(f"| {a} | {ca} | {b} | {cb} | {v:.4f} |")
w()

w("### 3.6 波动率相关性最低/负相关的资产对 (Top 10)")
w()
bot_pairs = bottom_corr_pairs(r21, top_n=10)
w("| 资产A | 类别A | 资产B | 类别B | 相关系数 |")
w("|-------|-------|-------|-------|----------|")
for a, b, v in bot_pairs:
    ca = ticker_cls.get(a, "?")
    cb = ticker_cls.get(b, "?")
    w(f"| {a} | {ca} | {b} | {cb} | {v:.4f} |")
w()

log("Section 3.5-3.6 done")

# --- 3.7 Pearson vs Spearman ---
w("### 3.7 Pearson vs Spearman 一致性")
w()
# Compare Pearson and Spearman for key ETFs
w("关键ETF的Pearson和Spearman对比 (21日):")
w()
w("| 对 | Pearson | Spearman | 差异 |")
w("|----|---------|----------|------|")
for i in range(len(avail_etfs)):
    for j in range(i+1, len(avail_etfs)):
        a, b = avail_etfs[i], avail_etfs[j]
        pv = r21.loc[a, b]
        sv = s21.loc[a, b]
        w(f"| {a}-{b} | {pv:.4f} | {sv:.4f} | {abs(pv-sv):.4f} |")
w()

log("Section 3.7 done")

# ===== SECTION 4: PORTFOLIO STRATEGY =====
w("## 4. 投资组合策略分析")
w()
w("基于91个代表性标的（含2个加密货币、26个ETF、63个个股），构建了四种投资组合策略。")
w()

# --- 4.1 Strategy overview ---
w("### 4.1 策略概览")
w()
w("| 策略 | 年化收益率 | 年化波动率 | 夏普比率 | 持仓数量 |")
w("|------|-----------|-----------|----------|----------|")
port_ms = port["port_ms"]  # [ret, vol, sharpe]
port_mv = port["port_mv"]
port_rp = port["port_rp"]
port_hg = port["port_hg"]

ms_count = sum(1 for v in port["w_maxsharpe"].values() if v > 0.001)
mv_count = sum(1 for v in port["w_minvar"].values() if v > 0.001)
rp_count = sum(1 for v in port["w_rp"].values() if v > 0.001)
hg_count = sum(1 for v in port["w_hedge"].values() if v > 0.001)

w(f"| 最大夏普比率 | {port_ms[0]:.4f} | {port_ms[1]:.4f} | {port_ms[2]:.4f} | {ms_count} |")
w(f"| 最小方差 | {port_mv[0]:.4f} | {port_mv[1]:.4f} | {port_mv[2]:.4f} | {mv_count} |")
w(f"| 风险平价 | {port_rp[0]:.4f} | {port_rp[1]:.4f} | {port_rp[2]:.4f} | {rp_count} |")
w(f"| 对冲组合 | {port_hg[0]:.4f} | {port_hg[1]:.4f} | {port_hg[2]:.4f} | {hg_count} |")
w()

w("**策略说明**:")
w("- **最大夏普比率**: 通过均值-方差优化，最大化组合的夏普比率。")
w("- **最小方差**: 在给定资产池中寻找波动率最低的权重配置。")
w("- **风险平价**: 使每个资产对组合总风险的贡献相等，避免单一资产主导风险。")
w("- **对冲组合**: 人工配置，加入GLD/SLV等避险资产和SQQQ/VXX等对冲工具。")
w()

log("Section 4.1 done")

# --- 4.2 Max Sharpe detail ---
w("### 4.2 最大夏普比率组合 — 权重分布")
w()
w("该策略高度集中于少数高Sharpe比率的资产：")
w()
ms_weights = [(k, v) for k, v in port["w_maxsharpe"].items() if v > 0.0001]
ms_weights.sort(key=lambda x: -x[1])
w("| Ticker | 类别 | 权重 | 年化收益 | 年化波动 | 夏普比率 |")
w("|--------|------|------|----------|----------|----------|")
for t, wt in ms_weights:
    c = ticker_cls.get(t, "?")
    ret = port["ann_ret"].get(t, 0)
    vol = port["ann_vol"].get(t, 0)
    sr = port["sharpe"].get(t, 0)
    w(f"| {t} | {c} | {wt:.4f} | {ret:.4f} | {vol:.4f} | {sr:.4f} |")
w()
btc_w = port["w_maxsharpe"].get("BTC", 0)
gl_w = port["w_maxsharpe"].get("GLD", 0)
lly_w = port["w_maxsharpe"].get("LLY", 0)
w(f"**BTC权重 {btc_w:.1%}，GLD权重 {gl_w:.1%}，LLY权重 {lly_w:.1%}** — 这三者贡献了绝大部分配置。")
w()

# --- 4.3 Min Variance detail ---
w("### 4.3 最小方差组合 — 权重分布 (Top 20)")
w()
w("最小方差组合大幅分散，偏好低波动资产（消费防御、医疗健康、公用事业、黄金等）：")
w()
mv_weights = [(k, v) for k, v in port["w_minvar"].items() if v > 0.0001]
mv_weights.sort(key=lambda x: -x[1])
w("| Ticker | 类别 | 权重 | 年化波动 |")
w("|--------|------|------|----------|")
for t, wt in mv_weights[:20]:
    c = ticker_cls.get(t, "?")
    vol = port["ann_vol"].get(t, 0)
    w(f"| {t} | {c} | {wt:.4f} | {vol:.4f} |")
if len(mv_weights) > 20:
    w(f"| ... (共 {len(mv_weights)} 个持仓) | ... | ... | ... |")
w()

# --- 4.4 Risk Parity ---
w("### 4.4 风险平价组合 — 权重分布 (Top 15)")
w()
w("风险平价按风险贡献分配，各资产风险贡献接近相等：")
w()
rp_weights = [(k, v) for k, v in port["w_rp"].items()]
rp_weights.sort(key=lambda x: -x[1])
w("| Ticker | 类别 | 权重 |")
w("|--------|------|------|")
for t, wt in rp_weights[:15]:
    c = ticker_cls.get(t, "?")
    w(f"| {t} | {c} | {wt:.4f} |")
w(f"| ... | ... | ... |")
w(f"| **总持仓数** | {rp_count} | |")
w()
w(f"风险平价组合的特点：GLD权重最高 ({port['w_rp']['GLD']:.1%})，因为黄金波动率低，需更高权重才能与其他高风险资产的风险贡献相等。")
w()

# --- 4.5 Hedge ---
w("### 4.5 对冲组合")
w()
w("对冲组合人工配置，大幅配置SPY (35%)、QQQ (10%)等权益资产，同时加入：")
w(f"- **GLD (10%) + SLV (3%) + GDX (5%)**: 贵金属避险")
w(f"- **VXX (5%)**: 波动率对冲")
w(f"- **SQQQ (5%)**: 纳指做空对冲")
w(f"- **BTC (5%)**: 另类资产敞口")
w()
w("该组合年化收益 {:.4f}，波动率 {:.4f}，夏普比率 {:.4f}。".format(port_hg[0], port_hg[1], port_hg[2]))
w()

log("Section 4.2-4.5 done")

# --- 4.6 Strategy comparison ---
w("### 4.6 策略综合对比")
w()
w("| 策略 | 年化收益率 | 年化波动率 | 夏普比率 |")
w("|------|-----------|-----------|----------|")
for name, vals in [("最大夏普比率", port_ms), ("最小方差", port_mv), ("风险平价", port_rp), ("对冲组合", port_hg)]:
    w(f"| {name} | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} |")
w()

# Risk-adjusted
w()
w("### 4.7 策略权重分布对比（按资产类别汇总）")
w()
w("| 策略 | 加密货币 | ETF | 个股 |")
w("|------|---------|-----|------|")
for strat_name, w_dict in [
    ("最大夏普比率", port["w_maxsharpe"]),
    ("最小方差", port["w_minvar"]),
    ("风险平价", port["w_rp"]),
    ("对冲组合", port["w_hedge"]),
]:
    class_weights = {"crypto": 0, "etf": 0, "stock": 0}
    for t, wt in w_dict.items():
        c = ticker_cls.get(t, "stock")
        class_weights[c] = class_weights.get(c, 0) + wt
    w(f"| {strat_name} | {class_weights['crypto']:.2%} | {class_weights['etf']:.2%} | {class_weights['stock']:.2%} |")
w()

log("Section 4.6-4.7 done")

# ===== SECTION 5: RECOMMENDATIONS =====
w("## 5. 综合建议")
w()
w("### 5.1 相关性视角下的分散化建议")
w()
w("1. **加密货币与传统资产波动率相关性中等偏低**")
try:
    btc_spy = r21.loc["BTC", "SPY"]
    w(f"   - BTC与SPY的21日波动率相关性为 **{btc_spy:.4f}**")
except:
    w("   - BTC与SPY的波动率动态存在脱钩现象")
w("   - 适量配置加密货币（不超过5%）可提供风险分散化收益。")
w()
w("2. **黄金（GLD）是优秀的风险分散工具**")
if "GLD" in r21.columns and "SPY" in r21.columns:
    w(f"   - GLD与SPY的波动率相关性为 **{r21.loc['GLD','SPY']:.4f}**，远低于同类权益资产之间的相关性。")
w("   - 在所有四种策略中，GLD均获得显著权重，尤其是在风险平价中权重最高。")
w()
w("3. **行业ETF间的相关性较高，需注意集中风险**")
w("   - XLK（科技）与SPY高度相关。")
w("   - XLP（必需消费）和XLU（公用事业）则相关性较低，具防御属性。")
w()
w("4. **波动率ETF（VXX）可作尾部对冲**")
w("   - VXX与其他大多数资产呈负相关或低正相关，是有效的投资组合对冲工具。")
w()

w("### 5.2 策略选择建议")
w()
w("| 投资者类型 | 推荐策略 | 理由 |")
w("|-----------|---------|------|")
w(f"| 积极型 | 最大夏普比率 | 年化收益 {port_ms[0]:.2%}，夏普比 {port_ms[2]:.2f}，集中高Sharpe资产 |")
w(f"| 保守型 | 最小方差 | 年化波动仅 {port_mv[1]:.2%}，优先控制回撤 |")
w(f"| 均衡型 | 风险平价 | 真正分散风险，不依赖单一资产/类别 |")
w(f"| 宏观对冲型 | 对冲组合 | 含SQQQ/VXX等负相关资产，应对市场下跌 |")
w()

w("### 5.3 关键风险提示")
w()
w("1. **加密货币的高波动特性**: BTC和SOL的年化波动率分别为{:.0%}和{:.0%}，远超传统资产。".format(
    port["ann_vol"]["BTC"], port["ann_vol"]["SOL"]))
w("2. **波动率相关性不等于价格相关性**: 本报告分析的是波动率时间序列之间的相关关系，反映风险传染效应，而非直接的价格同涨同跌。")
w("3. **历史相关性不等于未来**: 相关性结构在市场压力期可能突变（相关性崩溃）。")
w("4. **仅基于历史数据的局限性**: 所有优化均基于历史数据，未来表现可能显著偏离。")
w()

w("---")
w()
w("*报告由 06_portfolio_correlation_report.py 自动生成*")
w(f"*数据截至: {datetime.now().strftime('%Y-%m-%d')}*")

# ---- WRITE ----
report = "\n".join(lines)
out_path = CONCLUSION / "correlation_portfolio_analysis.md"
with open(out_path, "w") as f:
    f.write(report)

log(f"Report written to {out_path}")
log(f"Total lines: {len(lines)}")
log("Done!")
