"""
================================================================
  跨行业投资组合 - FinRL + A2C 完整版

  特点：
    - 60 只股票，覆盖 GICS 11 大行业
    - Yahoo Finance 数据源（截止 2026-04-27）
    - 特征：8 技术指标 + VIX + Turbulence + 协方差矩阵 + 7 个 HV 波动率
    - A2C 算法，40 万步训练
    - 完整回测 + 与 S&P 500 对比 + 行业权重分析

  作者：Chen SiYuan
================================================================
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.plot import backtest_stats, get_baseline

# ================================================================
# 1. 全局配置
# ================================================================
SECTOR_PORTFOLIO = {
    "科技 (Tech)":           ["AAPL", "MSFT", "NVDA", "GOOGL", "META",
                              "AMD", "CRM", "ADBE", "ORCL", "CSCO"],
    "金融 (Financials)":     ["JPM", "BAC", "GS", "MS", "BLK",
                              "V", "MA", "AXP"],
    "医疗 (Healthcare)":     ["UNH", "JNJ", "LLY", "PFE", "ABBV",
                              "TMO", "ABT"],
    "消费 (Consumer Disc.)": ["AMZN", "TSLA", "HD", "MCD", "NKE",
                              "SBUX", "TGT", "LOW"],
    "必需消费 (Staples)":    ["WMT", "PG", "KO", "PEP", "COST"],
    "能源 (Energy)":         ["XOM", "CVX", "COP", "SLB"],
    "工业 (Industrials)":    ["CAT", "BA", "HON", "UPS", "GE"],
    "通讯 (Communication)":  ["NFLX", "DIS", "VZ", "T"],
    "公用事业 (Utilities)":  ["NEE", "DUK", "SO"],
    "材料 (Materials)":      ["LIN", "SHW", "FCX"],
    "地产 (Real Estate)":    ["AMT", "PLD", "EQIX"],
}

TICKERS = list(dict.fromkeys(
    [t for sec in SECTOR_PORTFOLIO.values() for t in sec]
))
TICKER_TO_SECTOR = {t: s for s, ts in SECTOR_PORTFOLIO.items() for t in ts}

# ⭐ 时间窗口（截止 2026-04-27）
TRAIN_START = "2015-01-01"
TRAIN_END   = "2024-12-31"
TRADE_START = "2025-01-01"
TRADE_END   = "2026-04-27"

# 基础技术指标
INDICATORS = ["macd", "rsi_30", "cci_30", "dx_30",
              "boll_ub", "boll_lb", "close_30_sma", "close_60_sma"]

# 训练参数
TOTAL_TIMESTEPS  = 400_000
INITIAL_AMOUNT   = 1_000_000
TRANSACTION_COST = 0.001
HMAX             = 100

A2C_PARAMS = {
    "n_steps": 10,
    "ent_coef": 0.005,
    "learning_rate": 0.0004,
}

os.makedirs("data", exist_ok=True)
os.makedirs("trained_models", exist_ok=True)

print("=" * 60)
print(f"📊 股票池：{len(TICKERS)} 只，覆盖 {len(SECTOR_PORTFOLIO)} 个行业")
print(f"📅 数据窗口：{TRAIN_START} ~ {TRADE_END}")
print("=" * 60)

# ================================================================
# 2. 数据下载（Yahoo Finance，带本地缓存）
# ================================================================
DATA_CACHE = f"data/raw_60stocks_{TRADE_END}.csv"

if os.path.exists(DATA_CACHE):
    print(f"\n📥 加载本地缓存：{DATA_CACHE}")
    df = pd.read_csv(DATA_CACHE)
else:
    print(f"\n📥 从 Yahoo Finance 下载（截止 {TRADE_END}）...")
    df = YahooDownloader(
        start_date=TRAIN_START,
        end_date=TRADE_END,
        ticker_list=TICKERS
    ).fetch_data()
    df.to_csv(DATA_CACHE, index=False)
    print(f"   保存到 {DATA_CACHE}")

print(f"   数据形状：{df.shape}")
print(f"   实际股票数：{df['tic'].nunique()}")
print(f"   日期范围：{df['date'].min()} ~ {df['date'].max()}")

if df['date'].max() < "2026-04-25":
    print(f"⚠️  警告：最新数据只到 {df['date'].max()}，建议删除缓存重跑")

# ================================================================
# 3. 特征工程：技术指标 + VIX + Turbulence
# ================================================================
print("\n🔧 特征工程：技术指标 + VIX + Turbulence...")
fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=INDICATORS,
    use_turbulence=True,
    use_vix=True,
)
processed = fe.preprocess_data(df)

# ================================================================
# 3.5 ⭐ 已实现波动率（HV）—— 隐含波动率的免费代理
# ================================================================
print("\n📈 计算已实现波动率（HV）...")

def add_realized_vol(df, windows=(5, 10, 20, 60)):
    """
    给每只股票添加 7 个波动率特征：
      hv_5/10/20/60      多窗口年化波动率
      hv_term_structure  短/长期波动率比值（曲面斜率）
      hv_change          波动率一阶差分
      hv_vix_spread      个股 vs 市场紧张度
    """
    df = df.sort_values(["tic", "date"]).copy()

    # 对数收益率
    df["log_ret"] = df.groupby("tic")["close"].apply(
        lambda x: np.log(x / x.shift(1))
    ).reset_index(level=0, drop=True)

    # 多窗口年化 HV
    for w in windows:
        df[f"hv_{w}"] = (
            df.groupby("tic")["log_ret"]
              .rolling(w).std()
              .reset_index(level=0, drop=True)
            * np.sqrt(252)
        )

    # 期限结构
    df["hv_term_structure"] = df["hv_10"] / df["hv_60"]

    # 波动率变化
    df["hv_change"] = df.groupby("tic")["hv_20"].diff()

    # HV-VIX 价差
    if "vix" in df.columns:
        df["hv_vix_spread"] = df["hv_20"] - df["vix"] / 100

    df = df.drop(columns=["log_ret"])
    return df.sort_values(["date", "tic"]).reset_index(drop=True)

processed = add_realized_vol(processed)

# 处理 NaN
hv_cols = ["hv_5", "hv_10", "hv_20", "hv_60",
           "hv_term_structure", "hv_change", "hv_vix_spread"]
processed[hv_cols] = (
    processed.groupby("tic")[hv_cols]
             .ffill()
             .fillna(0)
)

# ⭐ 把 HV 特征加入 INDICATORS
INDICATORS = INDICATORS + hv_cols
print(f"✅ 添加 {len(hv_cols)} 个波动率特征，总指标数：{len(INDICATORS)}")

# ================================================================
# 3.7 协方差矩阵（组合分配的灵魂）
# ================================================================
print("\n🧮 计算 252 日滚动协方差矩阵...")
processed = processed.sort_values(["date", "tic"]).reset_index(drop=True)
processed.index = processed.date.factorize()[0]

cov_list, return_list = [], []
lookback = 252
for i in range(lookback, len(processed.index.unique())):
    sub = processed.loc[i - lookback : i, :]
    px = sub.pivot_table(index="date", columns="tic", values="close")
    rets = px.pct_change().dropna()
    cov_list.append(rets.cov().values)
    return_list.append(rets)

df_cov = pd.DataFrame({
    "date": processed.date.unique()[lookback:],
    "cov_list": cov_list,
    "return_list": return_list
})
processed = (
    processed.merge(df_cov, on="date")
             .sort_values(["date", "tic"])
             .reset_index(drop=True)
)
print(f"   特征工程后：{processed.shape}")

# ================================================================
# 3.9 验证特征生效
# ================================================================
print("\n🔍 特征验证（NVDA 最新 5 天）：")
sample = processed[processed.tic == "NVDA"].tail(5)
print(sample[["date", "close", "vix", "turbulence",
              "hv_20", "hv_term_structure", "hv_vix_spread"]]
      .to_string(index=False))
print(f"\n   协方差矩阵形状：{processed['cov_list'].iloc[0].shape}")

# ================================================================
# 4. 切分训练/测试集
# ================================================================
train = data_split(processed, TRAIN_START, TRAIN_END)
trade = data_split(processed, TRADE_START, TRADE_END)
print(f"\n📂 训练集：{train.shape} | 测试集：{trade.shape}")
print(f"   训练日期：{train.date.min()} ~ {train.date.max()}")
print(f"   测试日期：{trade.date.min()} ~ {trade.date.max()}")

# ================================================================
# 5. 构建组合分配环境
# ================================================================
stock_dim = len(train.tic.unique())
env_kwargs = dict(
    hmax=HMAX,
    initial_amount=INITIAL_AMOUNT,
    transaction_cost_pct=TRANSACTION_COST,
    state_space=stock_dim,
    stock_dim=stock_dim,
    tech_indicator_list=INDICATORS,
    action_space=stock_dim,
    reward_scaling=1e-4,
)
e_train = StockPortfolioEnv(df=train, **env_kwargs)
env_train, _ = e_train.get_sb_env()
print(f"\n🌍 环境：股票数={stock_dim}, 状态特征数={len(INDICATORS)}")

# ================================================================
# 6. 训练 A2C Agent
# ================================================================
print(f"\n🤖 训练 A2C，{TOTAL_TIMESTEPS:,} 步（约 30-90 分钟）...")
agent = DRLAgent(env=env_train)
model_a2c = agent.get_model("a2c", model_kwargs=A2C_PARAMS)

trained_a2c = agent.train_model(
    model=model_a2c,
    tb_log_name="a2c_60stocks_hv",
    total_timesteps=TOTAL_TIMESTEPS,
)
trained_a2c.save("trained_models/a2c_60stocks_hv")
print("✅ 模型已保存到 trained_models/a2c_60stocks_hv.zip")

# ================================================================
# 7. 样本外回测
# ================================================================
print(f"\n📈 样本外回测（{TRADE_START} ~ {TRADE_END}）...")
e_trade = StockPortfolioEnv(df=trade, **env_kwargs)
df_value, df_actions = DRLAgent.DRL_prediction(trained_a2c, e_trade)

df_value.to_csv("data/portfolio_value.csv", index=False)
df_actions.to_csv("data/portfolio_actions.csv")

# ================================================================
# 8. 绩效评估
# ================================================================
print("\n" + "=" * 60)
print("📊 A2C 跨行业组合 绩效统计")
print("=" * 60)
stats_a2c = backtest_stats(account_value=df_value,
                            value_col_name="account_value")
print(stats_a2c)

print("\n" + "=" * 60)
print("📊 S&P 500 基准 绩效统计")
print("=" * 60)
baseline = get_baseline(ticker="^GSPC", start=TRADE_START, end=TRADE_END)
stats_baseline = backtest_stats(baseline, value_col_name="close")
print(stats_baseline)

# ================================================================
# 9. 可视化：组合 vs 基准
# ================================================================
print("\n📉 生成图表...")
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

ax1 = axes[0]
ax1.plot(pd.to_datetime(df_value["date"]),
         df_value["account_value"] / df_value["account_value"].iloc[0],
         label="A2C 组合", linewidth=2, color="#2E86AB")
ax1.plot(pd.to_datetime(baseline["date"]),
         baseline["close"] / baseline["close"].iloc[0],
         label="S&P 500", linewidth=2, linestyle="--", color="#E63946")
ax1.set_title(f"跨行业 60 股 A2C 组合 vs S&P 500（{TRADE_START} ~ {TRADE_END}）",
              fontsize=14)
ax1.set_xlabel("日期")
ax1.set_ylabel("累计收益（归一化）")
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3)

ax2 = axes[1]
for tic in ["NVDA", "AAPL", "JPM", "XOM", "JNJ"]:
    sub = processed[processed.tic == tic].sort_values("date")
    sub = sub[sub["date"] >= TRADE_START]
    ax2.plot(pd.to_datetime(sub["date"]), sub["hv_20"],
             label=tic, alpha=0.8)
ax2.set_title("代表股票 20 日已实现波动率（HV_20）", fontsize=14)
ax2.set_xlabel("日期")
ax2.set_ylabel("年化波动率")
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("data/performance_curve.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ 图表已保存到 data/performance_curve.png")

# ================================================================
# 10. 行业权重分析
# ================================================================
print("\n" + "=" * 60)
print("📊 平均行业权重分布")
print("=" * 60)

avg_weights = df_actions.mean(axis=0)
sector_weights = {}
for tic, w in avg_weights.items():
    sec = TICKER_TO_SECTOR.get(tic, "Unknown")
    sector_weights[sec] = sector_weights.get(sec, 0) + w

sector_df = pd.DataFrame(
    sorted(sector_weights.items(), key=lambda x: -x[1]),
    columns=["行业", "平均权重"]
)
sector_df["占比 %"] = (sector_df["平均权重"] * 100).round(2)
print(sector_df.to_string(index=False))
sector_df.to_csv("data/sector_weights.csv", index=False)

fig2, ax3 = plt.subplots(figsize=(10, 8))
ax3.pie(sector_df["平均权重"],
        labels=sector_df["行业"],
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 10})
ax3.set_title("A2C 学到的行业权重分布", fontsize=14)
plt.tight_layout()
plt.savefig("data/sector_pie.png", dpi=150, bbox_inches="tight")
plt.show()

# ================================================================
# 11. 单只股票权重 Top 10
# ================================================================
print("\n📊 单股权重 Top 10：")
top10 = avg_weights.sort_values(ascending=False).head(10)
for tic, w in top10.items():
    sec = TICKER_TO_SECTOR.get(tic, "?")
    print(f"   {tic:6s} ({sec:20s}): {w:.4f}  ({w*100:.2f}%)")

print("\n" + "=" * 60)
print("✅ 全部完成！")
print("=" * 60)
print(f"📁 输出文件：")
print(f"   data/raw_60stocks_{TRADE_END}.csv      原始数据")
print(f"   data/portfolio_value.csv               组合每日价值")
print(f"   data/portfolio_actions.csv             每日权重")
print(f"   data/sector_weights.csv                行业权重")
print(f"   data/performance_curve.png             收益曲线")
print(f"   data/sector_pie.png                    行业饼图")
print(f"   trained_models/a2c_60stocks_hv.zip    训练好的模型")