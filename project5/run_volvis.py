"""
run_volvis.py — 隐含波动率研究主脚本
数据源: CBOE Delayed Quotes API（可从中国大陆访问）
研究对象: 7只科技股 + 其他行业龙头
到期月份: 2026年4月、5月、6月
"""

import os
import sys
import logging
import traceback
import warnings
import time
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ── 路径 ─────────────────────────────────────────────────────────────────────
BASE_DIR = '/root/autodl-tmp/CSY/project5'
LOG_DIR  = os.path.join(BASE_DIR, 'log')
DIR_2D   = os.path.join(BASE_DIR, 'charts', '2D')
DIR_3D   = os.path.join(BASE_DIR, 'charts', '3D')

for d in [LOG_DIR, DIR_2D, DIR_3D]:
    os.makedirs(d, exist_ok=True)

# ── 日志 ─────────────────────────────────────────────────────────────────────
def _make_logger(name: str, fname: str) -> logging.Logger:
    fmt = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(os.path.join(LOG_DIR, fname), encoding='utf-8')
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    lg = logging.getLogger(name)
    if not lg.handlers:
        lg.setLevel(logging.DEBUG)
        lg.addHandler(fh)
        lg.addHandler(sh)
    return lg

main_log  = _make_logger('main',     'main.log')
dl_log    = _make_logger('download', 'download.log')
chart_log = _make_logger('chart',    'chart.log')
err_log   = _make_logger('error',    'error.log')

# ── 标的 ─────────────────────────────────────────────────────────────────────
TECH_STOCKS = {
    'AAPL':  'Apple (科技)',
    'MSFT':  'Microsoft (科技)',
    'NVDA':  'NVIDIA (科技)',
    'GOOGL': 'Alphabet (科技)',
    'META':  'Meta (科技)',
    'AMZN':  'Amazon (科技)',
    'TSLA':  'Tesla (科技)',
}

OTHER_STOCKS = {
    'JPM':   '摩根大通 (金融)',
    'GS':    '高盛 (金融)',
    'JNJ':   '强生 (医疗)',
    'UNH':   '联合健康 (医疗)',
    'XOM':   '埃克森美孚 (能源)',
    'CVX':   '雪佛龙 (能源)',
    'WMT':   '沃尔玛 (消费)',
    'MCD':   '麦当劳 (消费)',
}

ALL_STOCKS = {**TECH_STOCKS, **OTHER_STOCKS}

# ── imports ───────────────────────────────────────────────────────────────────
sys.path.insert(0, BASE_DIR)
from data_fetcher import build_precomputed_data
from volvisualizer.volatility import Volatility


# ── 图表配置 ──────────────────────────────────────────────────────────────────
CHARTS_2D = [
    ('line',   'line',    {}),
]
CHARTS_3D = [
    ('scatter',        'scatter', {}),
    ('surface_spline', 'surface', {'surfacetype': 'spline', 'smoothing': True, 'scatter': True}),
    ('surface_mesh',   'surface', {'surfacetype': 'mesh',   'smoothing': True}),
]


def _save_chart(vol: Volatility, ticker: str, out_dir: str,
                suffix: str, graphtype: str, kwargs: dict):
    """调用 volvisualizer 内置 save_image，然后重命名为包含 suffix 的文件名。"""
    target = os.path.join(out_dir, f'{ticker}_{suffix}.png')
    chart_log.info(f"[{ticker}] 绘制 {suffix} -> {target}")
    try:
        vol.visualize(
            graphtype=graphtype,
            save_image=True,
            image_folder=out_dir,
            image_dpi=150,
            notebook=False,
            **kwargs,
        )
        # volvisualizer 用类似 "AAPL2026-4-1.png" 命名，找到并重命名
        candidates = sorted(glob.glob(os.path.join(out_dir, f'{ticker}*.png')))
        for c in candidates:
            if c != target:
                os.replace(c, target)
                break
        chart_log.info(f"[{ticker}] {suffix} 完成 -> {target}")
    except Exception as e:
        chart_log.error(f"[{ticker}] {suffix} 失败: {e}")
        err_log.error(f"[{ticker}] {suffix}:\n{traceback.format_exc()}")
    finally:
        plt.close('all')


def process_ticker(ticker: str, label: str) -> str:
    main_log.info(f"\n{'='*60}")
    main_log.info(f"处理: {ticker}  ({label})")
    main_log.info(f"{'='*60}")

    # 1. 获取数据
    dl_log.info(f"[{ticker}] 开始从 CBOE 获取期权数据")
    df = build_precomputed_data(
        ticker=ticker,
        start_date='2026-4-1',
        target_months=[4, 5, 6],
        target_year=2026,
        monthlies=False,   # CBOE 提供所有到期日，不强制月度
        r=0.045,
        q=0.0,
    )
    if df is None or len(df) == 0:
        dl_log.error(f"[{ticker}] 数据获取失败或为空")
        return 'FAILED'

    dl_log.info(f"[{ticker}] 数据获取成功: {len(df)} 行 | "
                f"到期日: {sorted(df['Expiry'].unique())}")

    # 保存原始数据
    data_path = os.path.join(BASE_DIR, 'data', f'{ticker}_options.csv')
    df.to_csv(data_path, index=False)
    dl_log.info(f"[{ticker}] 原始数据已保存: {data_path}")

    # 2. 初始化 Volatility
    try:
        spot = float(df['Spot Price'].iloc[0])
        vol  = Volatility(
            ticker=ticker,
            start_date='2026-4-1',
            precomputed_data=df,
            spot=spot,
        )
        main_log.info(f"[{ticker}] Volatility 初始化成功 | spot={spot:.2f}")
    except Exception as e:
        main_log.error(f"[{ticker}] Volatility 初始化失败: {e}")
        err_log.error(f"[{ticker}] Volatility init:\n{traceback.format_exc()}")
        return 'FAILED'

    # 3. 绘图
    td2 = os.path.join(DIR_2D, ticker)
    td3 = os.path.join(DIR_3D, ticker)
    os.makedirs(td2, exist_ok=True)
    os.makedirs(td3, exist_ok=True)

    for suffix, gtype, kw in CHARTS_2D:
        _save_chart(vol, ticker, td2, suffix, gtype, kw)

    for suffix, gtype, kw in CHARTS_3D:
        _save_chart(vol, ticker, td3, suffix, gtype, kw)

    main_log.info(f"[{ticker}] 全部图表完成")
    return 'OK'


def main():
    main_log.info("=" * 70)
    main_log.info("隐含波动率研究 启动")
    main_log.info(f"数据源: CBOE Delayed Quotes API")
    main_log.info(f"研究区间: 2026年4月/5月/6月 到期期权")
    main_log.info(f"科技股   ({len(TECH_STOCKS)}): {list(TECH_STOCKS.keys())}")
    main_log.info(f"其他行业 ({len(OTHER_STOCKS)}): {list(OTHER_STOCKS.keys())}")
    main_log.info("=" * 70)

    results: dict[str, str] = {}

    for ticker, label in ALL_STOCKS.items():
        status = process_ticker(ticker, label)
        results[ticker] = status
        time.sleep(2)  # 礼貌间隔

    # 汇总
    main_log.info("\n" + "=" * 70)
    main_log.info("运行结果汇总:")
    for t, s in results.items():
        icon = '✓' if s == 'OK' else '✗'
        main_log.info(f"  {icon} {t:<8}  {s}  ({ALL_STOCKS[t]})")
    ok   = sum(1 for v in results.values() if v == 'OK')
    fail = len(results) - ok
    main_log.info(f"\n  成功: {ok}  失败: {fail}  共: {len(results)}")
    main_log.info(f"\n  2D 图表: {DIR_2D}")
    main_log.info(f"  3D 图表: {DIR_3D}")
    main_log.info(f"  日志:    {LOG_DIR}")
    main_log.info("=" * 70)


if __name__ == '__main__':
    main()
