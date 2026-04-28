"""
data_fetcher.py
使用 CBOE 延迟数据 API 获取期权链，构建 volvisualizer precomputed_data。
CBOE API 可从中国大陆直接访问，无需代理。
"""

import re
import time
import logging
import traceback
from datetime import datetime, date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests
from scipy.optimize import brentq
from scipy.stats import norm

logger = logging.getLogger('download')

# ── HTTP session ──────────────────────────────────────────────────────────────
HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/124.0.0.0 Safari/537.36'
    ),
    'Accept': 'application/json,*/*',
}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)

# CBOE delayed quotes endpoint
CBOE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/{ticker}.json"


def _get_json(url: str, timeout: int = 20, retries: int = 3, wait: float = 3.0) -> Optional[dict]:
    for attempt in range(1, retries + 1):
        try:
            resp = SESSION.get(url, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
            logger.warning(f"HTTP {resp.status_code} | {url} | attempt {attempt}/{retries}")
        except Exception as e:
            logger.warning(f"Request error attempt {attempt}/{retries}: {e}")
        if attempt < retries:
            time.sleep(wait * attempt)
    return None


# ── Option symbol parser ──────────────────────────────────────────────────────
# Format: AAPL260516C00200000  (ticker + YYMMDD + C/P + strike*1000 zero-padded)
_OPT_RE = re.compile(r'^([A-Z\^]+)(\d{6})([CP])(\d{8})$')

def _parse_option_symbol(sym: str) -> Optional[dict]:
    m = _OPT_RE.match(sym)
    if not m:
        return None
    ticker, yymmdd, cp, strike_str = m.groups()
    expiry = datetime.strptime('20' + yymmdd, '%Y%m%d').date()
    strike = int(strike_str) / 1000.0
    opt_type = 'call' if cp == 'C' else 'put'
    return {'expiry': expiry, 'strike': strike, 'opt_type': opt_type}


# ── BS implied vol ────────────────────────────────────────────────────────────
def _bs_price(S: float, K: float, T: float, r: float, q: float,
              sigma: float, opt_type: str) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt_type == 'call':
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def _implied_vol(price: float, S: float, K: float, T: float,
                 r: float, q: float, opt_type: str) -> Optional[float]:
    if T <= 0 or price <= 0 or S <= 0 or K <= 0:
        return None
    try:
        iv = brentq(
            lambda sig: _bs_price(S, K, T, r, q, sig, opt_type) - price,
            1e-6, 10.0, xtol=1e-5, maxiter=200
        )
        return float(iv)
    except (ValueError, RuntimeError):
        return None


# ── 主入口 ────────────────────────────────────────────────────────────────────
def build_precomputed_data(
    ticker:        str,
    start_date:    str        = '2026-4-1',
    target_months: list[int]  = [4, 5, 6],
    target_year:   int        = 2026,
    r:             float      = 0.045,
    q:             float      = 0.0,
    wait:          float      = 1.0,
    monthlies:     bool       = True,
    min_volume:    int        = 0,
    min_oi:        int        = 0,
) -> Optional[pd.DataFrame]:
    """
    Fetch full option chain from CBOE, filter to target months,
    compute implied vols, return precomputed_data DataFrame for volvisualizer.
    """
    logger.info(f"[{ticker}] 开始从 CBOE 获取数据")
    url  = CBOE_URL.format(ticker=ticker)
    data = _get_json(url)
    if not data:
        logger.error(f"[{ticker}] CBOE 请求失败")
        return None

    body = data.get('data', {})
    spot = body.get('current_price') or body.get('close') or body.get('prev_day_close')
    if not spot:
        logger.error(f"[{ticker}] 无法获取现货价格")
        return None
    spot = float(spot)
    logger.info(f"[{ticker}] 现货价格 = {spot}")

    raw_options = body.get('options', [])
    logger.info(f"[{ticker}] CBOE 原始期权记录: {len(raw_options)}")

    # 统一日期解析，支持 '2026-4-1' / '2026/4/1' 等格式
    _sd = start_date.replace('/', '-')
    parts = _sd.split('-')
    start_dt = date(int(parts[0]), int(parts[1]), int(parts[2]))

    today = date.today()
    rows  = []

    for opt in raw_options:
        sym  = opt.get('option', '')
        info = _parse_option_symbol(sym)
        if not info:
            continue

        expiry   = info['expiry']
        strike   = info['strike']
        opt_type = info['opt_type']

        # 过滤：目标月份 & 年份
        if expiry.month not in target_months or expiry.year != target_year:
            continue
        if expiry < start_dt:
            continue

        # 过滤月度到期（第三个周五）
        if monthlies and not _is_monthly_expiry(expiry):
            continue

        bid  = float(opt.get('bid', 0) or 0)
        ask  = float(opt.get('ask', 0) or 0)
        last = float(opt.get('last_trade_price', 0) or 0)
        oi   = float(opt.get('open_interest', 0) or 0)
        vol  = float(opt.get('volume', 0) or 0)

        if vol < min_volume or oi < min_oi:
            continue

        mid_price    = (bid + ask) / 2 if (bid + ask) > 0 else last
        market_price = mid_price if mid_price > 0 else last

        T = (expiry - today).days / 365.0

        # 优先用 CBOE 提供的 IV，如 0 则自行计算
        cboe_iv = float(opt.get('iv', 0) or 0)
        if cboe_iv > 0.001:
            iv = cboe_iv
        else:
            iv = _implied_vol(market_price, spot, strike, T, r, q, opt_type)

        if iv is None or iv <= 0 or iv > 5:
            continue

        rows.append({
            'Contract Symbol':    sym,
            'Last Price':         last,
            'Bid':                bid,
            'Ask':                ask,
            'Last Trade Date':    pd.Timestamp(opt.get('last_trade_time') or datetime.now()),
            'Expiry':             expiry,
            'Strike':             strike,
            'Option Type':        opt_type,
            'Open Interest':      oi,
            'Volume':             vol,
            'Implied Volatility': iv,
            'Spot Price':         spot,
            'Discount Rate':      r,
            'Direct Discount Rate': r,
            'Smooth Discount Rate': r,
        })

    if not rows:
        # 如果月度过滤无结果，放宽
        logger.warning(f"[{ticker}] 月度筛选无结果，放宽为所有目标月份到期日")
        return build_precomputed_data(
            ticker=ticker, start_date=start_date,
            target_months=target_months, target_year=target_year,
            r=r, q=q, wait=wait, monthlies=False,
            min_volume=min_volume, min_oi=min_oi,
        ) if monthlies else None

    df = pd.DataFrame(rows)
    logger.info(f"[{ticker}] precomputed_data 构建完成: {len(df)} 行 | "
                f"到期日: {sorted(df['Expiry'].unique())}")
    return df


def _is_monthly_expiry(d: date) -> bool:
    """Return True if d is the 3rd Friday of its month."""
    if d.weekday() != 4:
        return False
    first = d.replace(day=1)
    fridays = 0
    cur = first
    while cur <= d:
        if cur.weekday() == 4:
            fridays += 1
        cur += timedelta(days=1)
    return fridays == 3
