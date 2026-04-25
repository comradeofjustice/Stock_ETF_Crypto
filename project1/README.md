# TTP Stock/ETF/Crypto Volatility Analysis Project

A multi-asset quantitative analysis pipeline built on **Microsoft Qlib**, covering US stocks, ETFs, and cryptocurrencies.

## Overview

This project downloads 10 years of daily OHLCV data for stocks, ETFs, and cryptocurrencies, converts them to Qlib binary format, computes volatility features (ATR, intraday range, open-close spread), and generates interactive HTML reports with distribution plots and time series charts.

## Input Files

- **Stocks**: `/root/CSY/Stock_analysis/list/stocks_full_list.txt` — one ticker per line with exchange suffix (e.g., `AAPL.OQ`, `BABA.N`, `SIE.DE`)
- **ETFs**: `/root/CSY/Stock_analysis/list/etfs_full_list_v2.txt` — one ticker per line (e.g., `SPY.ETF`, `QQQ.OQ`)
- **Cryptos**: `/root/CSY/Stock_analysis/list/cryptos_full_list_v2.txt` — one symbol per line (e.g., `BTCUSD`, `ETHUSD`)

## Environment Setup

```bash
# Python >= 3.10 required
cd /root/CSY/project1
pip install -r requirements.txt

# Initialize Qlib (one-time)
python -c "import qlib; qlib.init()"
```

## Running the Pipeline

### Run all scripts sequentially

```bash
bash /root/CSY/project1/scripts/run_all.sh
```

### Run individual scripts

```bash
# 1. Classify stocks by market cap
python scripts/01_classify_universe.py

# 2. Download stock data (yfinance)
python scripts/02_download_stocks.py

# 3. Download ETF data (yfinance)
python scripts/03_download_etfs.py

# 4. Download crypto data (ccxt/Binance)
python scripts/04_download_crypto.py

# 5. Convert CSV to Qlib binary format
python scripts/05_to_qlib_format.py

# 6. Compute volatility features
python scripts/06_compute_features.py

# 7. Generate HTML reports
python scripts/07_generate_reports.py
```

## Output Structure

```
/root/CSY/project1/
├── config/
│   └── universe.yaml              # Stock classification (large/mid/small/unclassified)
├── data/
│   ├── raw/                       # Raw CSV data
│   │   ├── stocks/<TICKER>.csv
│   │   ├── etfs/<TICKER>.csv
│   │   └── crypto/<SYMBOL>.csv
│   ├── features/                  # Feature parquet files
│   │   └── <TICKER>_features.parquet
│   └── qlib_data/                 # Qlib binary format
│       ├── us_stock/
│       ├── us_etf/
│       └── crypto/
├── reports/
│   ├── large_cap.html             # Large cap cross-sectional average
│   ├── mid_cap.html               # Mid cap cross-sectional average
│   ├── small_cap.html             # Small cap cross-sectional average
│   ├── etfs/<TICKER>.html         # Per-ETF report
│   └── crypto/<SYMBOL>.html       # Per-crypto report
├── output/
│   ├── universe_classification.csv
│   ├── download_errors.log
│   └── TTP_Stock_ETF_Index.xlsx   # Index Excel with hyperlinks to reports
└── README.md
```

## Jupyter Notebook Examples

### Example 1: Load Qlib data and view a stock

```python
import qlib
from qlib.data import D

qlib.init(provider_uri="/root/CSY/project1/data/qlib_data/us_stock", region="us")

# Load daily OHLCV for AAPL.OQ
df = D.features(["AAPL.OQ"], fields=["$close", "$volume", "$open", "$high", "$low"],
                start_time="2020-01-01", end_time="2023-12-31", freq="day")
print(df.head())
```

### Example 2: Compute ATR factor using D.features

```python
import qlib
from qlib.data import D

qlib.init(provider_uri="/root/CSY/project1/data/qlib_data/us_stock", region="us")

# Compute ATR(14) using Qlib expression
fields = ["Ref($close, 1)"]
tr_expr = "Max($high - $low, Abs($high - Ref($close, 1)), Abs($low - Ref($close, 1)))"
atr14_expr = f"EMA({tr_expr}, 14)"

df = D.features(["AAPL.OQ"], fields=[atr14_expr],
                start_time="2020-01-01", end_time="2023-12-31", freq="day")
df.columns = ["ATR14"]
print(df.head(20))
```

### Example 3: Plot closing price and ATR

```python
import qlib
from qlib.data import D
import plotly.graph_objects as go

qlib.init(provider_uri="/root/CSY/project1/data/qlib_data/us_stock", region="us")

# Load data
df = D.features(["AAPL.OQ"], fields=["$close"],
                start_time="2020-01-01", end_time="2023-12-31", freq="day")

# Plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index.get_level_values(1), y=df["$close"].values,
                          mode="lines", name="Close"))
fig.update_layout(title="AAPL.OQ Daily Close", xaxis_title="Date", yaxis_title="Price")
fig.show()
```

## Error Handling

- Failed downloads are logged to `output/download_errors.log` with error reasons
- Network failures are retried 3 times before marking as failed
- Assets with insufficient data are skipped and recorded

## Rate Limiting

- yfinance: 0.5s sleep between batch calls
- ccxt: Built-in rate limiting for Binance API

## Timezone

- Stocks/ETFs: `America/New_York` → converted to `UTC` on export
- Cryptocurrencies: `UTC`

## Notes

- Crypto data may be less than 10 years for newer tokens; all available data from listing date is used
- Market cap classification follows international standards:
  - Large Cap: ≥ $10B
  - Mid Cap: $2B – $10B
  - Small Cap: $0.3B – $2B
  - Others: `unclassified`
