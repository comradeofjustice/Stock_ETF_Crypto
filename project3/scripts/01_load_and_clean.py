"""
01_load_and_clean.py
加载全部标的 close 价格，计算日 log return，对齐到工作日索引，保存处理结果。
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path(__file__).parent.parent / "data" / "raw"
OUT  = Path(__file__).parent.parent / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)


def _ticker_label(path: Path, asset_class: str) -> str:
    """从文件名提取简洁 ticker，去掉交易所后缀。"""
    name = path.stem  # e.g. AAPL.OQ -> AAPL.OQ
    # crypto 直接去掉 USD 结尾
    if asset_class == "crypto":
        return name.replace("USD", "")
    # stocks/etfs: 保留第一段 (AAPL.OQ -> AAPL)，但保留 .ETF 类的第一段
    parts = name.split(".")
    return parts[0]


def load_category(folder: Path, asset_class: str) -> dict[str, pd.Series]:
    """读取一个文件夹下的所有 CSV，返回 {label: close_series}。"""
    series = {}
    for csv in sorted(folder.glob("*.csv")):
        label = _ticker_label(csv, asset_class)
        # 对重名 ticker（同公司多交易所）加后缀区分
        if label in series:
            label = csv.stem  # 退回完整文件名
        try:
            df = pd.read_csv(csv, parse_dates=["Date"], index_col="Date")
            if "close" not in df.columns:
                continue
            s = df["close"].dropna()
            s = s[s > 0]
            if len(s) < 60:
                continue
            series[label] = s
        except Exception as e:
            print(f"  skip {csv.name}: {e}")
    return series


def main():
    print("=== 01 Load & Clean ===")

    all_series = {}

    # 定义加载顺序，crypto 放最前便于后续分组
    categories = [
        ("crypto", RAW / "crypto"),
        ("etf",    RAW / "etfs"),
        ("stock",  RAW / "stocks"),
    ]

    labels_meta = {}  # label -> asset_class

    for asset_class, folder in categories:
        print(f"  loading {asset_class} from {folder.name} …")
        raw = load_category(folder, asset_class)
        for lbl, s in raw.items():
            all_series[lbl] = s
            labels_meta[lbl] = asset_class
        print(f"    → {len(raw)} assets loaded")

    # 合并到宽表（外连接），对齐价格序列
    price_df = pd.DataFrame(all_series)

    # 只保留工作日（周一~周五），统一时间轴
    price_df = price_df[price_df.index.dayofweek < 5]
    price_df.sort_index(inplace=True)

    # 计算日 log return
    ret_df = np.log(price_df / price_df.shift(1))

    # 过滤掉数据量太少的列（要求至少 200 个有效收益率）
    valid_cols = ret_df.count()[ret_df.count() >= 200].index
    ret_df = ret_df[valid_cols]
    price_df = price_df[valid_cols]

    print(f"\n  有效标的数: {len(valid_cols)}")
    print(f"  日期范围: {ret_df.index[0].date()} ~ {ret_df.index[-1].date()}")
    print(f"  总行数: {len(ret_df)}")

    # 保存
    price_df.to_csv(OUT / "prices_aligned.csv")
    ret_df.to_csv(OUT / "returns_log.csv")

    # 保存 meta 信息
    meta = pd.DataFrame([
        {"ticker": lbl, "asset_class": labels_meta.get(lbl, "unknown")}
        for lbl in valid_cols
    ])
    meta.to_csv(OUT / "asset_meta.csv", index=False)

    print(f"\n  已保存到 {OUT}")
    print("  - prices_aligned.csv")
    print("  - returns_log.csv")
    print("  - asset_meta.csv")
    print("=== 01 Done ===")


if __name__ == "__main__":
    main()
