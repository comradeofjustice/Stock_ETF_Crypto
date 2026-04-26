"""
03_covariance_analysis.py
计算波动率时间序列的协方差矩阵（21d 和 63d），保存为 CSV。
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path(__file__).parent.parent / "data" / "processed"


def ordered_cols(df: pd.DataFrame, meta: pd.DataFrame) -> list[str]:
    """按 crypto -> etf -> stock 排列列名。"""
    order = {"crypto": 0, "etf": 1, "stock": 2}
    meta_map = meta.set_index("ticker")["asset_class"].to_dict()
    cols = [c for c in df.columns if c in meta_map]
    return sorted(cols, key=lambda c: (order.get(meta_map.get(c, "stock"), 2), c))


def compute_cov(vol_df: pd.DataFrame) -> pd.DataFrame:
    """pairwise 协方差（忽略 NaN 对）。"""
    return vol_df.cov()


def main():
    print("=== 03 Covariance Analysis ===")

    meta   = pd.read_csv(DATA / "asset_meta.csv")
    vol_21 = pd.read_csv(DATA / "vol_21d.csv", index_col=0, parse_dates=True)
    vol_63 = pd.read_csv(DATA / "vol_63d.csv", index_col=0, parse_dates=True)

    cols = ordered_cols(vol_21, meta)

    cov_21 = compute_cov(vol_21[cols])
    cov_63 = compute_cov(vol_63[cols])

    cov_21.to_csv(DATA / "cov_matrix_21d.csv")
    cov_63.to_csv(DATA / "cov_matrix_63d.csv")

    # 统计摘要：与 BTC/ETH/SOL 的协方差排名
    cryptos = [c for c in ["BTC", "ETH", "SOL"] if c in cov_21.columns]
    for crypto in cryptos:
        top = cov_21[crypto].drop(cryptos).nlargest(10)
        print(f"\n  Top-10 covariance with {crypto} (21d vol):")
        for ticker, val in top.items():
            cls = meta.set_index("ticker")["asset_class"].get(ticker, "?")
            print(f"    {ticker:<12} [{cls:<6}]  {val:.6f}")

    print(f"\n  已保存到 {DATA}")
    print("  - cov_matrix_21d.csv")
    print("  - cov_matrix_63d.csv")
    print("=== 03 Done ===")


if __name__ == "__main__":
    main()
