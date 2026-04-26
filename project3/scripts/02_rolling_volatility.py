"""
02_rolling_volatility.py
基于日 log return 计算滚动年化波动率（21d 月度 / 63d 季度），保存宽表。
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path(__file__).parent.parent / "data" / "processed"
SQRT252 = np.sqrt(252)


def rolling_vol(ret_df: pd.DataFrame, window: int) -> pd.DataFrame:
    """计算滚动年化波动率，min_periods = window // 2。"""
    return ret_df.rolling(window=window, min_periods=window // 2).std() * SQRT252


def main():
    print("=== 02 Rolling Volatility ===")

    ret_df = pd.read_csv(DATA / "returns_log.csv", index_col=0, parse_dates=True)
    print(f"  加载收益率矩阵: {ret_df.shape}")

    vol_21  = rolling_vol(ret_df, 21)
    vol_63  = rolling_vol(ret_df, 63)

    vol_21.to_csv(DATA / "vol_21d.csv")
    vol_63.to_csv(DATA / "vol_63d.csv")

    # 同时保存全期平均波动率（用于静态分析）
    mean_vol = pd.DataFrame({
        "mean_vol_21d": vol_21.mean(),
        "mean_vol_63d": vol_63.mean(),
    })
    mean_vol.to_csv(DATA / "mean_vol.csv")

    print(f"  vol_21d shape: {vol_21.shape}")
    print(f"  vol_63d shape: {vol_63.shape}")
    print(f"\n  已保存到 {DATA}")
    print("  - vol_21d.csv")
    print("  - vol_63d.csv")
    print("  - mean_vol.csv")
    print("=== 02 Done ===")


if __name__ == "__main__":
    main()
