"""
04_correlation_analysis.py
Pearson + Spearman 相关系数矩阵及 p 值检验（基于 21d 和 63d 滚动波动率时间序列）。
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

DATA = Path(__file__).parent.parent / "data" / "processed"


def ordered_cols(df: pd.DataFrame, meta: pd.DataFrame) -> list[str]:
    order = {"crypto": 0, "etf": 1, "stock": 2}
    meta_map = meta.set_index("ticker")["asset_class"].to_dict()
    cols = [c for c in df.columns if c in meta_map]
    return sorted(cols, key=lambda c: (order.get(meta_map.get(c, "stock"), 2), c))


def pearson_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """全量 Pearson 相关系数矩阵 + p 值矩阵（pairwise complete obs）。"""
    cols = df.columns.tolist()
    n = len(cols)
    r_mat = np.ones((n, n))
    p_mat = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            x = df[cols[i]]
            y = df[cols[j]]
            mask = x.notna() & y.notna()
            if mask.sum() < 30:
                r, p = np.nan, np.nan
            else:
                r, p = stats.pearsonr(x[mask], y[mask])
            r_mat[i, j] = r_mat[j, i] = r
            p_mat[i, j] = p_mat[j, i] = p

    r_df = pd.DataFrame(r_mat, index=cols, columns=cols)
    p_df = pd.DataFrame(p_mat, index=cols, columns=cols)
    return r_df, p_df


def spearman_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman 秩相关矩阵（对非正态分布更鲁棒）。"""
    return df.corr(method="spearman")


def main():
    print("=== 04 Correlation Analysis ===")

    meta   = pd.read_csv(DATA / "asset_meta.csv")
    vol_21 = pd.read_csv(DATA / "vol_21d.csv", index_col=0, parse_dates=True)
    vol_63 = pd.read_csv(DATA / "vol_63d.csv", index_col=0, parse_dates=True)

    cols = ordered_cols(vol_21, meta)
    v21  = vol_21[cols]
    v63  = vol_63[cols]

    print("  计算 21d Pearson 相关矩阵 …")
    r21_pearson, p21_pearson = pearson_matrix(v21)

    print("  计算 63d Pearson 相关矩阵 …")
    r63_pearson, p63_pearson = pearson_matrix(v63)

    print("  计算 21d Spearman 相关矩阵 …")
    r21_spearman = spearman_matrix(v21)

    print("  计算 63d Spearman 相关矩阵 …")
    r63_spearman = spearman_matrix(v63)

    r21_pearson.to_csv(DATA / "corr_pearson_21d.csv")
    p21_pearson.to_csv(DATA / "pval_pearson_21d.csv")
    r63_pearson.to_csv(DATA / "corr_pearson_63d.csv")
    p63_pearson.to_csv(DATA / "pval_pearson_63d.csv")
    r21_spearman.to_csv(DATA / "corr_spearman_21d.csv")
    r63_spearman.to_csv(DATA / "corr_spearman_63d.csv")

    # 打印 BTC/ETH/SOL 与各类资产的相关系数摘要
    meta_map  = meta.set_index("ticker")["asset_class"].to_dict()
    cryptos   = [c for c in ["BTC", "ETH", "SOL"] if c in r21_pearson.columns]

    for crypto in cryptos:
        row = r21_pearson[crypto].drop(cryptos).dropna()
        sig = p21_pearson[crypto].drop(cryptos)
        row_sig = row[sig < 0.05]  # 只看显著相关
        top5    = row_sig.abs().nlargest(5).index
        print(f"\n  {crypto} ← Top-5 显著相关 (21d Pearson, p<0.05):")
        for t in top5:
            print(f"    {t:<12} [{meta_map.get(t,'?'):<6}]  r={row[t]:+.3f}  p={sig[t]:.2e}")

    print(f"\n  已保存到 {DATA}")
    print("  - corr_pearson_21d/63d.csv  pval_pearson_21d/63d.csv")
    print("  - corr_spearman_21d/63d.csv")
    print("=== 04 Done ===")


if __name__ == "__main__":
    main()
