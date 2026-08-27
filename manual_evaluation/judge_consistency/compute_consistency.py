"""
Compute agreement between deepseek-coder-v2:16b judge scores and:
  - Selected_Score (previous LLM judge, GPT-4o)
  - p1 (annotator 1, binary -1/1)
  - p2 (annotator 2, binary -1/1)

Outputs:
  - Score-level consistency (DeepSeek vs Selected_Score 1-10):
      Spearman, Pearson, Cohen's kappa (multi-class binning)
  - Binary-mapped consistency (map 2-5 -> -1, 6-9 -> 1) against
      Selected_Score-binary, p1, p2: Cohen's kappa, raw agreement
  - Fleiss' kappa across {DeepSeek-bin, Selected-bin, p1, p2}
"""
import os
import glob
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score

HERE = os.path.dirname(os.path.abspath(__file__))
SHARDS = sorted(glob.glob(os.path.join(HERE, "deepseek_coder_v2_16b_scores_shard*.csv")))


def to_binary_1_1(score):
    """Map 1-10 score -> -1 (score 2-5) / 1 (score 6-9). Score 1 or 10 at edges."""
    if score <= 5:
        return -1
    if score >= 6:
        return 1
    return 0


def label_kappa(k):
    if np.isnan(k): return "n/a"
    if k < 0: return "worse-than-chance"
    if k < 0.20: return "poor"
    if k < 0.40: return "fair"
    if k < 0.60: return "moderate"
    if k < 0.80: return "substantial"
    return "almost-perfect"


def fleiss_kappa(ratings_matrix):
    """Fleiss' Kappa. ratings_matrix shape (n_items, n_categories)."""
    n, k = ratings_matrix.shape
    n_raters = ratings_matrix[0].sum()
    if n_raters < 2: return float('nan')
    p_j = ratings_matrix.sum(axis=0) / (n * n_raters)
    P_e = (p_j ** 2).sum()
    P_i = ((ratings_matrix ** 2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = P_i.mean()
    return (P_bar - P_e) / (1 - P_e) if P_e < 1 else 0.0


def main():
    # Combine shards
    dfs = [pd.read_csv(s) for s in SHARDS]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} rows from {len(SHARDS)} shards")
    # Filter valid rows
    df = df[df["deepseek_score"] > 0].reset_index(drop=True)
    print(f"  valid (deepseek > 0): {len(df)}")

    # ====== Score-level agreement ======
    ds = df["deepseek_score"].astype(int).values
    gs = df["Selected_Score"].astype(int).values
    spearman_r, spearman_p = stats.spearmanr(ds, gs)
    pearson_r, pearson_p = stats.pearsonr(ds, gs)
    mad = float(np.mean(np.abs(ds - gs)))
    bins = [0, 4, 6, 8, 11]  # same 4-bucket binning as last round
    ds_bin = pd.cut(ds, bins=bins, labels=False).astype(int)
    gs_bin = pd.cut(gs, bins=bins, labels=False).astype(int)
    kappa_mc = cohen_kappa_score(ds_bin, gs_bin)

    print("\n=== Score-level (1-10) agreement: deepseek-coder-v2:16b vs Selected_Score ===")
    print(f"  Spearman rho    = {spearman_r:.4f} (p={spearman_p:.2e})")
    print(f"  Pearson  r      = {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"  Mean |Δ|        = {mad:.2f}")
    print(f"  Cohen's κ (multi-class, 4 buckets) = {kappa_mc:.4f}  ({label_kappa(kappa_mc)})")
    print(f"  DeepSeek mean   = {ds.mean():.2f}   Selected mean = {gs.mean():.2f}")

    # ====== Binary-mapped agreement ======
    df_b = df.copy()
    df_b["ds_bin"] = df_b["deepseek_score"].map(to_binary_1_1)
    df_b["sel_bin"] = df_b["Selected_Score"].map(to_binary_1_1)
    df_b["p1_bin"] = df_b["p1"].astype(int)   # already -1/1
    df_b["p2_bin"] = df_b["p2"].astype(int)

    print("\n=== Binary-mapped agreement (score 2-5 -> -1, score 6-9 -> 1) ===")
    print(f"{'pair':<30} {'n':>4} {'raw_agree':>10} {'kappa':>8}  {'label':<15}")
    for col_a, col_b, name in [
        ("ds_bin", "sel_bin", "DeepSeek vs Selected_Score"),
        ("ds_bin", "p1_bin",  "DeepSeek vs annotator p1"),
        ("ds_bin", "p2_bin",  "DeepSeek vs annotator p2"),
        ("sel_bin", "p1_bin", "Selected vs annotator p1"),
        ("sel_bin", "p2_bin", "Selected vs annotator p2"),
        ("p1_bin", "p2_bin",  "p1 vs p2 (annotator baseline)"),
    ]:
        a = df_b[col_a].values
        b = df_b[col_b].values
        agree = (a == b).mean() * 100
        kappa = cohen_kappa_score(a, b)
        print(f"  {name:<28} {len(a):>4d}  {agree:>8.2f}%  {kappa:>+.4f}  {label_kappa(kappa):<15}")

    # Fleiss' kappa across all 4 binary raters
    # Convert -1/1 to categories 0/1 for Fleiss
    ratings = np.zeros((len(df_b), 2), dtype=int)
    for col in ["ds_bin", "sel_bin", "p1_bin", "p2_bin"]:
        vals = df_b[col].values
        for i, v in enumerate(vals):
            ratings[i, 0 if v < 0 else 1] += 1
    fk = fleiss_kappa(ratings)
    print(f"\nFleiss' kappa across 4 raters (DeepSeek, Selected, p1, p2): {fk:.4f}  ({label_kappa(fk)})")

    # High-rate (% marked 1 / acceptable) per rater
    print("\nPer-rater 'acceptable' (=1) rate:")
    for col, name in [("ds_bin","DeepSeek"), ("sel_bin","Selected"), ("p1_bin","p1"), ("p2_bin","p2")]:
        rate = (df_b[col] == 1).mean() * 100
        print(f"  {name:<10}  {rate:>5.1f}%")

    # Save merged results
    df_b.to_csv(os.path.join(HERE, "deepseek_coder_consistency_results.csv"), index=False)
    print(f"\nSaved merged results to deepseek_coder_consistency_results.csv")


if __name__ == "__main__":
    main()
