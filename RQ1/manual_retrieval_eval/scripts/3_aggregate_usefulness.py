"""
Aggregate per-query and per-method Usefulness from usefulness_decisions.csv.

Outputs (under ../data/):
  per_query_usefulness.csv  (test_idx, method, n_units, n_useful, usefulness)
  usefulness_summary.csv    (method, mean_usefulness, ci_lo, ci_hi,
                             avg_units_per_query, n_queries)
  usefulness_pairwise.csv   (method_a, method_b, n_paired, w, p,
                             cliff_d, es)
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
SEED = 42
B = 1000


def to_bool(s):
    return s.astype(str).str.lower().eq("true")


def bootstrap_ci(arr, B=1000, seed=42):
    rng = np.random.default_rng(seed)
    a = np.asarray(arr, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return (np.nan, np.nan)
    idx = rng.integers(0, len(a), size=(B, len(a)))
    means = a[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cliff_d(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return (gt - lt) / (len(a) * len(b))


def es_label(d):
    ad = abs(d)
    if np.isnan(d): return "-"
    if ad < 0.147: return "negligible"
    if ad < 0.33:  return "small"
    if ad < 0.474: return "medium"
    return "large"


def main():
    p = pd.read_csv(os.path.join(DATA, "usefulness_decisions.csv"))
    p["yes"] = to_bool(p["useful"])

    # Per-query usefulness
    pq = p.groupby(["test_idx", "method"]).agg(
        n_units=("yes", "size"),
        n_useful=("yes", "sum"),
    ).reset_index()
    pq["usefulness"] = pq["n_useful"] / pq["n_units"]
    pq.to_csv(os.path.join(DATA, "per_query_usefulness.csv"), index=False)

    # Per-method summary
    rows = []
    for m, sub in pq.groupby("method"):
        vals = sub["usefulness"].to_numpy()
        lo, hi = bootstrap_ci(vals, B=B, seed=SEED)
        rows.append({
            "method": m,
            "mean_usefulness": float(np.mean(vals)),
            "ci_lo": lo, "ci_hi": hi,
            "avg_units_per_query": float(sub["n_units"].mean()),
            "n_queries": int(len(sub)),
        })
    summary = pd.DataFrame(rows).sort_values("mean_usefulness", ascending=False)
    summary.to_csv(os.path.join(DATA, "usefulness_summary.csv"), index=False)

    print("=== Per-method Usefulness (mean [95% CI]) ===")
    print(f"{'method':<12}  {'Usefulness (95% CI)':<22}  {'avg #units/q':>12}  {'n_q':>4}")
    for _, r in summary.iterrows():
        print(f"{r.method:<12}  {r.mean_usefulness:.3f} [{r.ci_lo:.3f},{r.ci_hi:.3f}]   {r.avg_units_per_query:>10.2f}  {r.n_queries:>4}")

    # Pairwise Wilcoxon on per-query Usefulness (paired)
    f1 = pq.pivot(index="test_idx", columns="method", values="usefulness")
    methods = sorted(pq["method"].unique())
    rows = []
    for i, ma in enumerate(methods):
        for mb in methods[i+1:]:
            pair = f1[[ma, mb]].dropna()
            if len(pair) < 5:
                rows.append({"method_a": ma, "method_b": mb, "n": len(pair),
                             "delta": np.nan, "w": np.nan, "p": np.nan,
                             "cliff_d": np.nan, "es": "-"})
                continue
            a, b = pair[ma].to_numpy(), pair[mb].to_numpy()
            try:
                w, pv = wilcoxon(a, b, zero_method="wilcox")
            except ValueError:
                w, pv = np.nan, np.nan
            d = cliff_d(a, b)
            rows.append({"method_a": ma, "method_b": mb, "n": int(len(pair)),
                         "delta": float(np.mean(a - b)),
                         "w": float(w) if not np.isnan(w) else np.nan,
                         "p": float(pv) if not np.isnan(pv) else np.nan,
                         "cliff_d": float(d), "es": es_label(d)})
    pd.DataFrame(rows).to_csv(os.path.join(DATA, "usefulness_pairwise.csv"), index=False)

    print("\n=== HB1 vs other methods on Usefulness (paired Wilcoxon) ===")
    print(f"{'opponent':<12}  {'n':>3}  {'HB1 mean':>9}  {'opp mean':>9}  {'Delta':>7}  {'p':>10}  {'cliff_d':>8}  ES")
    for m in methods:
        if m == "HB1":
            continue
        pair = f1[["HB1", m]].dropna()
        if len(pair) < 5:
            print(f"{m:<12}  insufficient")
            continue
        a, b = pair["HB1"].to_numpy(), pair[m].to_numpy()
        try:
            w, pv = wilcoxon(a, b, zero_method="wilcox")
        except ValueError:
            w, pv = np.nan, np.nan
        d = cliff_d(a, b)
        print(f"{m:<12}  {len(pair):>3}  {a.mean():>9.3f}  {b.mean():>9.3f}  {a.mean()-b.mean():>+7.3f}  {pv:>10.2e}  {d:>+8.3f}  {es_label(d)}")

    print(f"\nSaved per_query_usefulness.csv, usefulness_summary.csv, usefulness_pairwise.csv")


if __name__ == "__main__":
    main()
