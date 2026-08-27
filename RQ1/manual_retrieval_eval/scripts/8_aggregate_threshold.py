"""
Aggregate per-(method, threshold) Usefulness from
usefulness_threshold_decisions.csv.

Outputs (under ../data/):
  per_query_usefulness_threshold.csv  (test_idx, method, threshold, n_units, n_useful, usefulness)
  threshold_sweep_summary.csv         (method, threshold, mean_usefulness,
                                       ci_lo, ci_hi, avg_units_per_query, n_queries)
"""
import os
import numpy as np
import pandas as pd

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


def main():
    p = pd.read_csv(os.path.join(DATA, "usefulness_threshold_decisions.csv"))
    p["yes"] = to_bool(p["useful"])

    pq = p.groupby(["test_idx", "method", "threshold"]).agg(
        n_units=("yes", "size"),
        n_useful=("yes", "sum"),
    ).reset_index()
    pq["usefulness"] = pq["n_useful"] / pq["n_units"]
    pq.to_csv(os.path.join(DATA, "per_query_usefulness_threshold.csv"), index=False)

    rows = []
    for (m, t), sub in pq.groupby(["method", "threshold"]):
        vals = sub["usefulness"].to_numpy()
        lo, hi = bootstrap_ci(vals, B=B, seed=SEED)
        rows.append({
            "method": m, "threshold": t,
            "mean_usefulness": float(np.mean(vals)),
            "ci_lo": lo, "ci_hi": hi,
            "avg_units_per_query": float(sub["n_units"].mean()),
            "n_queries": int(len(sub)),
        })
    summary = pd.DataFrame(rows).sort_values(["method", "threshold"])
    summary.to_csv(os.path.join(DATA, "threshold_sweep_summary.csv"), index=False)

    print("=== Per-(method, threshold) Usefulness ===")
    print(f"{'method':<5}{'thr':>6}  {'Usefulness (95% CI)':<22}  {'avg #units/q':>12}  {'n_q':>4}")
    for _, r in summary.iterrows():
        print(f"{r.method:<5}{r.threshold:>6.1f}  "
              f"{r.mean_usefulness:.3f} [{r.ci_lo:.3f},{r.ci_hi:.3f}]   "
              f"{r.avg_units_per_query:>10.2f}  {r.n_queries:>4}")

    # Pretty pivot table
    print("\n=== Pivot: Usefulness mean (rows: thr, cols: method) ===")
    pivot_u = summary.pivot(index="threshold", columns="method", values="mean_usefulness")
    print(pivot_u.round(3).to_string())

    print("\n=== Pivot: avg units/query (rows: thr, cols: method) ===")
    pivot_n = summary.pivot(index="threshold", columns="method", values="avg_units_per_query")
    print(pivot_n.round(2).to_string())

    print("\n=== Pivot: coverage n_queries (rows: thr, cols: method) ===")
    pivot_q = summary.pivot(index="threshold", columns="method", values="n_queries")
    print(pivot_q.to_string())


if __name__ == "__main__":
    main()
