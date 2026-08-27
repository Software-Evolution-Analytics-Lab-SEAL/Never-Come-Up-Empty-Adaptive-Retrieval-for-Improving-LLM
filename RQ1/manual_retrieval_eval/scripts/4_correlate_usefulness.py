"""
Per-query Spearman correlation between Usefulness and final
LLM-as-Judge answer score, per (model, method) and pooled per model.

Outputs (under ../data/):
  spearman_usefulness_by_model_method.csv
  spearman_usefulness_pooled_by_model.csv
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
EVAL_DIR = os.path.join(ROOT, "data", "evaluation")

MODELS = ["llama-3.1-8b", "gpt-4.1", "qwen3-8b",
          "mistral-7b", "deepseek-r1-70b", "granite-3.1-8b"]
INTERNAL_PIPES = ["QB1", "QB2", "QB3", "QB4", "HB1", "HB2", "HYB"]
BASELINES = ["BM25", "RAGFUSION", "ADAPTIVERAG", "SELFRAG"]
METHODS = INTERNAL_PIPES + BASELINES
ADAPTIVE_THRS = [0.9, 0.8, 0.7, 0.6, 0.5]


def get_per_query_judge(model, method, sample_idxs):
    out = {}
    if method in BASELINES:
        path = os.path.join(EVAL_DIR, model, f"{method}.csv")
        if not os.path.exists(path): return out
        df = pd.read_csv(path)
        for i in sample_idxs:
            if i < len(df):
                s = df["judge_score"].iloc[i]
                if pd.notna(s) and s > 0:
                    out[i] = float(s)
        return out

    for t in ADAPTIVE_THRS:
        path = os.path.join(EVAL_DIR, model, f"{method}_{t}.csv")
        if not os.path.exists(path): continue
        df = pd.read_csv(path)
        for i in sample_idxs:
            if i in out: continue
            if i >= len(df): continue
            n = df["num_retrieved"].iloc[i] if "num_retrieved" in df.columns else 0
            s = df["judge_score"].iloc[i]
            if pd.notna(n) and n >= 1 and pd.notna(s) and s > 0:
                out[i] = float(s)
    return out


def main():
    pq = pd.read_csv(os.path.join(DATA, "per_query_usefulness.csv"))
    sample = pd.read_csv(os.path.join(DATA, "sample_queries.csv"))
    sample_idxs = sample["test_idx"].astype(int).tolist()

    rows = []
    for model in MODELS:
        for method in METHODS:
            scores = get_per_query_judge(model, method, sample_idxs)
            sub = pq[pq["method"] == method].copy()
            sub["judge"] = sub["test_idx"].map(scores)
            sub = sub.dropna(subset=["judge", "usefulness"])
            if len(sub) < 10:
                continue
            rho, pv = spearmanr(sub["usefulness"], sub["judge"])
            rows.append({"model": model, "method": method,
                         "rho": float(rho) if pd.notna(rho) else np.nan,
                         "p": float(pv) if pd.notna(pv) else np.nan,
                         "n": int(len(sub))})
    granular = pd.DataFrame(rows)
    granular.to_csv(os.path.join(DATA, "spearman_usefulness_by_model_method.csv"), index=False)

    print("=== Spearman ρ (Usefulness ↔ judge_score) per (model, method) ===")
    pivot = granular.pivot(index="model", columns="method", values="rho")
    print(pivot.round(3).to_string())

    print("\n=== Mean / median ρ per model (across 11 methods) ===")
    summary_m = granular.groupby("model").agg(
        mean_rho=("rho", "mean"),
        median_rho=("rho", "median"),
        n_pos=("rho", lambda s: int((s > 0).sum())),
        n_sig=("p", lambda s: int((s < 0.05).sum())),
        n_cells=("rho", "count"),
    )
    print(summary_m.round(3).to_string())

    print("\n=== Mean ρ per method (across 6 models) ===")
    summary_x = granular.groupby("method").agg(
        mean_rho=("rho", "mean"),
        median_rho=("rho", "median"),
        n_pos=("rho", lambda s: int((s > 0).sum())),
        n_sig=("p", lambda s: int((s < 0.05).sum())),
    ).sort_values("mean_rho", ascending=False)
    print(summary_x.round(3).to_string())

    # Pooled per model
    pooled_rows = []
    for model in MODELS:
        triples = []
        for method in METHODS:
            scores = get_per_query_judge(model, method, sample_idxs)
            sub = pq[pq["method"] == method]
            for _, r in sub.iterrows():
                ti = int(r["test_idx"])
                if ti in scores and not pd.isna(r["usefulness"]):
                    triples.append((r["usefulness"], scores[ti]))
        if not triples:
            continue
        df = pd.DataFrame(triples, columns=["usefulness", "judge"])
        rho, pv = spearmanr(df["usefulness"], df["judge"])
        pooled_rows.append({"model": model, "rho": float(rho), "p": float(pv), "n": len(df)})
    pooled = pd.DataFrame(pooled_rows)
    pooled.to_csv(os.path.join(DATA, "spearman_usefulness_pooled_by_model.csv"), index=False)

    print("\n=== Pooled Spearman ρ per model (Usefulness vs judge_score, all 11 methods stacked) ===")
    print(f"{'model':<18}  {'rho':>6}  {'p':>10}  {'n':>5}")
    for _, r in pooled.iterrows():
        print(f"{r.model:<18}  {r.rho:>+6.3f}  {r.p:>10.2e}  {int(r.n):>5}")


if __name__ == "__main__":
    main()
