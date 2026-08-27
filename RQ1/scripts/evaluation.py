"""
RQ1 evaluation — per-method significance testing and the Table II builder.

Usage:
    python evaluation.py significance      # HB1 vs each method: Wilcoxon + Cliff's δ
    python evaluation.py table-ii          # Full Table II with adaptive/simple/weighted rows
"""
import argparse
import bisect
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


# ---------------------------------------------------------------------------
# Significance testing: HB1 vs each method (pooled across 6 generators)
# ---------------------------------------------------------------------------

_GENS = ["llama-3.1-8b", "mistral-7b", "qwen3-8b", "granite-3.1-8b", "deepseek-r1-70b", "gpt-4.1"]
_INTERNAL = ["HB1", "HYB", "HB2", "QB1", "QB2", "QB3", "QB4"]
_BASELINES = ["RAGFUSION", "SELFRAG", "ADAPTIVERAG", "BM25"]


def _eval_dir():
    return os.path.join(config.DATA_DIR, "evaluation")


def _load_adaptive(model, pipe):
    """Adaptive score per query: earliest (highest) threshold that retrieved."""
    chosen = {}
    for thr in config.THRESHOLDS[::-1]:                 # 0.9 → 0.1
        f = os.path.join(_eval_dir(), model, f"{pipe}_{thr}.csv")
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f)
        for _, r in df.iterrows():
            pid = r["post_idx"]
            if pid in chosen:
                continue
            if pd.notna(r.get("num_retrieved")) and r["num_retrieved"] >= 1 and pd.notna(r.get("judge_score")):
                chosen[pid] = float(r["judge_score"])
    return chosen


def _load_baseline(model, pipe):
    f = os.path.join(_eval_dir(), model, f"{pipe}.csv")
    if not os.path.exists(f):
        return {}
    df = pd.read_csv(f)
    return {r["post_idx"]: float(r["judge_score"])
            for _, r in df.iterrows() if pd.notna(r.get("judge_score"))}


def _cliffs(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    bs = sorted(b)
    gt = sum(bisect.bisect_left(bs, x) for x in a)
    lt = sum(len(bs) - bisect.bisect_right(bs, x) for x in a)
    return (gt - lt) / (len(a) * len(b))


def cmd_significance():
    def load(model, pipe):
        return _load_adaptive(model, pipe) if pipe in _INTERNAL else _load_baseline(model, pipe)

    methods = _INTERNAL[1:] + _BASELINES               # everything except HB1

    print("=" * 70)
    print("POOLED HB1 vs each method (paired per-query across 6 generators)")
    print(f"{'method':12}{'HB1':>6}{'meth':>6}{'Δ':>7}{'Wilcoxon p':>12}{'Cliff δ':>9}")
    for m in methods:
        H, M = [], []
        for g in _GENS:
            h, o = load(g, "HB1"), load(g, m)
            for pid in set(h) & set(o):
                H.append(h[pid]); M.append(o[pid])
        H, M = np.array(H), np.array(M)
        try:
            p = wilcoxon(H, M, zero_method="wilcox").pvalue
        except Exception:
            p = float("nan")
        print(f"{m:12}{H.mean():6.2f}{M.mean():6.2f}{(H - M).mean():+7.3f}{p:12.1e}{_cliffs(H, M):+9.3f}")

    print("=" * 70)
    print("RAG-Fusion: per-generator HB1 vs RAGFUSION")
    print(f"{'generator':16}{'HB1':>6}{'RAGF':>6}{'Δ':>7}{'Wilcoxon p':>12}{'Cliff δ':>9}")
    for g in _GENS:
        h, o = load(g, "HB1"), load(g, "RAGFUSION")
        common = sorted(set(h) & set(o))
        H = np.array([h[p] for p in common])
        M = np.array([o[p] for p in common])
        try:
            p = wilcoxon(H, M, zero_method="wilcox").pvalue
        except Exception:
            p = float("nan")
        print(f"{g:16}{H.mean():6.2f}{M.mean():6.2f}{(H - M).mean():+7.3f}{p:12.2e}{_cliffs(H, M):+9.3f}")


# ---------------------------------------------------------------------------
# Table II — per-model per-threshold means + Avg(simple/weighted/adaptive)
# ---------------------------------------------------------------------------

_TBL_MODELS = ["llama-3.1-8b", "gpt-4.1", "qwen3-8b", "mistral-7b", "deepseek-r1-70b", "granite-3.1-8b"]
_TBL_DISPLAY = {"granite-3.1-8b": "granite-3.1-8b"}
_TBL_PIPES = ["QB1", "QB2", "QB3", "QB4", "HB1", "HB2", "HYB"]
_TBL_THRS = config.THRESHOLDS
_TBL_ADAPTIVE_THRS = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
_TBL_BASELINES = ["ZEROSHOT", "BM25", "RAGFUSION", "ADAPTIVERAG", "SELFRAG"]
_N = 666


def _zs_dir():
    return os.path.join(config.DATA_DIR, "evaluation", "zeroshot")


def _tbl_load_arr(path):
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path)
    s = pd.to_numeric(df["judge_score"], errors="coerce").to_numpy(dtype=float)
    n = pd.to_numeric(df.get("num_retrieved", pd.Series([0] * len(df))),
                      errors="coerce").to_numpy(dtype=float)
    s = np.where(s > 0, s, np.nan)
    if len(s) != _N:
        s2, n2 = np.full(_N, np.nan), np.zeros(_N)
        L = min(len(s), _N)
        s2[:L], n2[:L] = s[:L], n[:L]
        s, n = s2, n2
    return s, n


def _tbl_load_baseline(model, name):
    path = (os.path.join(_zs_dir(), f"{model}.csv") if name == "ZEROSHOT"
            else os.path.join(_eval_dir(), model, f"{name}.csv"))
    s, _ = _tbl_load_arr(path)
    return s


def _tbl_adaptive(model, pipeline, zs):
    out = np.full(_N, np.nan)
    for t in _TBL_ADAPTIVE_THRS:
        s, n = _tbl_load_arr(os.path.join(_eval_dir(), model, f"{pipeline}_{t}.csv"))
        if s is None:
            continue
        elig = np.isnan(out) & (n >= 1)
        out[elig] = s[elig]
    fb = np.isnan(out)
    out[fb] = zs[fb]
    return out


def _tbl_cliff_d(a, b):
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return (gt - lt) / (len(a) * len(b))


def _tbl_es_label(d):
    if np.isnan(d): return "-"
    ad = abs(d)
    if ad < 0.147: return "negligible"
    if ad < 0.33:  return "small"
    if ad < 0.474: return "medium"
    return "large"


def cmd_table_ii():
    for M in _TBL_MODELS:
        disp = _TBL_DISPLAY.get(M, M)
        print(f"\n{'#' * 100}\n#  {disp}\n{'#' * 100}")

        cell, n_retr, sum_score = {}, {}, {}
        for p in _TBL_PIPES:
            for t in _TBL_THRS:
                s, n = _tbl_load_arr(os.path.join(_eval_dir(), M, f"{p}_{t}.csv"))
                if s is None:
                    cell[(p, t)] = np.nan; n_retr[(p, t)] = 0; sum_score[(p, t)] = 0
                    continue
                mask = (n >= 1) & (~np.isnan(s))
                cell[(p, t)] = float(s[mask].mean()) if mask.any() else np.nan
                n_retr[(p, t)] = int(mask.sum())
                sum_score[(p, t)] = float(s[mask].sum())

        base = {b: _tbl_load_baseline(M, b) for b in _TBL_BASELINES}
        base_mean_full = {b: float(np.nanmean(arr)) if arr is not None else np.nan
                          for b, arr in base.items()}

        hb1_retrieve_mask = {}
        for t in _TBL_THRS:
            s, n = _tbl_load_arr(os.path.join(_eval_dir(), M, f"HB1_{t}.csv"))
            hb1_retrieve_mask[t] = (n >= 1) if n is not None else np.zeros(_N, dtype=bool)

        cols = _TBL_PIPES + ["RowMean"] + _TBL_BASELINES
        print("\n  Per-row baseline columns: restricted to queries that HB1 retrieved at that threshold.")
        print(f"\n{'thr':<5}" + "".join([f"{c:>12s}" for c in cols]) + f"{'#HB1q':>8}")
        for t in _TBL_THRS:
            cells, vals = [], []
            for p in _TBL_PIPES:
                v = cell[(p, t)]
                cells.append(f"{'--':>12}" if pd.isna(v) else f"{v:>12.2f}")
                if not pd.isna(v): vals.append(v)
            row_mean = np.mean(vals) if vals else np.nan
            cells.append(f"{'--':>12}" if pd.isna(row_mean) else f"{row_mean:>12.2f}")
            mask = hb1_retrieve_mask[t]
            n_q = int(mask.sum())
            for b in _TBL_BASELINES:
                arr = base[b]
                if arr is None or n_q == 0:
                    cells.append(f"{'--':>12}")
                else:
                    m_b = np.nanmean(arr[mask])
                    cells.append(f"{m_b:>12.2f}" if not pd.isna(m_b) else f"{'--':>12}")
            print(f"{t:<5.1f}" + "".join(cells) + f"{n_q:>8d}")

        simple, weighted = {}, {}
        for p in _TBL_PIPES:
            ssum = sum(sum_score[(p, t)] for t in _TBL_THRS if n_retr[(p, t)] > 0 and not pd.isna(cell[(p, t)]))
            nsum = sum(n_retr[(p, t)] for t in _TBL_THRS if n_retr[(p, t)] > 0 and not pd.isna(cell[(p, t)]))
            weighted[p] = ssum / nsum if nsum else np.nan
            vals = [cell[(p, t)] for t in _TBL_THRS if not pd.isna(cell[(p, t)])]
            simple[p] = float(np.mean(vals)) if vals else np.nan

        zs = base["ZEROSHOT"]
        adaptive_arr = {p: _tbl_adaptive(M, p, zs) for p in _TBL_PIPES}
        adaptive = {p: float(np.nanmean(adaptive_arr[p])) for p in _TBL_PIPES}

        print(f"\n{'Avg(simple)':<13}" + "".join(
            [f"{simple[p]:>12.2f}" for p in _TBL_PIPES] +
            [f"{np.mean(list(simple.values())):>12.2f}"] +
            [f"{base_mean_full[b]:>12.2f}" for b in _TBL_BASELINES]
        ))
        print(f"{'Avg(weighted)':<13}" + "".join(
            [f"{weighted[p]:>12.2f}" for p in _TBL_PIPES] +
            [f"{np.mean(list(weighted.values())):>12.2f}"] +
            [f"{base_mean_full[b]:>12.2f}" for b in _TBL_BASELINES]
        ))
        print(f"{'Avg(adaptive)':<13}" + "".join(
            [f"{adaptive[p]:>12.2f}" for p in _TBL_PIPES] +
            [f"{np.mean(list(adaptive.values())):>12.2f}"] +
            [f"{base_mean_full[b]:>12.2f}" for b in _TBL_BASELINES]
        ))

        hb1 = adaptive_arr["HB1"]
        print(f"\n  ## HB1 (adaptive) vs other pipelines/baselines  -- Wilcoxon paired + Cliff's delta")
        print(f"  {'opponent':<22} {'opp μ':>7} {'HB1 μ':>7} {'Δ':>7} {'W':>10} {'p':>10} {'Cliff δ':>9} {'ES':<12}")
        rivals = [(f"{p} (adaptive)", adaptive_arr[p]) for p in _TBL_PIPES if p != "HB1"]
        rivals += [(b, base[b]) for b in _TBL_BASELINES if base[b] is not None]
        for name, opp in rivals:
            mask = (~np.isnan(hb1)) & (~np.isnan(opp))
            a, o = hb1[mask], opp[mask]
            if len(a) < 5:
                print(f"  {name:<22} insufficient data"); continue
            try:
                W, p_ = stats.wilcoxon(a, o, zero_method="wilcox")
            except Exception:
                W, p_ = float("nan"), float("nan")
            d = _tbl_cliff_d(a, o)
            sig = "***" if p_ < 1e-3 else ("**" if p_ < 1e-2 else ("*" if p_ < 5e-2 else "n.s."))
            print(f"  {name:<22} {o.mean():>7.2f} {a.mean():>7.2f} {a.mean() - o.mean():>+7.2f} "
                  f"{W:>10.1f} {p_:>10.2e} {d:>+9.2f} {_tbl_es_label(d):<8}{sig}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="RQ1 evaluation")
    ap.add_argument("command", choices=["significance", "table-ii"])
    args = ap.parse_args()
    {"significance": cmd_significance, "table-ii": cmd_table_ii}[args.command]()


if __name__ == "__main__":
    main()
