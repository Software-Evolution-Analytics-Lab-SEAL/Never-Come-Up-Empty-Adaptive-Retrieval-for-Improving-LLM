"""
Threshold sweep: judge per-unit Usefulness for HB1 and HYB at every
threshold 0.1 - 0.9 on the 84-query sample.

Reads retrieved_context from RQ1_n666/data/retrieval_results/{method}_{thr}.csv
and writes binary useful/not-useful decisions (with rationale) to
../data/usefulness_threshold_decisions.csv (resumable).

Optimization: judges each unique (method, test_idx, unit_text) ONCE and
replays the decision for every threshold where the same unit appears.
Also reuses decisions already stored in usefulness_decisions.csv (the
adaptive run from earlier). For HB1/HYB this typically cuts new judge
calls by ~50-70% since low thresholds share many units with adaptive.

Usage:
    python3 7_judge_threshold_sweep.py
    python3 7_judge_threshold_sweep.py --methods HB1
    python3 7_judge_threshold_sweep.py --thresholds 0.5 0.7 0.9
"""
import os
import sys
import csv
import argparse
import importlib.util

import pandas as pd
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
RETRIEVAL_DIR = os.path.join(ROOT, "data", "retrieval_results")
OUT_PATH = os.path.join(DATA, "usefulness_threshold_decisions.csv")
PRIOR_PATH = os.path.join(DATA, "usefulness_decisions.csv")

spec = importlib.util.spec_from_file_location("jr", os.path.join(HERE, "2_judge_retrieval.py"))
jr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jr)

DEFAULT_METHODS = ["HB1", "HYB"]
DEFAULT_THRS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def load_done_keys():
    """Existing rows in this script's output -> (test_idx, method, threshold, unit_id) keys."""
    if not os.path.exists(OUT_PATH):
        return set()
    df = pd.read_csv(OUT_PATH)
    if df.empty: return set()
    return set(zip(df["test_idx"].astype(str), df["method"].astype(str),
                   df["threshold"].astype(str), df["unit_id"].astype(str)))


def load_decision_cache():
    """Build (method, test_idx, unit_text_head) -> (useful, reason) from BOTH
    the prior adaptive run AND any existing rows in this sweep's output.
    Use the first 200 chars of unit_text as the dedup key (full text occasionally
    has CSV-quoting noise after a round-trip)."""
    cache = {}
    for path in (PRIOR_PATH, OUT_PATH):
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            method = str(r["method"])
            ti = int(r["test_idx"])
            text_key = str(r["unit_text"])[:200].strip()
            useful = str(r["useful"])
            reason = str(r.get("reason", ""))
            cache[(method, ti, text_key)] = (useful, reason)
    return cache


def append_row(row):
    new_file = not os.path.exists(OUT_PATH)
    with open(OUT_PATH, "a", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        if new_file:
            w.writerow(["test_idx", "method", "threshold", "unit_id",
                        "unit_text", "useful", "reason", "from_cache"])
        w.writerow([row["test_idx"], row["method"], row["threshold"],
                    row["unit_id"], row["unit_text"][:1500],
                    row["useful"], row["reason"], row["from_cache"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    ap.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRS)
    args = ap.parse_args()

    sample = pd.read_csv(os.path.join(DATA, "sample_queries.csv"))
    sample_idxs = sample["test_idx"].astype(int).tolist()
    qmap = dict(zip(sample["test_idx"].astype(int),
                    sample["Paraphrased Question"].astype(str)))

    done_rows = load_done_keys()
    cache = load_decision_cache()
    print(f"Resume: {len(done_rows)} (test_idx, method, thr, unit_id) rows already on disk")
    print(f"Decision cache: {len(cache)} unique (method, query, unit_text) judgments available")

    n_new_calls = n_cache_hits = 0
    for method in args.methods:
        for thr in args.thresholds:
            path = os.path.join(RETRIEVAL_DIR, f"{method}_{thr}.csv")
            if not os.path.exists(path):
                print(f"[{method}@{thr}] missing {path}; skipping")
                continue
            df = pd.read_csv(path)

            jobs = []
            for i in sample_idxs:
                if i >= len(df):
                    continue
                ctx = df["retrieved_context"].iloc[i]
                nret = df["num_retrieved"].iloc[i] if "num_retrieved" in df.columns else 0
                if not isinstance(ctx, str) or not ctx.strip() or pd.isna(nret) or nret <= 0:
                    continue
                units = jr.split_units(ctx)
                for uid, unit in enumerate(units):
                    key_row = (str(i), method, str(thr), str(uid))
                    if key_row in done_rows:
                        continue
                    jobs.append((i, uid, unit))

            if not jobs:
                print(f"[{method}@{thr}] all done; skipping")
                continue

            print(f"[{method}@{thr}] {len(jobs)} (query, unit) pairs to handle")
            for i, uid, unit in tqdm(jobs, desc=f"{method}@{thr}"):
                tkey = unit[:200].strip()
                ckey = (method, i, tkey)
                if ckey in cache:
                    useful, reason = cache[ckey]
                    n_cache_hits += 1
                    from_cache = "1"
                else:
                    prompt = jr.PRECISION_PROMPT.format(question=qmap[i], content=unit[:4000])
                    v, reason = jr.call_judge(prompt, "relevant")
                    useful = "" if v is None else str(v)
                    cache[ckey] = (useful, reason)
                    n_new_calls += 1
                    from_cache = "0"
                append_row({"test_idx": i, "method": method, "threshold": thr,
                            "unit_id": uid, "unit_text": unit,
                            "useful": useful, "reason": reason,
                            "from_cache": from_cache})

    print(f"\nDone. New judge calls: {n_new_calls}.  Cache hits: {n_cache_hits}.")
    print(f"Output: {OUT_PATH}")


if __name__ == "__main__":
    main()
