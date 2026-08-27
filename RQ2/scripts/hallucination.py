"""
Hallucination analysis for RQ2 — three-class taxonomy on the 4 small LLMs.

Every claim in a generated answer is classified as:
  - supported   : explicitly stated or directly implied by the accepted answer
  - compatible  : not in gold, but correct programming knowledge / valid alternative
  - contradicts : factually wrong or fabricated (real hallucination)

Real hallucination rate = (# contradicts) / (# total claims).

Subcommands:
    check      Run the GPT-4o judge over generated answers.
               Output: data/evaluation/unseen_2025/{model}/{pipeline}_hallucination.csv
    aggregate  Roll the per-answer CSVs into a Table III summary and per-model
               paired Wilcoxon (HB1 vs ZS).
               Output: hallucination_summary.csv + hallucination_pairwise.csv

Usage:
    python hallucination.py check
    python hallucination.py check --models llama-3.1-8b --pipelines adaptive
    python hallucination.py aggregate
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from openai import OpenAI
from scipy.stats import wilcoxon
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config


GEN_DIR = os.path.join(config.DATA_DIR, "generation", "unseen_2025")
EVAL_DIR = os.path.join(config.DATA_DIR, "evaluation", "unseen_2025")
JUDGE_MODEL = "openai/gpt-4o"
SMALL_MODELS = ["llama-3.1-8b", "mistral-7b", "qwen3-8b", "granite-3.1-8b"]
PIPELINES_CHECK = ["adaptive", "zeroshot"]
PIPELINES_AGG = [("HB1", "adaptive"), ("ZS", "zeroshot")]


# ===========================================================================
# check — GPT-4o judge over generated answers
# ===========================================================================

JUDGE_SYSTEM = """You are an expert programming evaluator detecting real hallucinations in a generated answer.

You will receive a programming question, a generated answer, and the official accepted answer (gold reference). Identify every factual or technical claim in the generated answer and classify each into ONE of these three categories:

- "supported": the claim is explicitly stated or directly implied by the accepted answer.
- "compatible": the claim is NOT in the accepted answer, but it is correct programming knowledge OR a valid alternative approach. It does NOT contradict the gold and would not mislead the developer.
- "contradicts": the claim is factually wrong, contradicts the accepted answer, or contains a fabricated fact (wrong API name, non-existent library, incorrect behavior). This is a real hallucination.

Boilerplate ("I hope this helps", "feel free to ask"), code comments, and pure formatting do NOT count as claims.

Output a single line of JSON, no other text:
{"claims": [{"claim": "<short paraphrase>", "verdict": "supported|compatible|contradicts", "reason": "<one short sentence>"}], "hallucination_rate": <float 0..1>, "verdict_overall": "faithful|partial|hallucinated"}

hallucination_rate = (number of contradicts) / (total number of claims). If there are no checkable claims, output {"claims": [], "hallucination_rate": 0.0, "verdict_overall": "faithful"}.
verdict_overall: "faithful" if hallucination_rate == 0; "hallucinated" if all claims contradict; "partial" otherwise."""

JUDGE_USER_TEMPLATE = """[QUESTION]
{question}

[GENERATED ANSWER]
{answer}

[OFFICIAL ACCEPTED ANSWER]
{accepted}"""

_client_lock = threading.Lock()
_client = None


def _client_get():
    global _client
    with _client_lock:
        if _client is None:
            _client = OpenAI(base_url=config.OPENROUTER_BASE_URL,
                             api_key=config.get_openrouter_api_key())
        return _client


def _judge_one(question, answer, accepted, retries=3):
    prompt = JUDGE_USER_TEMPLATE.format(
        question=question[:1500], answer=answer[:3500], accepted=accepted[:3500],
    )
    for attempt in range(retries):
        try:
            r = _client_get().chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "system", "content": JUDGE_SYSTEM},
                          {"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=900,
            )
            text = r.choices[0].message.content.strip()
            m = re.search(r"\{.*\}", text, re.S)
            if not m:
                return {"hallu_status": "no_json", "total_claims": -1,
                        "supported": -1, "compatible": -1, "contradicts": -1,
                        "hallucination_rate": -1, "hallu_reason": text[:240]}
            obj = json.loads(m.group(0))
            claims = obj.get("claims", []) or []
            if not isinstance(claims, list): claims = []
            n_total = len(claims)
            n_sup = sum(1 for c in claims if c.get("verdict") == "supported")
            n_com = sum(1 for c in claims if c.get("verdict") == "compatible")
            n_con = sum(1 for c in claims if c.get("verdict") == "contradicts")
            rate = float(obj.get("hallucination_rate", n_con / n_total if n_total else 0.0))
            verdict = str(obj.get("verdict_overall", "")).strip().lower() or (
                "faithful" if rate == 0 else ("hallucinated" if rate >= 0.999 else "partial"))
            reason = ""
            for c in claims:
                if c.get("verdict") == "contradicts":
                    reason = str(c.get("reason", ""))[:200]
                    break
            if not reason and claims:
                reason = str(claims[0].get("reason", ""))[:200]
            return {
                "hallu_status": verdict, "total_claims": n_total,
                "supported": n_sup, "compatible": n_com, "contradicts": n_con,
                "hallucination_rate": rate, "hallu_reason": reason,
                "claims_json": json.dumps(claims, ensure_ascii=False)[:6000],
            }
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {"hallu_status": "error", "total_claims": -1,
                        "supported": -1, "compatible": -1, "contradicts": -1,
                        "hallucination_rate": -1,
                        "hallu_reason": f"error: {e}"[:240]}


def _process_check(model, pipeline, workers=8):
    src = os.path.join(GEN_DIR, model, f"{pipeline}.csv")
    if not os.path.exists(src):
        print(f"[{model}/{pipeline}] missing source: {src}")
        return
    df = pd.read_csv(src)
    out_dir = os.path.join(EVAL_DIR, model)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{pipeline}_hallucination.csv")

    needed = ["hallu_status", "total_claims", "supported", "compatible",
              "contradicts", "hallucination_rate", "hallu_reason", "claims_json"]
    if os.path.exists(out):
        prior = pd.read_csv(out)
        if len(prior) == len(df) and all(c in prior.columns for c in needed):
            for c in needed:
                df[c] = prior[c]
            print(f"[{model}/{pipeline}] resumed from {out}")
        else:
            for c in needed:
                df[c] = "" if c in ("hallu_status", "hallu_reason", "claims_json") else -1
    else:
        for c in needed:
            df[c] = "" if c in ("hallu_status", "hallu_reason", "claims_json") else -1

    todo_idx = df.index[
        ~pd.to_numeric(df["hallucination_rate"], errors="coerce").between(0, 1)
    ].tolist()
    print(f"[{model}/{pipeline}] total={len(df)}  done={len(df)-len(todo_idx)}  todo={len(todo_idx)}")
    if not todo_idx:
        return

    write_lock = threading.Lock()
    saved = [0]

    def task(i):
        row = df.iloc[i]
        q = str(row.get("Title") or row.get("Paraphrased Question") or "")
        ans = str(row.get("generated_response") or "")
        gold = str(row.get("Accepted_Answer") or row.get("Accepted Answer") or "")
        if not ans.strip() or ans.strip().lower() in ("nan", "none") or ans.startswith("[ERROR]"):
            return i, {"hallu_status": "empty", "total_claims": 0,
                       "supported": 0, "compatible": 0, "contradicts": 0,
                       "hallucination_rate": -1,
                       "hallu_reason": "empty answer", "claims_json": "[]"}
        return i, _judge_one(q, ans, gold)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(task, i) for i in todo_idx]
        with tqdm(total=len(futures), desc=f"{model}/{pipeline}") as pbar:
            for fut in as_completed(futures):
                i, res = fut.result()
                with write_lock:
                    for k, v in res.items():
                        df.at[i, k] = v
                    saved[0] += 1
                    if saved[0] % 100 == 0:
                        df.to_csv(out, index=False)
                pbar.update(1)
    df.to_csv(out, index=False)
    n_valid = (pd.to_numeric(df["hallucination_rate"], errors="coerce") >= 0).sum()
    n_zero = (pd.to_numeric(df["hallucination_rate"], errors="coerce") == 0).sum()
    print(f"[{model}/{pipeline}] saved {out}  valid={n_valid}  zero_hallucination={n_zero}/{n_valid}")


def cmd_check(args):
    for m in args.models:
        for p in args.pipelines:
            _process_check(m, p, workers=args.workers)


# ===========================================================================
# aggregate — summary table + paired Wilcoxon
# ===========================================================================

_SEED = 42
_B = 1000


def _bootstrap_ci(arr, B=_B, seed=_SEED):
    rng = np.random.default_rng(seed)
    a = np.asarray(arr, dtype=float); a = a[~np.isnan(a)]
    if len(a) == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(a), size=(B, len(a)))
    means = a[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _cliff_d(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return (gt - lt) / (len(a) * len(b))


def _es_label(d):
    if np.isnan(d): return "-"
    ad = abs(d)
    if ad < 0.147: return "negligible"
    if ad < 0.33:  return "small"
    if ad < 0.474: return "medium"
    return "large"


def _load_agg(model, pipeline_subdir):
    p = os.path.join(EVAL_DIR, model, f"{pipeline_subdir}_hallucination.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    df["hallucination_rate"] = pd.to_numeric(df["hallucination_rate"], errors="coerce")
    return df


def cmd_aggregate(args):
    rows, pq = [], []
    for m in SMALL_MODELS:
        for label, sub in PIPELINES_AGG:
            df = _load_agg(m, sub)
            if df is None:
                continue
            v = df["hallucination_rate"]
            valid = v.between(0, 1)
            sub_v = v[valid]
            n = int(valid.sum())
            mean_rate = float(sub_v.mean())
            lo, hi = _bootstrap_ci(sub_v.to_numpy())
            n_zero = int((sub_v == 0).sum())
            n_full = int((sub_v >= 0.999).sum())
            total_claims = int(pd.to_numeric(df["total_claims"], errors="coerce").fillna(0).sum())
            n_contradicts = int(pd.to_numeric(df["contradicts"], errors="coerce").fillna(0).sum())
            n_supported = int(pd.to_numeric(df["supported"], errors="coerce").fillna(0).sum())
            n_compatible = int(pd.to_numeric(df["compatible"], errors="coerce").fillna(0).sum())
            rows.append({
                "model": m, "pipeline": label, "n_valid": n,
                "mean_hallu_rate": mean_rate, "ci_lo": lo, "ci_hi": hi,
                "pct_zero_hallucination": n_zero / n, "pct_full_hallucination": n_full / n,
                "any_hallu_rate": (n - n_zero) / n,
                "total_claims": total_claims, "contradicts": n_contradicts,
                "supported": n_supported, "compatible": n_compatible,
                "claim_contradicts_pct": n_contradicts / max(total_claims, 1),
            })
            for i, r in df[valid].iterrows():
                pq.append({"model": m, "pipeline": label,
                           "query_idx": int(r.get("query_idx", i)) if pd.notna(r.get("query_idx", i)) else i,
                           "hallu_rate": float(r["hallucination_rate"])})

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(EVAL_DIR, "hallucination_summary.csv"), index=False)
    pq_df = pd.DataFrame(pq)

    print("=== hallucination summary (3-class taxonomy) ===\n")
    hdr = (f"{'model':<18}{'pipe':<5}{'n':>6}{'mean rate (95% CI)':>26}"
           f"{'zero-hallu %':>14}{'any-hallu %':>14}{'claim contr %':>14}")
    print(hdr); print("-" * len(hdr))
    for _, r in summary.iterrows():
        ci = f"{r['mean_hallu_rate']:.3f} [{r['ci_lo']:.3f},{r['ci_hi']:.3f}]"
        print(f"{r['model']:<18}{r['pipeline']:<5}{int(r['n_valid']):>6}{ci:>26}"
              f"{r['pct_zero_hallucination']*100:>13.1f}%{r['any_hallu_rate']*100:>13.1f}%"
              f"{r['claim_contradicts_pct']*100:>13.1f}%")

    print("\n=== HB1 vs ZS hallucination rate (paired Wilcoxon, Cliff's δ) ===")
    print(f"{'model':<18}{'n':>6}{'HB1':>9}{'ZS':>9}{'Δ':>9}{'p':>11}{'Cliff δ':>10}{'ES':>14}")
    pair_rows = []
    for m in SMALL_MODELS:
        f1 = pq_df[pq_df["model"] == m].pivot_table(
            index="query_idx", columns="pipeline", values="hallu_rate")
        if "HB1" not in f1.columns or "ZS" not in f1.columns:
            continue
        pair = f1[["HB1", "ZS"]].dropna()
        if len(pair) < 5:
            continue
        a, b = pair["HB1"].to_numpy(), pair["ZS"].to_numpy()
        try:
            w, pv = wilcoxon(a, b, zero_method="wilcox")
        except ValueError:
            w, pv = float("nan"), float("nan")
        d = _cliff_d(a, b)
        delta = a.mean() - b.mean()
        sig = "***" if pv < 1e-3 else ("**" if pv < 1e-2 else ("*" if pv < 5e-2 else "n.s."))
        print(f"{m:<18}{len(pair):>6}{a.mean():>9.3f}{b.mean():>9.3f}{delta:>+9.3f}"
              f"{pv:>11.2e}{d:>+10.3f}{_es_label(d):>10} {sig}")
        pair_rows.append({"model": m, "n_paired": len(pair),
                          "mean_HB1": a.mean(), "mean_ZS": b.mean(),
                          "delta": delta, "p": pv, "cliff_d": d,
                          "es": _es_label(d), "sig": sig})
    pd.DataFrame(pair_rows).to_csv(
        os.path.join(EVAL_DIR, "hallucination_pairwise.csv"), index=False)
    print("\nSaved hallucination_summary.csv and hallucination_pairwise.csv")


# ===========================================================================

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_check = sub.add_parser("check", help="Run GPT-4o hallucination judge")
    ap_check.add_argument("--models", nargs="+", default=SMALL_MODELS)
    ap_check.add_argument("--pipelines", nargs="+", default=PIPELINES_CHECK)
    ap_check.add_argument("--workers", type=int, default=8)
    ap_check.set_defaults(func=cmd_check)

    ap_agg = sub.add_parser("aggregate", help="Summarize per-answer CSVs into Table III")
    ap_agg.set_defaults(func=cmd_aggregate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
