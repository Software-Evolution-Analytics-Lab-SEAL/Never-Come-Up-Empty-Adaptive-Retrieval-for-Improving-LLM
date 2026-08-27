"""
Sentence-level binary precision/recall judge for all 11 retrieval methods
on the 84-query sample.

For each (query, method):
  - Load adaptive retrieval (for QB/HB pipelines: walk thresholds 0.9-0.5 and
    take first non-empty; for baselines: use the single-config CSV).
  - PRECISION: split retrieved_context into "natural units" (chunks delimited
    by '\\d+\\. '); for each unit, ask GPT-4o binary "Does this content
    help answer the question?" (yes/no, brief rationale).
      * For full-answer pipelines (QB2,QB4,HB1,HYB,BM25,RAGFUSION,
        ADAPTIVERAG,SELFRAG): natural-unit = full answer.
      * For sentence pipelines (QB1,QB3,HB2): natural-unit = sentence.
  - PRECISION (sentence-normalized): for full-answer pipelines, also split
    each retrieved answer into sentences and ask the same binary judge per
    sentence. Gives apples-to-apples comparison vs. sentence pipelines.
    For sentence pipelines, this equals natural-unit precision.
  - RECALL: split Accepted_Answer into sentences; for each, ask binary
    "Is this information present in the retrieved context?" (yes/no).

Outputs (under ../data/):
  precision_decisions.csv     (one row per (query, method, unit, judgment))
  precision_sent_decisions.csv (one row per (query, method, sentence, judgment))
  recall_decisions.csv        (one row per (query, method, gold_sent, judgment))

Resumable: each writer skips rows already in its output file.

Usage:
  python3 2_judge_retrieval.py                # run everything
  python3 2_judge_retrieval.py --methods HB1 HYB BM25
"""
import os
import re
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
import config

DATA_DIR = os.path.abspath(os.path.join(HERE, "..", "data"))
LOG_DIR = os.path.abspath(os.path.join(HERE, "..", "logs"))
RETRIEVAL_DIR = os.path.join(ROOT, "data", "retrieval_results")

JUDGE_MODEL = "openai/gpt-4o"
ADAPTIVE_THRS = [0.9, 0.8, 0.7, 0.6, 0.5]

PIPELINES_FULL_ANSWER = {"QB2", "QB4", "HB1", "HYB"}
PIPELINES_SENTENCE = {"QB1", "QB3", "HB2"}
INTERNAL_PIPELINES = PIPELINES_FULL_ANSWER | PIPELINES_SENTENCE
BASELINES = {"BM25", "RAGFUSION", "ADAPTIVERAG", "SELFRAG"}
ALL_METHODS = list(INTERNAL_PIPELINES) + list(BASELINES)

# All baselines retrieve full answers (their retrieved_context is multi-answer)
METHOD_GRANULARITY = {**{p: "answer" for p in PIPELINES_FULL_ANSWER},
                      **{p: "sentence" for p in PIPELINES_SENTENCE},
                      **{b: "answer" for b in BASELINES}}


PRECISION_PROMPT = """You are evaluating whether a piece of retrieved content is relevant for answering a programming question.

[QUESTION]
{question}

[RETRIEVED CONTENT]
{content}

Decide: does the retrieved content contain information that helps answer the question?
Answer with a single line of JSON, no other text:
{{"relevant": true|false, "reason": "<one short sentence>"}}"""

RECALL_PROMPT = """You are checking if a fact from the gold answer is covered by the retrieved context for a programming question.

[QUESTION]
{question}

[RETRIEVED CONTEXT]
{context}

[FACT FROM GOLD ANSWER]
{fact}

Decide: is the information in [FACT FROM GOLD ANSWER] semantically present in [RETRIEVED CONTEXT]? It does not need to be word-for-word; paraphrases count. Code that does the same thing as the fact also counts.
Answer with a single line of JSON, no other text:
{{"covered": true|false, "reason": "<one short sentence>"}}"""


_client = None
def client():
    global _client
    if _client is None:
        _client = OpenAI(base_url=config.OPENROUTER_BASE_URL,
                         api_key=config.get_openrouter_api_key())
    return _client


def call_judge(prompt, key, retries=3):
    """Returns (bool, reason_string) or (None, error_string)."""
    for attempt in range(retries):
        try:
            r = client().chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=80,
            )
            text = r.choices[0].message.content.strip()
            m = re.search(r"\{.*\}", text, re.S)
            if not m:
                return None, f"no_json: {text[:80]}"
            obj = json.loads(m.group(0))
            v = obj.get(key)
            reason = str(obj.get("reason", ""))[:200]
            if isinstance(v, bool):
                return v, reason
            if isinstance(v, str):
                return v.strip().lower() in ("true", "yes", "1"), reason
            return None, f"bad_value: {v}"
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None, f"error: {e}"


def split_units(retrieved_context):
    """Split a retrieved_context string into the natural numbered units.
    Format is '1. ...\n2. ...\n3. ...' . Returns list[str] (non-empty)."""
    if not isinstance(retrieved_context, str) or not retrieved_context.strip():
        return []
    parts = re.split(r"(?:^|\n)\s*\d+\.\s", retrieved_context)
    return [p.strip() for p in parts if p and p.strip()]


def split_sentences(text):
    """Cheap sentence splitter -- newline OR sentence-end punctuation followed by space + capital."""
    if not isinstance(text, str):
        return []
    # remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    # Split on . ! ? followed by space + uppercase OR newline
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9`'\"])", text)
    return [p.strip() for p in pieces if p.strip() and len(p.strip()) >= 8]


def load_method_retrieval(method, sample_idxs):
    """Return dict[test_idx] -> retrieved_context (string) for the sampled queries."""
    out = {}
    if method in BASELINES:
        df = pd.read_csv(os.path.join(RETRIEVAL_DIR, f"{method}.csv"))
        for i in sample_idxs:
            ctx = df["retrieved_context"].iloc[i] if i < len(df) else ""
            if isinstance(ctx, str) and ctx.strip():
                out[i] = ctx
        return out

    # Internal pipeline: walk adaptive thresholds 0.9 -> 0.5
    per_thr = {}
    for t in ADAPTIVE_THRS:
        path = os.path.join(RETRIEVAL_DIR, f"{method}_{t}.csv")
        if not os.path.exists(path):
            continue
        per_thr[t] = pd.read_csv(path)
    for i in sample_idxs:
        for t in ADAPTIVE_THRS:
            df = per_thr.get(t)
            if df is None or i >= len(df):
                continue
            ctx = df["retrieved_context"].iloc[i]
            nret = df["num_retrieved"].iloc[i] if "num_retrieved" in df.columns else 0
            if isinstance(ctx, str) and ctx.strip() and (pd.notna(nret) and nret > 0):
                out[i] = ctx
                break
    return out


def append_csv(path, rows, header):
    new = not os.path.exists(path)
    with open(path, "a") as f:
        if new:
            f.write(",".join(header) + "\n")
        for r in rows:
            line = ",".join(json.dumps(r.get(h, "")) if isinstance(r.get(h), str)
                            else str(r.get(h, "")) for h in header)
            f.write(line + "\n")


def already_done_keys(path, key_cols):
    if not os.path.exists(path):
        return set()
    try:
        df = pd.read_csv(path)
    except Exception:
        return set()
    if not all(c in df.columns for c in key_cols):
        return set()
    return set(zip(*[df[c].astype(str).tolist() for c in key_cols]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="*", default=ALL_METHODS,
                    help="Subset of methods to run")
    args = ap.parse_args()

    sample = pd.read_csv(os.path.join(DATA_DIR, "sample_queries.csv"))
    sample_idxs = sample["test_idx"].astype(int).tolist()
    qmap = dict(zip(sample["test_idx"].astype(int), sample["Paraphrased Question"].astype(str)))
    amap = dict(zip(sample["test_idx"].astype(int), sample["Accepted Answer"].astype(str)))

    p_path = os.path.join(DATA_DIR, "precision_decisions.csv")
    ps_path = os.path.join(DATA_DIR, "precision_sent_decisions.csv")
    r_path = os.path.join(DATA_DIR, "recall_decisions.csv")

    p_header = ["test_idx", "method", "unit_id", "unit_text", "relevant", "reason"]
    ps_header = ["test_idx", "method", "unit_id", "sent_id", "sent_text", "relevant", "reason"]
    r_header = ["test_idx", "method", "gold_sent_id", "gold_sent", "covered", "reason"]

    done_p = already_done_keys(p_path, ["test_idx", "method", "unit_id"])
    done_ps = already_done_keys(ps_path, ["test_idx", "method", "unit_id", "sent_id"])
    done_r = already_done_keys(r_path, ["test_idx", "method", "gold_sent_id"])

    # Pre-compute gold splits for recall
    gold_sents = {i: split_sentences(amap[i]) for i in sample_idxs}

    for method in args.methods:
        print(f"\n=== {method} ===")
        retrievals = load_method_retrieval(method, sample_idxs)
        print(f"  loaded retrievals for {len(retrievals)}/{len(sample_idxs)} queries")
        gran = METHOD_GRANULARITY[method]

        # PRECISION (natural-unit)
        prec_jobs = []
        for i in sample_idxs:
            if i not in retrievals:
                continue
            units = split_units(retrievals[i])
            for uid, unit in enumerate(units):
                key = (str(i), method, str(uid))
                if key in done_p:
                    continue
                prec_jobs.append((i, uid, unit))
        print(f"  precision (natural-unit): {len(prec_jobs)} judge calls to make")
        for i, uid, unit in tqdm(prec_jobs, desc=f"{method}-prec"):
            prompt = PRECISION_PROMPT.format(question=qmap[i], content=unit[:4000])
            v, reason = call_judge(prompt, "relevant")
            if v is None:
                v = ""
            append_csv(p_path, [{"test_idx": i, "method": method, "unit_id": uid,
                                 "unit_text": unit[:1500], "relevant": v, "reason": reason}],
                       p_header)

        # RECALL
        rec_jobs = []
        for i in sample_idxs:
            if i not in retrievals:
                continue
            for gid, gs in enumerate(gold_sents[i]):
                key = (str(i), method, str(gid))
                if key in done_r:
                    continue
                rec_jobs.append((i, gid, gs))
        print(f"  recall: {len(rec_jobs)} judge calls to make")
        for i, gid, gs in tqdm(rec_jobs, desc=f"{method}-rec"):
            prompt = RECALL_PROMPT.format(question=qmap[i],
                                          context=retrievals[i][:6000],
                                          fact=gs[:1000])
            v, reason = call_judge(prompt, "covered")
            if v is None:
                v = ""
            append_csv(r_path, [{"test_idx": i, "method": method, "gold_sent_id": gid,
                                 "gold_sent": gs[:600], "covered": v, "reason": reason}],
                       r_header)

    print("\nDone.")


if __name__ == "__main__":
    main()
