"""
HyDE robustness experiment: swap GPT-4o for Qwen3-8B as the HyDE generator
and check whether HB1's retrieval and downstream answer quality stay
consistent on a 54-query sample from the synthetic test set.

Pipeline arms (everything held constant except the HyDE step):
  Arm A (baseline): GPT-4o HyDE -> mpnet embed -> KB top-10 adaptive 0.9->0.5
                     -> LLaMa-3.1-8B final generation -> GPT-4o judge
  Arm B (test):     Qwen3-8B HyDE -> ... same as above ...

For Arm A we reuse the already-cached HyDE strings, retrievals, generations,
and judge scores under data/. For Arm B we generate the Qwen3-8B HyDE here,
embed and retrieve against the same KB, generate the final answer with
LLaMa-3.1-8B and score with the LLM-as-a-Judge.

Outputs (under data/hyde_robustness/):
  qwen3_hyde.json                       -- HyDE strings for the 54 queries
  qwen3_retrievals.csv                  -- adaptive retrieval per query
  qwen3_final_generations.csv           -- LLaMa final answer + judge
  comparison_summary.txt                -- mean retrieval-usefulness, judge
                                           score, paired Wilcoxon, Cliff's d
"""
import os
import sys
import json
import time
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from openai import OpenAI

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config
import kb

SAMPLE_N = 54
SEED = 42
HYDE_MODEL = "qwen/qwen3-8b"
DOWNSTREAM_MODEL = "meta-llama/llama-3.1-8b-instruct"
JUDGE_MODEL = "openai/gpt-4o"

OUT_DIR = os.path.join(config.DATA_DIR, "hyde_robustness")
os.makedirs(OUT_DIR, exist_ok=True)

# Reuse the existing HyDE cache (GPT-4o results) for Arm A
HYDE_CACHE_GPT4O = config.HYDE_CACHE_PATH
ENCODER = "sentence-transformers/all-mpnet-base-v2"
THRESHOLDS = [0.9, 0.8, 0.7, 0.6, 0.5]
TOP_K = 10


def _client():
    api_key = config.get_openrouter_api_key()
    return OpenAI(base_url=config.OPENROUTER_BASE_URL, api_key=api_key)


HYDE_SYSTEM = ("You are a senior software engineer. Given a Stack Overflow "
               "question, write the hypothetical accepted answer that a "
               "developer might post. Be concise and technically specific. "
               "Output only the answer body, no preamble.")
DOWN_SYSTEM = ("You are a senior software engineer answering a Stack Overflow "
               "question. Use the retrieved context as supporting material, "
               "but write your own concise, technically correct answer with "
               "complete code where appropriate.")
DOWN_USER = ("[QUESTION]\n{q}\n\n[RETRIEVED CONTEXT]\n{ctx}\n\nWrite the answer.")
JUDGE_SYSTEM = """You are an expert programming evaluator. Rate the following answer to a Stack Overflow question.
Please evaluate on these criteria:
1. Helpfulness - Does the answer address the question?
2. Technical Correctness - Is the answer technically accurate?
3. Level of Detail - Does the answer provide sufficient detail and explanation?
If there is incomplete code snippet in the response, deduct the overall score.
Output a single integer score from 1 to 10, where 10 is the best.
Output ONLY the number, nothing else."""
JUDGE_USER = ("[QUESTION]\n{q}\n\n[ANSWER TO EVALUATE]\n{a}\n\n"
              "[REFERENCE ACCEPTED ANSWER]\n{gold}")


def call_openrouter(client, model, system, user, max_tokens, temperature=0.7, retries=3):
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=temperature, max_tokens=max_tokens,
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return f"[ERROR] {e}"


def judge_one(client, q, ans, gold, retries=3):
    user = JUDGE_USER.format(q=q[:1500], a=ans[:3500], gold=gold[:3500])
    import re
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "system", "content": JUDGE_SYSTEM},
                          {"role": "user", "content": user}],
                temperature=0.3, max_tokens=20,
            )
            txt = r.choices[0].message.content.strip()
            for tok in re.findall(r"\d+", txt):
                s = int(tok)
                if 1 <= s <= 10:
                    return s
            return -1
        except Exception:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return -1


def sample_queries():
    df = pd.read_csv(config.TEST_SET_PATH).reset_index(drop=False).rename(columns={"index": "test_idx"})
    rng = np.random.default_rng(SEED)
    pick = sorted(rng.choice(len(df), size=SAMPLE_N, replace=False).tolist())
    return df.iloc[pick].reset_index(drop=True)


def generate_qwen3_hyde(sample, force=False):
    out = os.path.join(OUT_DIR, "qwen3_hyde.json")
    cache = {}
    if os.path.exists(out) and not force:
        with open(out) as f: cache = json.load(f)
    client = _client()
    print(f"Generating Qwen3-8B HyDE for {len(sample)} queries (cache hits = {len(cache)})")
    for _, r in tqdm(sample.iterrows(), total=len(sample), desc="qwen3-hyde"):
        q = str(r["Paraphrased Question"])
        if q in cache: continue
        h = call_openrouter(client, HYDE_MODEL, HYDE_SYSTEM, q, max_tokens=400)
        cache[q] = h
    with open(out, "w") as f: json.dump(cache, f, indent=2)
    return cache


def adaptive_retrieve(query_emb, kb_norm, thresholds=THRESHOLDS, top_k=TOP_K):
    # query_emb shape (768,), kb_norm shape (N, 768)
    sims = (kb_norm @ query_emb)
    for t in thresholds:
        mask = sims >= t
        if mask.any():
            cand_idx = torch.where(mask)[0]
            cand_sim = sims[cand_idx]
            order = torch.argsort(cand_sim, descending=True)
            keep = cand_idx[order][:top_k]
            return float(t), keep.cpu().numpy().tolist(), sims[keep].cpu().numpy().tolist()
    return None, [], []


def run_retrieval(sample, hyde_map, label):
    from sentence_transformers import SentenceTransformer
    print(f"[{label}] Loading encoder + KB embeddings ...")
    enc = SentenceTransformer(ENCODER, device="cuda" if torch.cuda.is_available() else "cpu")
    kb = torch.load(config.ALL_FULL_ANS_EMB_PATH, map_location="cpu").to(dtype=torch.float32)
    kb_norm = kb / kb.norm(dim=1, keepdim=True).clamp_min(1e-9)
    if torch.cuda.is_available(): kb_norm = kb_norm.cuda()

    answers_ctx, _ = kb.full_answer_pool()
    rows = []
    for _, r in tqdm(sample.iterrows(), total=len(sample), desc=f"{label}-retrieve"):
        q = str(r["Paraphrased Question"])
        h = hyde_map.get(q, "")
        if not h or h.startswith("[ERROR]"):
            rows.append({"test_idx": int(r["test_idx"]), "Paraphrased Question": q,
                         "hyde": h, "threshold_hit": None, "retrieved_idxs": "",
                         "retrieved_context": "", "num_retrieved": 0,
                         "Accepted Answer": r.get("Accepted Answer", "")})
            continue
        e = enc.encode(h, convert_to_tensor=True, show_progress_bar=False)
        e = e.to(dtype=torch.float32)
        if torch.cuda.is_available(): e = e.cuda()
        e = e / e.norm().clamp_min(1e-9)
        thr, idxs, sims = adaptive_retrieve(e, kb_norm)
        ctx = "\n".join(f"{k+1}. {answers_ctx[i]}" for k, i in enumerate(idxs))
        rows.append({"test_idx": int(r["test_idx"]), "Paraphrased Question": q,
                     "hyde": h, "threshold_hit": thr, "retrieved_idxs": ",".join(map(str, idxs)),
                     "retrieved_context": ctx, "num_retrieved": len(idxs),
                     "Accepted Answer": r.get("Accepted Answer", "")})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, f"{label}_retrievals.csv"), index=False)
    return df


def run_downstream(retrievals, label):
    client = _client()
    out = os.path.join(OUT_DIR, f"{label}_final_generations.csv")
    if os.path.exists(out):
        prev = pd.read_csv(out)
        if len(prev) == len(retrievals) and "generated_response" in prev.columns:
            retrievals["generated_response"] = prev["generated_response"]
        else:
            retrievals["generated_response"] = ""
    else:
        retrievals["generated_response"] = ""

    def task(i, row):
        if str(row.get("generated_response") or "").strip():
            return i, str(row["generated_response"])
        q = str(row["Paraphrased Question"])
        ctx = str(row["retrieved_context"])
        ans = call_openrouter(client, DOWNSTREAM_MODEL, DOWN_SYSTEM,
                               DOWN_USER.format(q=q, ctx=ctx[:6000]),
                               max_tokens=600, temperature=0.7)
        return i, ans

    todo = retrievals[retrievals["generated_response"].astype(str).str.strip() == ""].index.tolist()
    print(f"[{label}] downstream generation: {len(todo)} todo")
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(task, i, retrievals.iloc[i]) for i in todo]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{label}-gen"):
            i, ans = fut.result()
            retrievals.at[i, "generated_response"] = ans
    retrievals.to_csv(out, index=False)
    return retrievals


def run_judge(df, label):
    client = _client()
    out = os.path.join(OUT_DIR, f"{label}_final_generations.csv")
    if "judge_score" not in df.columns:
        df["judge_score"] = -1
    todo = df[(df["judge_score"] == -1) &
              (df["generated_response"].astype(str).str.strip() != "")].index.tolist()
    print(f"[{label}] judge: {len(todo)} todo")
    def task(i, row):
        s = judge_one(client, str(row["Paraphrased Question"]),
                       str(row["generated_response"]),
                       str(row["Accepted Answer"]))
        return i, s
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(task, i, df.iloc[i]) for i in todo]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{label}-judge"):
            i, s = fut.result()
            df.at[i, "judge_score"] = s
    df.to_csv(out, index=False)
    return df


def load_baseline_for_sample(sample):
    """Reuse existing GPT-4o-HyDE LLaMa-HB1 results.

    Use the same adaptive walk to pick the per-query final answer + judge."""
    eval_dir = os.path.join(config.DATA_DIR, "evaluation", "llama-3.1-8b")
    out = []
    test_idxs = sample["test_idx"].astype(int).tolist()
    per = {i: {"judge": None, "ans": ""} for i in test_idxs}
    for t in THRESHOLDS:
        path = os.path.join(eval_dir, f"HB1_{t}.csv")
        if not os.path.exists(path): continue
        df = pd.read_csv(path)
        s = pd.to_numeric(df["judge_score"], errors="coerce")
        nret = pd.to_numeric(df.get("num_retrieved", pd.Series([0]*len(df))), errors="coerce")
        for i in test_idxs:
            if per[i]["judge"] is not None: continue
            if i >= len(df): continue
            if nret.iloc[i] >= 1 and s.iloc[i] > 0:
                per[i]["judge"] = float(s.iloc[i])
                per[i]["ans"] = str(df["generated_response"].iloc[i])
    for i in test_idxs:
        out.append({"test_idx": i,
                    "generated_response": per[i]["ans"],
                    "judge_score": per[i]["judge"] if per[i]["judge"] is not None else -1})
    return pd.DataFrame(out)


def compare(baseline, qwen3):
    from scipy.stats import wilcoxon
    a = pd.merge(baseline[["test_idx", "judge_score"]].rename(columns={"judge_score": "baseline"}),
                 qwen3[["test_idx", "judge_score"]].rename(columns={"judge_score": "qwen3"}),
                 on="test_idx")
    a = a[(a["baseline"] > 0) & (a["qwen3"] > 0)]
    summary = []
    summary.append(f"Paired n: {len(a)}")
    summary.append(f"Baseline (GPT-4o HyDE) mean = {a['baseline'].mean():.3f}, median = {a['baseline'].median():.1f}")
    summary.append(f"Qwen3-8B HyDE mean = {a['qwen3'].mean():.3f}, median = {a['qwen3'].median():.1f}")
    summary.append(f"Mean delta (qwen3 - baseline) = {a['qwen3'].mean() - a['baseline'].mean():+.3f}")
    try:
        w, p = wilcoxon(a["qwen3"], a["baseline"], zero_method="wilcox")
        summary.append(f"Paired Wilcoxon: W = {w:.1f}, p = {p:.3e}")
    except ValueError as e:
        summary.append(f"Paired Wilcoxon: {e}")
    bv = a["baseline"].to_numpy(); qv = a["qwen3"].to_numpy()
    gt = sum((q > bv).sum() for q in qv); lt = sum((q < bv).sum() for q in qv)
    cliff = (gt - lt) / (len(qv) * len(bv))
    summary.append(f"Cliff's delta (qwen3 vs baseline) = {cliff:+.3f}")
    return "\n".join(summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-hyde", action="store_true")
    args = ap.parse_args()

    sample = sample_queries()
    sample.to_csv(os.path.join(OUT_DIR, "sample.csv"), index=False)
    print(f"Sampled {len(sample)} queries (seed={SEED})")

    # --- Arm B: Qwen3-8B HyDE pipeline ---
    if not args.skip_hyde:
        hyde = generate_qwen3_hyde(sample)
    else:
        with open(os.path.join(OUT_DIR, "qwen3_hyde.json")) as f: hyde = json.load(f)
    qwen3_retr = run_retrieval(sample, hyde, "qwen3")
    qwen3_gen = run_downstream(qwen3_retr, "qwen3")
    qwen3_judged = run_judge(qwen3_gen, "qwen3")

    # --- Arm A: baseline (GPT-4o HyDE, LLaMa downstream) ---
    baseline = load_baseline_for_sample(sample)
    baseline.to_csv(os.path.join(OUT_DIR, "baseline_judged.csv"), index=False)

    summary = compare(baseline, qwen3_judged)
    print("\n" + "=" * 60 + "\n" + summary)
    with open(os.path.join(OUT_DIR, "comparison_summary.txt"), "w") as f:
        f.write(summary + "\n")


if __name__ == "__main__":
    main()
