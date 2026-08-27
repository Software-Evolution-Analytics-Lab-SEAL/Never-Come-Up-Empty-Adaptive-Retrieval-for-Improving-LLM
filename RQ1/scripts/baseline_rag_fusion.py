"""
RAG-Fusion baseline: multi-query expansion + Reciprocal Rank Fusion (RRF).

For each test question:
  1. Generate 4 paraphrased query variants via GPT-4o (cached).
  2. Encode original + 4 variants = 5 queries.
  3. For each query, dense-search the full-answer pool for top-(5*TOP_K).
  4. RRF-merge with k=60: score(d) = sum_i 1 / (60 + rank_i(d)).
  5. Output: top-TOP_K by RRF score.

Single configuration. Output: data/retrieval_results/RAGFUSION.csv
"""
import os
import sys
import time
import json
import re
import argparse

# CUDA selection before torch import
_p = argparse.ArgumentParser(add_help=False)
_p.add_argument("--gpu", type=int, default=0)
_a, _ = _p.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(_a.gpu)

import torch
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import kb

N_VARIANTS = 4
RRF_K = 60
VARIANTS_CACHE = os.path.join(config.DATA_DIR, "rag_fusion_variants_cache.json")

VARIANT_SYSTEM = (
    f"You are an expert at reformulating programming questions. Given a question, "
    f"produce {N_VARIANTS} alternative phrasings that ask the same thing in "
    f"different words. Each variant on a single line, numbered 1-{N_VARIANTS}. "
    "No extra text, no explanations."
)


def load_variants_cache():
    if os.path.exists(VARIANTS_CACHE):
        with open(VARIANTS_CACHE) as f:
            return json.load(f)
    return {}


def save_variants_cache(cache):
    with open(VARIANTS_CACHE, "w") as f:
        json.dump(cache, f, indent=2)


def generate_variants(client, question, retries=3):
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=config.GPT4O_MODEL,
                messages=[
                    {"role": "system", "content": VARIANT_SYSTEM},
                    {"role": "user", "content": question},
                ],
                temperature=0.7,
            )
            text = r.choices[0].message.content.strip()
            lines = []
            for line in text.split("\n"):
                line = re.sub(r"^\d+[\.\)]\s*", "", line.strip())
                if line:
                    lines.append(line)
            return lines[:N_VARIANTS]
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  [variant ERROR] {e}")
                return []


def ensure_variants(questions):
    cache = load_variants_cache()
    missing = [q for q in questions if q not in cache]
    if missing:
        print(f"[RAGFUSION] Generating variants for {len(missing)} queries (cached: {len(cache)})")
        api_key = config.get_openrouter_api_key()
        client = OpenAI(base_url=config.OPENROUTER_BASE_URL, api_key=api_key)
        for q in tqdm(missing, desc="variants"):
            cache[q] = generate_variants(client, q)
            time.sleep(0.05)
        save_variants_cache(cache)
    return cache


def rrf_merge(result_lists, k=RRF_K, top_k=config.TOP_K):
    rrf_scores = {}
    for results in result_lists:
        for rank, idx in enumerate(results):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    merged = sorted(rrf_scores.items(), key=lambda x: -x[1])
    return merged[:top_k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--input", type=str, default=config.TEST_SET_PATH)
    args = parser.parse_args()

    out_path = os.path.join(config.RETRIEVAL_DIR, "RAGFUSION.csv")
    if os.path.exists(out_path):
        print(f"[SKIP] {out_path} already exists")
        return

    df = pd.read_csv(args.input)
    print(f"[RAGFUSION] Loaded {len(df)} test queries")

    questions = df["Paraphrased Question"].fillna("").tolist()
    variants = ensure_variants(questions)

    print("[RAGFUSION] Loading full-answer pool ...")
    fa_text, fa_emb = kb.full_answer_pool()
    enc = kb.get_encoder()
    fa_emb = fa_emb.to(enc.device)

    depth = config.TOP_K * 5  # 50

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="RAGFUSION"):
        q = row["Paraphrased Question"]
        vs = variants.get(q, [])
        all_queries = [q] + vs
        result_lists = []
        for qt in all_queries:
            if not qt:
                continue
            q_emb = kb.encode_query(qt)
            scores = kb.cos_sim(q_emb, fa_emb).squeeze(0).cpu()
            top_idxs = torch.topk(scores, k=depth).indices.tolist()
            result_lists.append(top_idxs)

        merged = rrf_merge(result_lists)
        if not merged:
            rows.append({
                "Question": row.get("Question",""),
                "Accepted Answer": row.get("Accepted Answer",""),
                "Paraphrased Question": q,
                "post_idx": row.get("post_idx", -1),
                "hypothetical_answer": "",
                "retrieved_post_ids": "",
                "retrieved_context": "",
                "retrieval_scores": "",
                "num_retrieved": 0,
                "generated_response": "",
            })
            continue

        ids = [i for i, _ in merged]
        sc = [f"{s:.4f}" for _, s in merged]
        context = "\n".join(f"{k+1}. {fa_text[i]}" for k, i in enumerate(ids))
        rows.append({
            "Question": row.get("Question",""),
            "Accepted Answer": row.get("Accepted Answer",""),
            "Paraphrased Question": q,
            "post_idx": row.get("post_idx", -1),
            "hypothetical_answer": "",
            "retrieved_post_ids": ",".join(str(i) for i in ids),
            "retrieved_context": context,
            "retrieval_scores": ",".join(sc),
            "num_retrieved": len(ids),
            "generated_response": "",
        })

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[RAGFUSION] Saved {out_path}")


if __name__ == "__main__":
    main()
