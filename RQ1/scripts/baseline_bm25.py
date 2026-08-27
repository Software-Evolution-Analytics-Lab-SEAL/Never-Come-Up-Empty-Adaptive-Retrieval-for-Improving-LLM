"""
BM25 baseline over the OLD SO KB (full-answer corpus).

Single configuration: top-10 per query, no threshold sweep (BM25 scores are unbounded).

Output: data/retrieval_results/BM25.csv
"""
import os
import sys
import re
import pickle
import argparse

import pandas as pd
import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

BM25_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bm25_cache.pkl")

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text):
    return _TOKEN_RE.findall(str(text).lower())


def build_or_load_bm25():
    if os.path.exists(BM25_CACHE):
        print(f"[BM25] Loading cached BM25 from {BM25_CACHE}")
        with open(BM25_CACHE, "rb") as f:
            return pickle.load(f)

    print(f"[BM25] Loading answers_context.pkl ({os.path.getsize(config.ANSWERS_CONTEXT_PATH)/1e9:.1f} GB)")
    with open(config.ANSWERS_CONTEXT_PATH, "rb") as f:
        answers = pickle.load(f)
    print(f"[BM25]   {len(answers)} answers")

    print("[BM25] Tokenizing ...")
    corpus = [tokenize(a) for a in tqdm(answers, desc="tok")]
    print("[BM25] Building BM25Okapi index ...")
    bm25 = BM25Okapi(corpus)
    data = {"bm25": bm25, "texts": answers}

    print(f"[BM25] Caching to {BM25_CACHE}")
    with open(BM25_CACHE, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=config.TEST_SET_PATH)
    args = parser.parse_args()

    out_path = os.path.join(config.RETRIEVAL_DIR, "BM25.csv")
    if os.path.exists(out_path):
        print(f"[SKIP] {out_path} already exists")
        return

    data = build_or_load_bm25()
    bm25, texts = data["bm25"], data["texts"]

    df = pd.read_csv(args.input)
    print(f"[BM25] Loaded {len(df)} test queries")

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="BM25"):
        q_tokens = tokenize(row["Paraphrased Question"])
        if not q_tokens:
            rows.append({
                "Question": row.get("Question",""),
                "Accepted Answer": row.get("Accepted Answer",""),
                "Paraphrased Question": row["Paraphrased Question"],
                "post_idx": row.get("post_idx", -1),
                "hypothetical_answer": "",
                "retrieved_post_ids": "",
                "retrieved_context": "",
                "retrieval_scores": "",
                "num_retrieved": 0,
                "generated_response": "",
            })
            continue
        scores = bm25.get_scores(q_tokens)
        top_k = min(config.TOP_K, len(scores) - 1)
        top_idxs = np.argpartition(-scores, top_k)[:top_k]
        top_idxs = top_idxs[np.argsort(-scores[top_idxs])]
        # Filter out zero-score retrievals
        keep = [int(i) for i in top_idxs if scores[i] > 0]
        keep_scores = [float(scores[i]) for i in keep]
        context = "\n".join(f"{k+1}. {texts[i]}" for k, i in enumerate(keep))
        rows.append({
            "Question": row.get("Question",""),
            "Accepted Answer": row.get("Accepted Answer",""),
            "Paraphrased Question": row["Paraphrased Question"],
            "post_idx": row.get("post_idx", -1),
            "hypothetical_answer": "",
            "retrieved_post_ids": ",".join(str(i) for i in keep),
            "retrieved_context": context,
            "retrieval_scores": ",".join(f"{s:.4f}" for s in keep_scores),
            "num_retrieved": len(keep),
            "generated_response": "",
        })

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[BM25] Saved {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
