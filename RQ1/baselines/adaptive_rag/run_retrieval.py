"""
Adaptive-RAG retrieval stage on the 666 SE test set.

Faithful to Jeong et al. (NAACL 2024):
  - Retriever: BM25 (Robertson et al. 1994) — same retriever they use across
    all conditions. We reuse the pre-built rank_bm25 index from
    `RQ1_n666/bm25_cache.pkl` (built by `run_bm25.py`).
  - Classifier: T5-Large trained on their original silver labels
    (HotpotQA / NQ / TriviaQA / SQuAD / MuSiQue / 2Wiki, GPT-flavored).
    Trained off-the-shelf and applied to our SE questions — this matches
    "applying Adaptive-RAG out of the box to a new domain".
  - Routing:
        A → no retrieval (zero-shot at generation stage)
        B → BM25 single-step retrieval
        C → BM25 first-hop retrieval; iterative IRCoT hops are deferred to
            the generation stage (since they require the LLM)
  - Top-k: 10 (matches our other baselines and HB1; original IRCoT uses 15
            but Adaptive-RAG inherits IRCoT settings without prescribing k).

Inputs:
  - Trained classifier predictions: <args.predictions_json>
    (output of `run_classifier.py --do_eval` on `predict_n666.json`)
  - 666 test set: RQ1_n666/data/synthetic_questions_n666.csv
  - BM25 cache: RQ1_n666/bm25_cache.pkl

Output: RQ1_n666/data/retrieval_results/ADAPTIVERAG.csv

Usage:
    python3 run_retrieval.py \\
        --predictions_json /path/to/dict_id_pred_results.json
"""
import os
import sys
import re
import json
import pickle
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
from rank_bm25 import BM25Okapi

HERE = os.path.dirname(os.path.abspath(__file__))
RQ1_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "RQ1_n666"))
sys.path.insert(0, RQ1_DIR)
import config

OUT_PATH = os.path.join(config.RETRIEVAL_DIR, "ADAPTIVERAG.csv")
BM25_CACHE = os.path.join(RQ1_DIR, "bm25_cache.pkl")
TOP_K = 10  # matches other baselines

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text):
    return _TOKEN_RE.findall(str(text).lower())


def load_bm25():
    if not os.path.exists(BM25_CACHE):
        raise FileNotFoundError(
            f"BM25 cache not found at {BM25_CACHE}. Run RQ1_n666/run_bm25.py first."
        )
    print(f"[ADAPTIVERAG] Loading BM25 cache ({os.path.getsize(BM25_CACHE)/1e9:.1f} GB) ...")
    with open(BM25_CACHE, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["texts"]


def bm25_topk(bm25, texts, query, k=TOP_K):
    q_tokens = tokenize(query)
    if not q_tokens:
        return []
    scores = bm25.get_scores(q_tokens)
    if k >= len(scores):
        k = len(scores) - 1
    top_idxs = np.argpartition(-scores, k)[:k]
    top_idxs = top_idxs[np.argsort(-scores[top_idxs])]
    keep = [int(i) for i in top_idxs if scores[i] > 0]
    keep_scores = [float(scores[i]) for i in keep]
    return [(s, i, texts[i]) for s, i in zip(keep_scores, keep)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_json", type=str, required=True,
                        help="Path to dict_id_pred_results.json from run_classifier.py --do_eval")
    parser.add_argument("--input", type=str, default=config.TEST_SET_PATH)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if os.path.exists(OUT_PATH) and not args.force:
        print(f"[SKIP] {OUT_PATH} already exists (use --force to overwrite)")
        return

    # Load classifier predictions: {id: {"prediction": "A"/"B"/"C", ...}}
    with open(args.predictions_json) as f:
        preds_by_id = json.load(f)
    print(f"[ADAPTIVERAG] Loaded {len(preds_by_id)} classifier predictions from {args.predictions_json}")

    # Build id → label map
    id_to_label = {qid: rec.get("prediction", "B").strip() for qid, rec in preds_by_id.items()}
    counts = {"A": 0, "B": 0, "C": 0}
    for v in id_to_label.values():
        counts[v] = counts.get(v, 0) + 1
    print(f"[ADAPTIVERAG] Label distribution: A={counts.get('A',0)}, B={counts.get('B',0)}, C={counts.get('C',0)}")

    df = pd.read_csv(args.input)
    print(f"[ADAPTIVERAG] Loaded {len(df)} test queries")

    bm25, texts = load_bm25()

    rows = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="ADAPTIVERAG"):
        q = row.get("Paraphrased Question", "")
        post_idx = int(row.get("post_idx", i))
        rec_id = f"so_n666__{post_idx}__{i}"
        label = id_to_label.get(rec_id, "B")

        base = {
            "Question": row.get("Question", ""),
            "Accepted Answer": row.get("Accepted Answer", ""),
            "Paraphrased Question": q,
            "post_idx": post_idx,
            "hypothetical_answer": "",
            "retrieved_post_ids": "",
            "retrieved_context": "",
            "retrieval_scores": "",
            "num_retrieved": 0,
            "generated_response": "",
            "complexity_label": label,
            "needs_hop2": label == "C",
        }

        if not isinstance(q, str) or not q.strip():
            rows.append(base)
            continue

        if label == "A":
            # Path A: no retrieval (generation stage will do zero-shot)
            rows.append(base)
            continue

        # Path B: single BM25 retrieval. Path C: same first-hop BM25 retrieval;
        # subsequent IRCoT hops require the LLM and are deferred to generation.
        items = bm25_topk(bm25, texts, q)
        if items:
            ids = [str(it[1]) for it in items]
            ctx = "\n".join(f"{k+1}. {it[2]}" for k, it in enumerate(items))
            sc = ",".join(f"{it[0]:.4f}" for it in items)
            base["retrieved_post_ids"] = ",".join(ids)
            base["retrieved_context"] = ctx
            base["retrieval_scores"] = sc
            base["num_retrieved"] = len(items)
        rows.append(base)

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"[ADAPTIVERAG] Saved {OUT_PATH} ({len(rows)} rows)")
    n_retrieved = sum(1 for r in rows if r["num_retrieved"] > 0)
    print(f"[ADAPTIVERAG] Coverage: {n_retrieved}/{len(rows)} got >=1 BM25 hit")


if __name__ == "__main__":
    main()
