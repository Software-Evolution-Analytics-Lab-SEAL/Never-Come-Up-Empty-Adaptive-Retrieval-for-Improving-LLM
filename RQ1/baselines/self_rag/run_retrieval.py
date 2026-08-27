"""
Self-RAG retrieval stage on the 666 SE test set.

Faithful to Asai et al. (ICLR 2024):
  - Retriever: Contriever-MSMARCO (their default).
  - SO knowledge base re-encoded with Contriever (see encode_so_kb.sh).
  - Top-k: 5 (Self-RAG default for short-form QA).
  - Adaptive [Retrieve] decision: NOT done here — that decision is made by
    the Self-RAG generator at generation time using the [Retrieve] reflection
    token (threshold 0.2 on token probability). At retrieval stage we
    produce top-5 for every query; the generator decides whether to use
    them.

This script wraps Self-RAG's `passage_retrieval.py` with our query/output
format.

Inputs:
  - Encoded SO KB embedding shards: ../data/contriever_so_embeddings/passages_*
  - SO passage TSV (for id->text mapping): ../data/so_kb_passages.tsv
  - 666 test set: RQ1_n666/data/synthetic_questions_n666.csv

Output: RQ1_n666/data/retrieval_results/SELFRAG.csv

Usage:
    python3 run_retrieval.py [--gpu 0]
"""
import os
import sys
import json
import argparse
import time

# CUDA selection before torch import
_p = argparse.ArgumentParser(add_help=False)
_p.add_argument("--gpu", type=int, default=0)
_a, _ = _p.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(_a.gpu)

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
RQ1_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "RQ1_n666"))
sys.path.insert(0, RQ1_DIR)
import config

# Add Self-RAG retrieval_lm to path so we can import their Retriever class
SELFRAG_DIR = os.path.join(HERE, "upstream", "retrieval_lm")
sys.path.insert(0, SELFRAG_DIR)

import passage_retrieval  # noqa: E402
import src.contriever      # noqa: E402
import src.index           # noqa: E402
import src.data            # noqa: E402

CONTRIEVER = "facebook/contriever-msmarco"
PASSAGES_TSV = os.path.abspath(os.path.join(HERE, "..", "data", "so_kb_passages.tsv"))
EMBEDDINGS_GLOB = os.path.abspath(os.path.join(HERE, "..", "data", "contriever_so_embeddings", "passages_*"))
OUT_PATH = os.path.join(config.RETRIEVAL_DIR, "SELFRAG.csv")
N_DOCS = 5  # Self-RAG default for short-form QA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--input", type=str, default=config.TEST_SET_PATH)
    parser.add_argument("--n_docs", type=int, default=N_DOCS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if os.path.exists(OUT_PATH) and not args.force:
        print(f"[SELFRAG] [SKIP] {OUT_PATH} already exists (use --force to overwrite)")
        return

    # Build a mock args for Self-RAG's Retriever class
    class _RetrieverArgs:
        def __init__(self):
            self.model_name_or_path = CONTRIEVER
            self.passages = PASSAGES_TSV
            self.passages_embeddings = EMBEDDINGS_GLOB
            self.no_fp16 = False
            self.lowercase = False
            self.normalize_text = False
            self.question_maxlength = 512
            self.per_gpu_batch_size = 64
            self.indexing_batch_size = 1000000
            self.projection_size = 768
            self.n_subquantizers = 0
            self.n_bits = 8
            self.save_or_load_index = True
            self.n_docs = args.n_docs
    r_args = _RetrieverArgs()

    print(f"[SELFRAG] Setting up Self-RAG retriever (Contriever-MSMARCO, top-{args.n_docs}) ...")
    retriever = passage_retrieval.Retriever(r_args)
    retriever.setup_retriever()

    df = pd.read_csv(args.input)
    print(f"[SELFRAG] Loaded {len(df)} test queries")

    rows = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="SELFRAG"):
        q = row.get("Paraphrased Question", "")
        base = {
            "Question": row.get("Question", ""),
            "Accepted Answer": row.get("Accepted Answer", ""),
            "Paraphrased Question": q,
            "post_idx": row.get("post_idx", -1),
            "hypothetical_answer": "",
            "retrieved_post_ids": "",
            "retrieved_context": "",
            "retrieval_scores": "",
            "num_retrieved": 0,
            "generated_response": "",
        }
        if not isinstance(q, str) or not q.strip():
            rows.append(base)
            continue

        # Self-RAG's search_document returns a list of passage dicts
        # {id, text, title} (no scores in the basic search). For consistency
        # with our schema we'll use the order of returned passages as the
        # ranking and assign placeholder scores via the index.search_knn
        # output if available.
        q_emb = retriever.embed_queries(r_args, [q])
        top_ids_and_scores = retriever.index.search_knn(q_emb, args.n_docs)
        # top_ids_and_scores is [(ids, scores)] for the single query
        ids, scores = top_ids_and_scores[0]
        passages = [retriever.passage_id_map[pid] for pid in ids]

        ctx_lines = []
        retrieved_post_ids = []
        retrieval_scores = []
        for k, (pid, sc, p) in enumerate(zip(ids, scores, passages)):
            ctx_lines.append(f"{k+1}. {p['text']}")
            retrieved_post_ids.append(str(pid))
            retrieval_scores.append(f"{float(sc):.4f}")

        base["retrieved_post_ids"] = ",".join(retrieved_post_ids)
        base["retrieved_context"] = "\n".join(ctx_lines)
        base["retrieval_scores"] = ",".join(retrieval_scores)
        base["num_retrieved"] = len(passages)
        rows.append(base)

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"[SELFRAG] Saved {OUT_PATH} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
