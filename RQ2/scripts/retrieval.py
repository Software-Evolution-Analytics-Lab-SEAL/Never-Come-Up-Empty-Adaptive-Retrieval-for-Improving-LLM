"""
Single retrieval runner for all 7 RAG pipeline variants (QB1-4, HB1-2, HYB).

Retrieval logic:
  - QB1 (Direct + Sentence):   encode question, score against ALL sentences,
                               threshold on query↔sentence cosine, top-10
  - QB2 (Direct + Full answer): encode question, score against ALL full answers,
                               threshold on query↔full_answer cosine, top-10
  - QB3 (Indirect + Sentence): encode question, search question pool top-10,
                               iterate sentences of those 10 posts (the
                               OLD per-query encoding), threshold on
                               query↔sentence, top-10
  - QB4 (Indirect + Full answer): encode question, search question pool top-10,
                                 take full answers of those 10 posts (look up
                                 cached full-answer embedding), threshold on
                                 query↔full_answer, top-10
  - HB1 (HyDE + Direct + Full answer): same as QB2 but with HyDE answer as query
  - HB2 (HyDE + Direct + Sentence):   same as QB1 but with HyDE answer as query
  - HYB (Hybrid):              stage-1 = question embedding → question pool,
                               top-10 posts;
                               stage-2 = HyDE answer → score against those 10
                               posts' full answers, threshold, top-10

Cosine similarity uses `sentence_transformers.util.pytorch_cos_sim`, exactly
. Uses exact cosine similarity — no FAISS or approximation.

NO LLaMA generation today — output CSVs have an empty `generated_response`
column to be filled in tomorrow's generation step.

Usage:
    python3 retrieval.py --pipeline QB1 [--gpu 0] [--thresholds 0.1,0.2,...]
"""
import os
import sys
import time
import json
import argparse
import gc

# CUDA selection MUST happen before importing torch
_p = argparse.ArgumentParser(add_help=False)
_p.add_argument("--gpu", type=int, default=0)
_a, _ = _p.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(_a.gpu)

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import kb


HYDE_SYSTEM = (
    "You are an expert programmer. Given a programming question, "
    "generate a hypothetical answer that would be helpful. "
    "Be specific and technical."
)


# ---------------------------------------------------------------------------
# HyDE caching
# ---------------------------------------------------------------------------
def load_hyde_cache():
    p = config.HYDE_CACHE_PATH
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def save_hyde_cache(cache):
    p = config.HYDE_CACHE_PATH
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(cache, f, indent=2)


def generate_hyde(client, question, retries=3):
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=config.GPT4O_MODEL,
                messages=[
                    {"role": "system", "content": HYDE_SYSTEM},
                    {"role": "user", "content": question},
                ],
                temperature=0.7,
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  [HyDE ERROR] {e}")
                return ""


def ensure_hyde_for_questions(questions):
    """Make sure every question in `questions` has a cached HyDE answer.
    Returns the cache dict.
    """
    cache = load_hyde_cache()
    missing = [q for q in questions if q not in cache]
    if missing:
        print(f"[HyDE] Generating {len(missing)} new HyDE answers (cached: {len(cache)}) ...")
        api_key = config.get_openrouter_api_key()
        client = OpenAI(base_url=config.OPENROUTER_BASE_URL, api_key=api_key)
        for q in tqdm(missing, desc="HyDE"):
            cache[q] = generate_hyde(client, q)
            time.sleep(0.05)
        save_hyde_cache(cache)
        print(f"[HyDE] Saved cache with {len(cache)} entries")
    return cache


# ---------------------------------------------------------------------------
# Per-pipeline retrieval functions. Each returns:
#   list of (score, post_idx, sub_idx, text)
# where:
#   - post_idx is the OVO_data post index (where applicable)
#   - sub_idx is a per-post index for sentences, otherwise = post_idx
# ---------------------------------------------------------------------------

def retrieve_qb1(query_emb, threshold, *, sentence_pool, sent_to_post=None):
    """Direct + Sentence: score query vs ALL sentences, threshold, top_k.

    sent_to_post (numpy int32) maps sentence index -> OVO post_idx; if None,
    post_idx defaults to -1.
    """
    sent_text, sent_emb = sentence_pool
    scores = kb.cos_sim(query_emb, sent_emb).squeeze(0).cpu()
    valid = (scores >= threshold).nonzero(as_tuple=True)[0]
    if len(valid) == 0:
        return []
    valid_scores = scores[valid]
    sorted_pos = torch.argsort(valid_scores, descending=True)[:config.TOP_K]
    chosen = valid[sorted_pos].tolist()
    if sent_to_post is not None:
        return [(scores[i].item(), int(sent_to_post[i]), int(i), sent_text[i]) for i in chosen]
    return [(scores[i].item(), -1, int(i), sent_text[i]) for i in chosen]


def retrieve_qb2(query_emb, threshold, *, fa_pool):
    """Direct + Full answer: score query vs ALL full answers, threshold, top_k."""
    fa_text, fa_emb = fa_pool
    scores = kb.cos_sim(query_emb, fa_emb).squeeze(0).cpu()
    valid = (scores >= threshold).nonzero(as_tuple=True)[0]
    if len(valid) == 0:
        return []
    valid_scores = scores[valid]
    sorted_pos = torch.argsort(valid_scores, descending=True)[:config.TOP_K]
    chosen = valid[sorted_pos].tolist()
    return [(scores[i].item(), int(i), int(i), fa_text[i]) for i in chosen]


def retrieve_qb3(query_emb, threshold, *, q_matrix, ans_sents):
    """Indirect + Sentence (matches old step2-1.ipynb / step2-1-2-v2-.ipynb):
    Stage-1: top-10 question matches; Stage-2: encode their answer sentences,
    threshold on query↔sentence, top-10.
    """
    # Stage 1
    q_scores = kb.cos_sim(query_emb, q_matrix).squeeze(0).cpu()
    top_post_indices = torch.topk(q_scores, k=config.TOP_K).indices.tolist()

    # Stage 2: collect candidate sentences
    candidates = []  # list of (post_idx, sub_idx, sent_text)
    for pi in top_post_indices:
        for si, s in enumerate(ans_sents[pi]):
            if isinstance(s, str) and s.strip():
                candidates.append((pi, si, s))
    if not candidates:
        return []

    # Encode all candidate sentences in one batch
    sent_texts = [c[2] for c in candidates]
    sent_embs = kb.encode_query(sent_texts)
    a_scores = kb.cos_sim(query_emb, sent_embs).squeeze(0).cpu()

    valid = (a_scores >= threshold).nonzero(as_tuple=True)[0]
    if len(valid) == 0:
        return []
    valid_scores = a_scores[valid]
    sorted_pos = torch.argsort(valid_scores, descending=True)[:config.TOP_K]
    chosen = valid[sorted_pos].tolist()
    return [(a_scores[i].item(), candidates[i][0], candidates[i][1], candidates[i][2]) for i in chosen]


def retrieve_qb4(query_emb, threshold, *, q_matrix, fa_pool):
    """Indirect + Full answer (matches old step2-2-v2.ipynb):
    Stage-1: top-10 question matches; Stage-2: their full answers, threshold
    on query↔full_answer, top-10.
    """
    fa_text, fa_emb = fa_pool
    # Stage 1
    q_scores = kb.cos_sim(query_emb, q_matrix).squeeze(0).cpu()
    top_post_indices = torch.topk(q_scores, k=config.TOP_K).indices.tolist()

    # Stage 2: full-answer cosine for those posts
    cand_emb = fa_emb[top_post_indices]   # (top_k, 768)
    a_scores = kb.cos_sim(query_emb, cand_emb).squeeze(0).cpu()

    valid = (a_scores >= threshold).nonzero(as_tuple=True)[0]
    if len(valid) == 0:
        return []
    valid_scores = a_scores[valid]
    sorted_pos = torch.argsort(valid_scores, descending=True)[:config.TOP_K]
    chosen = valid[sorted_pos].tolist()
    return [
        (a_scores[i].item(), top_post_indices[i], top_post_indices[i], fa_text[top_post_indices[i]])
        for i in chosen
    ]


def retrieve_hb1(query_emb, threshold, *, fa_pool):
    """HyDE + Direct + Full answer == retrieve_qb2 with HyDE-encoded query."""
    return retrieve_qb2(query_emb, threshold, fa_pool=fa_pool)


def retrieve_hb2(query_emb, threshold, *, sentence_pool, sent_to_post=None):
    """HyDE + Direct + Sentence == retrieve_qb1 with HyDE-encoded query."""
    return retrieve_qb1(query_emb, threshold, sentence_pool=sentence_pool, sent_to_post=sent_to_post)


def retrieve_hyb(question_emb, hyde_emb, threshold, *, q_matrix, fa_pool):
    """Hybrid (BUG-FIXED):
      Stage-1: search question pool with QUESTION embedding -> top-10 posts.
      Stage-2: re-rank those posts' full answers with HyDE embedding,
               threshold on hyde↔full_answer cosine.
    """
    fa_text, fa_emb = fa_pool
    # Stage 1 — question-pool search with QUESTION embedding
    q_scores = kb.cos_sim(question_emb, q_matrix).squeeze(0).cpu()
    top_post_indices = torch.topk(q_scores, k=config.TOP_K).indices.tolist()

    # Stage 2 — re-rank with HyDE embedding against those posts' full answers
    cand_emb = fa_emb[top_post_indices]
    a_scores = kb.cos_sim(hyde_emb, cand_emb).squeeze(0).cpu()

    valid = (a_scores >= threshold).nonzero(as_tuple=True)[0]
    if len(valid) == 0:
        return []
    valid_scores = a_scores[valid]
    sorted_pos = torch.argsort(valid_scores, descending=True)[:config.TOP_K]
    chosen = valid[sorted_pos].tolist()
    return [
        (a_scores[i].item(), top_post_indices[i], top_post_indices[i], fa_text[top_post_indices[i]])
        for i in chosen
    ]


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def format_results(items):
    """items: list of (score, post_idx, sub_idx, text)"""
    if not items:
        return "", "", "", 0
    context = "\n".join(f"{i+1}. {it[3]}" for i, it in enumerate(items))
    scores = ",".join(f"{it[0]:.4f}" for it in items)
    pids = ",".join(str(it[1]) for it in items)
    return context, scores, pids, len(items)


def run(pipeline_name, thresholds=None, input_path=None):
    if pipeline_name not in config.PIPELINES:
        raise ValueError(f"Unknown pipeline: {pipeline_name}")
    cfg = config.PIPELINES[pipeline_name]
    thresholds = thresholds or config.THRESHOLDS
    input_path = input_path or config.TEST_SET_PATH

    # Load test set
    df = pd.read_csv(input_path)
    print(f"[{pipeline_name}] Loaded {len(df)} test queries from {input_path}")

    # Load only the resources this pipeline needs
    needs_q_matrix = cfg["retrieval"] == "indirect"
    needs_sentence_pool = (cfg["retrieval"] == "direct" and cfg["granularity"] == "sentence")
    needs_fa_pool = (
        (cfg["retrieval"] == "direct" and cfg["granularity"] == "full_answer")
        or (cfg["retrieval"] == "indirect" and cfg["granularity"] == "full_answer")
    )
    needs_ans_sents = (cfg["retrieval"] == "indirect" and cfg["granularity"] == "sentence")

    q_matrix = kb.question_embeddings_matrix() if needs_q_matrix else None
    sentence_pool = kb.sentence_pool() if needs_sentence_pool else None
    fa_pool = kb.full_answer_pool() if needs_fa_pool else None
    ans_sents = kb.answer_sentences_list() if needs_ans_sents else None
    sent_to_post = kb.sent_to_post_map() if needs_sentence_pool else None

    # Move pools to GPU once for fast cos_sim
    enc = kb.get_encoder()
    device = enc.device
    if q_matrix is not None:
        q_matrix = q_matrix.to(device)
    if sentence_pool is not None:
        sent_text, sent_emb = sentence_pool
        sentence_pool = (sent_text, sent_emb.to(device))
    if fa_pool is not None:
        fa_text, fa_emb = fa_pool
        fa_pool = (fa_text, fa_emb.to(device))

    # HyDE caching for HyDE pipelines
    hyde_cache = None
    if cfg["query"] in ("hyde", "question_then_hyde"):
        questions = df["Paraphrased Question"].dropna().tolist()
        hyde_cache = ensure_hyde_for_questions(questions)

    # Output dir
    os.makedirs(config.RETRIEVAL_DIR, exist_ok=True)

    for t in thresholds:
        out_path = os.path.join(config.RETRIEVAL_DIR, f"{pipeline_name}_{t:.1f}.csv")
        if os.path.exists(out_path):
            print(f"[{pipeline_name}@{t}] [SKIP] {out_path} exists")
            continue

        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{pipeline_name}@{t}"):
            question = row["Paraphrased Question"]
            if not isinstance(question, str) or not question.strip():
                rows.append({
                    "Question": row.get("Question", ""),
                    "Accepted Answer": row.get("Accepted Answer", ""),
                    "Paraphrased Question": question,
                    "post_idx": row.get("post_idx", -1),
                    "hypothetical_answer": "",
                    "retrieved_post_ids": "",
                    "retrieved_context": "",
                    "retrieval_scores": "",
                    "num_retrieved": 0,
                    "generated_response": "",
                })
                continue

            hyde_ans = ""
            if cfg["query"] == "question":
                q_emb = kb.encode_query(question)
                items = _dispatch(pipeline_name, q_emb=q_emb, hyde_emb=None,
                                  threshold=t, q_matrix=q_matrix,
                                  sentence_pool=sentence_pool, fa_pool=fa_pool,
                                  ans_sents=ans_sents, sent_to_post=sent_to_post)
            elif cfg["query"] == "hyde":
                hyde_ans = hyde_cache.get(question, "")
                if not hyde_ans:
                    items = []
                else:
                    h_emb = kb.encode_query(hyde_ans)
                    items = _dispatch(pipeline_name, q_emb=h_emb, hyde_emb=h_emb,
                                      threshold=t, q_matrix=q_matrix,
                                      sentence_pool=sentence_pool, fa_pool=fa_pool,
                                      ans_sents=ans_sents, sent_to_post=sent_to_post)
            elif cfg["query"] == "question_then_hyde":  # HYB
                hyde_ans = hyde_cache.get(question, "")
                if not hyde_ans:
                    items = []
                else:
                    q_emb = kb.encode_query(question)
                    h_emb = kb.encode_query(hyde_ans)
                    items = retrieve_hyb(q_emb, h_emb, t, q_matrix=q_matrix, fa_pool=fa_pool)
            else:
                raise ValueError(f"Bad query type: {cfg['query']}")

            ctx, scores, pids, n = format_results(items)
            rows.append({
                "Question": row.get("Question", ""),
                "Accepted Answer": row.get("Accepted Answer", ""),
                "Paraphrased Question": question,
                "post_idx": row.get("post_idx", -1),
                "hypothetical_answer": hyde_ans,
                "retrieved_post_ids": pids,
                "retrieved_context": ctx,
                "retrieval_scores": scores,
                "num_retrieved": n,
                "generated_response": "",   # to be filled tomorrow
            })

        out_df = pd.DataFrame(rows)
        out_df.to_csv(out_path, index=False)
        print(f"[{pipeline_name}@{t}] Saved {out_path}")


def _dispatch(pname, *, q_emb, hyde_emb, threshold, q_matrix, sentence_pool, fa_pool, ans_sents, sent_to_post=None):
    if pname == "QB1":
        return retrieve_qb1(q_emb, threshold, sentence_pool=sentence_pool, sent_to_post=sent_to_post)
    if pname == "QB2":
        return retrieve_qb2(q_emb, threshold, fa_pool=fa_pool)
    if pname == "QB3":
        return retrieve_qb3(q_emb, threshold, q_matrix=q_matrix, ans_sents=ans_sents)
    if pname == "QB4":
        return retrieve_qb4(q_emb, threshold, q_matrix=q_matrix, fa_pool=fa_pool)
    if pname == "HB1":
        return retrieve_hb1(hyde_emb, threshold, fa_pool=fa_pool)
    if pname == "HB2":
        return retrieve_hb2(hyde_emb, threshold, sentence_pool=sentence_pool, sent_to_post=sent_to_post)
    raise ValueError(f"Unknown pipeline {pname}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--pipeline", type=str, required=True,
                        help="One of: QB1 QB2 QB3 QB4 HB1 HB2 HYB")
    parser.add_argument("--thresholds", type=str, default=None,
                        help="Comma-separated, e.g. '0.1,0.2,0.5'")
    parser.add_argument("--input", type=str, default=None)
    args = parser.parse_args()

    thresholds = None
    if args.thresholds:
        thresholds = [float(t) for t in args.thresholds.split(",")]

    run(args.pipeline, thresholds=thresholds, input_path=args.input)


if __name__ == "__main__":
    main()
