"""
Adaptive-threshold retrieval for RQ2 (single-run variant, not a threshold sweep).

For each query in data/unseen_5510.csv:
  - Encode question (+ generate/encode HyDE answer if needed)
  - Try thresholds 0.9, 0.8, ..., 0.1 in order; STOP at the first threshold
    that returns at least one retrieval
  - Record: threshold_hit, retrieved_post_ids, retrieved_context, num_retrieved,
            retrieval_scores, hypothetical_answer

One pass per pipeline (HB1, HYB). These are shared by all 6 LLMs in the
downstream generation step.

Output: data/retrieval/{pipeline}_adaptive.csv

Usage:
    python3 retrieval_adaptive.py --pipeline HB1 --gpu 1
    python3 retrieval_adaptive.py --pipeline HYB --gpu 3
"""
import os
import sys
import time
import json
import argparse

# CUDA selection before torch
_p = argparse.ArgumentParser(add_help=False)
_p.add_argument("--gpu", type=int, default=0)
_a, _ = _p.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(_a.gpu)

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from openai import OpenAI

HERE = os.path.dirname(os.path.abspath(__file__))
# Order: RQ1 (lower priority) then RQ2 (higher priority) so `import config` picks RQ2
sys.path.insert(0, HERE)
import config  # RQ2 config
import kb
from run_pipeline import (
    HYDE_SYSTEM, generate_hyde, retrieve_hb1, retrieve_hyb,
)

THRESHOLDS_HIGH_TO_LOW = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

HYDE_CACHE_PATH = os.path.join(config.DATA_DIR, "hyde_cache_unseen.json")


def load_hyde_cache():
    if os.path.exists(HYDE_CACHE_PATH):
        with open(HYDE_CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_hyde_cache(cache):
    os.makedirs(os.path.dirname(HYDE_CACHE_PATH), exist_ok=True)
    with open(HYDE_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def ensure_hyde(questions):
    cache = load_hyde_cache()
    missing = [q for q in questions if q not in cache]
    if missing:
        print(f"[HyDE] {len(missing)} new answers to generate (cached: {len(cache)})")
        with open(config.OPENROUTER_API_KEY_PATH) as f:
            key = f.read().strip()
        client = OpenAI(base_url=config.OPENROUTER_BASE_URL, api_key=key)
        for i, q in enumerate(tqdm(missing, desc="HyDE")):
            cache[q] = generate_hyde(client, q)
            if (i + 1) % 50 == 0:
                save_hyde_cache(cache)
            time.sleep(0.05)
        save_hyde_cache(cache)
    return cache


def format_items(items):
    if not items:
        return "", "", "", 0
    context = "\n".join(f"{i+1}. {it[3]}" for i, it in enumerate(items))
    scores = ",".join(f"{it[0]:.4f}" for it in items)
    pids = ",".join(str(it[1]) for it in items)
    return context, scores, pids, len(items)


def run(pipeline, input_csv, out_csv):
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} queries from {input_csv}")

    # Resume
    done_idx = set()
    if os.path.exists(out_csv):
        prior = pd.read_csv(out_csv)
        done_idx = set(prior["query_idx"].tolist())
        print(f"Resuming — {len(done_idx)} rows already done")
        rows = prior.to_dict("records")
    else:
        rows = []

    # Pre-gen HyDE for all queries
    questions = df["Title"].astype(str).tolist()
    hyde_cache = ensure_hyde(questions)

    # Load pools on GPU
    enc = kb.get_encoder()
    device = enc.device
    # Pre-reserve GPU memory for the pools to protect against other users eating
    # free RAM while we spend ~20 min loading the OVO pickle on CPU.
    N_POSTS = 3428217
    DIM = 768
    print(f"[RQ2] Pre-reserving GPU buffers for fa_pool + q_matrix ...")
    fa_gpu = torch.empty(N_POSTS, DIM, dtype=torch.float16, device=device)
    q_gpu = None
    if pipeline == "HYB":
        q_gpu = torch.empty(N_POSTS, DIM, dtype=torch.float16, device=device)
    torch.cuda.synchronize()
    print(f"[RQ2] Pre-reserved; free now: "
          f"{torch.cuda.mem_get_info(device)[0] / 1e9:.1f} GB")

    # Load fa_pool and copy into the pre-reserved buffer
    fa_pool = kb.full_answer_pool()
    fa_text, fa_emb = fa_pool
    fa_gpu.copy_(fa_emb.half())
    del fa_emb
    # L2-normalize in place so downstream cos_sim avoids a 5 GB temp alloc per call
    fa_gpu = torch.nn.functional.normalize(fa_gpu, p=2, dim=1, out=fa_gpu)
    fa_pool = (fa_text, fa_gpu)

    q_matrix = None
    if pipeline == "HYB":
        ovo = kb.load_ovo_data()
        print(f"[RQ2] Stacking {len(ovo)} question embeddings on CPU as fp16 ...")
        q_cpu = torch.stack([p["question_embedding"].detach().cpu().half() for p in ovo])
        q_gpu.copy_(q_cpu)
        del q_cpu
        q_gpu = torch.nn.functional.normalize(q_gpu, p=2, dim=1, out=q_gpu)
        q_matrix = q_gpu
        print(f"[RQ2] q_matrix on {device}: shape={tuple(q_matrix.shape)} dtype={q_matrix.dtype}")

    # Monkey-patch cos_sim to skip re-normalizing pre-normalized pool tensors
    # (big temp allocation is what OOM'd us at runtime).
    _orig_cos_sim = kb.cos_sim
    def _fast_cos_sim(a, b):
        a_norm = torch.nn.functional.normalize(a.to(b.dtype), p=2, dim=-1)
        if a_norm.dim() == 1:
            a_norm = a_norm.unsqueeze(0)
        return torch.mm(a_norm, b.transpose(0, 1))
    kb.cos_sim = _fast_cos_sim
    # run_pipeline.py imported utils_encode already, but it references
    # kb.cos_sim at call time, so the patch takes effect.

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"adaptive/{pipeline}"):
        qi = int(row["query_idx"])
        if qi in done_idx:
            continue
        q = str(row["Title"])
        hyde_ans = hyde_cache.get(q, "")
        if not hyde_ans:
            rows.append({**row, "pipeline": pipeline, "hypothetical_answer": "",
                         "threshold_hit": 0.0, "retrieved_post_ids": "",
                         "retrieved_context": "", "retrieval_scores": "",
                         "num_retrieved": 0})
            continue
        hyde_emb = kb.encode_query(hyde_ans)
        q_emb = kb.encode_query(q) if pipeline == "HYB" else None

        # Adaptive sweep
        items = []
        thr_hit = 0.0
        for t in THRESHOLDS_HIGH_TO_LOW:
            if pipeline == "HB1":
                items = retrieve_hb1(hyde_emb, t, fa_pool=fa_pool)
            elif pipeline == "HYB":
                items = retrieve_hyb(q_emb, hyde_emb, t, q_matrix=q_matrix, fa_pool=fa_pool)
            if items:
                thr_hit = t
                break

        ctx, scores, pids, n = format_items(items)
        rows.append({
            "query_idx": qi,
            "Title": q,
            "Accepted_Answer": row.get("Accepted_Answer", ""),
            "Catalog": row.get("Catalog", ""),
            "pipeline": pipeline,
            "hypothetical_answer": hyde_ans,
            "threshold_hit": thr_hit,
            "retrieved_post_ids": pids,
            "retrieved_context": ctx,
            "retrieval_scores": scores,
            "num_retrieved": n,
        })
        if (idx + 1) % 200 == 0:
            pd.DataFrame(rows).to_csv(out_csv, index=False)

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--pipeline", choices=["HB1", "HYB"], required=True)
    ap.add_argument("--input", default=os.path.join(config.DATA_DIR, "unseen_5510.csv"))
    ap.add_argument("--output-dir", default=os.path.join(config.DATA_DIR, "retrieval"))
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, f"{args.pipeline}_adaptive.csv")
    run(args.pipeline, args.input, out_csv)


if __name__ == "__main__":
    main()
