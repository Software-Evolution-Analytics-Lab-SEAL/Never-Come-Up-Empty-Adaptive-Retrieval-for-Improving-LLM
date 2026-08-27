"""
Quick benchmark of the retrieval step in HB1 pipeline.

Per query, retrieval consists of:
  1. Encode the HyDE pseudo-answer with all-mpnet-base-v2 (forward pass on a
     single sentence/short passage).
  2. Cosine similarity against the precomputed full-answer embedding bank
     (~3.4 M vectors).
  3. Top-k selection (k = 10).

Reports the mean time over N=100 queries (using the precomputed HyDE strings
from the synthetic n=666 cache).
"""
import os
import sys
import time
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

KB_PATH = os.environ.get("KB_EMBEDDINGS_PATH", os.path.expanduser("~/Adaptive_HyDe_RAG/complete_answer_embeddings.pt"))
HYDE_CACHE = os.environ.get("HYDE_CACHE_PATH", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "RQ1", "data", "hyde_cache.json")))
N_BENCH = 100
TOP_K = 10
ENCODER = "sentence-transformers/all-mpnet-base-v2"


def main():
    print("Loading KB embeddings ...")
    t0 = time.perf_counter()
    kb = torch.load(KB_PATH, map_location="cpu")
    if isinstance(kb, dict):
        kb = kb.get("embeddings", kb)
    if isinstance(kb, list):
        kb = torch.stack([torch.as_tensor(x) for x in kb])
    if not isinstance(kb, torch.Tensor):
        kb = torch.as_tensor(kb)
    kb = kb.to(dtype=torch.float32)
    kb_norm = kb / kb.norm(dim=1, keepdim=True).clamp_min(1e-9)
    print(f"  KB shape: {tuple(kb.shape)}  load+normalize: {time.perf_counter()-t0:.2f}s")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")
    if device == "cuda":
        kb_norm = kb_norm.to(device)

    print("Loading encoder ...")
    t0 = time.perf_counter()
    encoder = SentenceTransformer(ENCODER, device=device)
    print(f"  encoder load: {time.perf_counter()-t0:.2f}s")

    with open(HYDE_CACHE) as f:
        hyde = json.load(f)
    queries = list(hyde.values())[:N_BENCH]
    print(f"  benchmark queries: {len(queries)}")

    # Warm-up
    _ = encoder.encode(queries[0], convert_to_tensor=True, device=device)

    # Per-query benchmark
    times_encode = []
    times_search = []
    for q in queries:
        t0 = time.perf_counter()
        e = encoder.encode(q, convert_to_tensor=True, device=device, show_progress_bar=False)
        e = e / e.norm().clamp_min(1e-9)
        t1 = time.perf_counter()
        sims = kb_norm @ e
        topk = torch.topk(sims, TOP_K).indices.cpu().numpy()
        t2 = time.perf_counter()
        times_encode.append((t1 - t0) * 1000)
        times_search.append((t2 - t1) * 1000)

    enc = np.array(times_encode)
    srch = np.array(times_search)
    total = enc + srch
    print(f"\n=== Retrieval step benchmark ({N_BENCH} queries, top-{TOP_K}, device={device}) ===")
    print(f"  Encode HyDE pseudo-answer  : mean {enc.mean():>7.2f} ms   median {np.median(enc):>7.2f} ms")
    print(f"  Top-k cosine similarity    : mean {srch.mean():>7.2f} ms   median {np.median(srch):>7.2f} ms")
    print(f"  TOTAL retrieval per query  : mean {total.mean():>7.2f} ms   median {np.median(total):>7.2f} ms")
    print(f"  ({total.mean()/1000:.3f} s per query)")


if __name__ == "__main__":
    main()
