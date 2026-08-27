# RQ1 ML-venue baselines

Four baselines compared against HB1 in Table II:

| Baseline     | Venue         | Retriever       |
|--------------|---------------|-----------------|
| BM25         | classical     | rank_bm25       |
| RAG-Fusion   | multi-query   | RRF over `all-mpnet-base-v2` |
| Self-RAG     | ICLR 2024     | Contriever-MSMARCO (official) |
| Adaptive-RAG | NAACL 2024    | BM25 + T5-Large router (official) |

## Setup

- Embedder: `all-mpnet-base-v2`
- Knowledge base: 3.4M Java/Python Stack Overflow posts (see `../../datasets/`)
- Test set: Synthetic Question Set (n=666)
- Generator: all six evaluated LLMs
- Similarity threshold: 0.7 for retrieval-thresholded baselines

BM25 and RAG-Fusion are implemented directly in `../scripts/` (`baseline_bm25.py`,
`baseline_rag_fusion.py`). Self-RAG and Adaptive-RAG use the official upstream
repos — clone them into `self_rag/upstream/` and `adaptive_rag/upstream/`, then
run the wrapper scripts.

## Aggregated results (used to build Table II)

- `table_adaptive_means.csv` — Avg(adaptive) row per model × pipeline
- `table_hb1_vs_all_stats.csv` — HB1 vs every rival, Wilcoxon + Cliff's δ
