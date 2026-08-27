# RQ4 — Deployment cost and latency

RQ4 measures the end-to-end cost and latency of running HB1 at deployment scale
on the Unseen SO 2025 set (3,376 questions × 6 LLMs). The RAG-specific overhead
is dominated by HyDE hypothetical-answer generation (~0.48 s/query, one GPT-4o
call) and cosine-similarity retrieval against the 3.4M-post KB (~22 ms/query
on a single A100). Final-generation latency is dominated by API queue variance.

## Scripts (`scripts/`)

| Script | Purpose |
|--------|---------|
| `parse_log_timings.py`      | Parse per-request timings from generation-run logs |
| `benchmark_retrieval_step.py` | Cold-cache retrieval microbenchmark against the KB |
| `build_cost_table.py`       | Table IV — per-1k-queries $ cost per LLM |

Cost/latency artifacts are regenerated on demand from the run logs (kept
alongside the generation runs in `../RQ2/data/generation/…/logs/`). Raw logs
are not shipped because they contain full prompt/response text.

## Reproduce

```bash
cd scripts

# 1. Extract timings from your RQ2/RQ3 generation logs
python parse_log_timings.py --logs <path/to/generation/logs> --out ../results/timings.csv

# 2. Retrieval microbenchmark (requires ../datasets/ KB + embeddings)
python benchmark_retrieval_step.py --kb <path/to/kb.json> \
                                   --embeddings <path/to/complete_answer_embeddings.pt> \
                                   --queries ../../RQ2/data/unseen_2025.csv \
                                   --out ../results/retrieval_latency.csv

# 3. Cost table
python build_cost_table.py --timings ../results/timings.csv \
                           --out ../results/cost_table.csv
```

## Key numbers

- HyDE generation: 0.48 s / query (one GPT-4o call).
- Retrieval step: 22 ms / query (cosine similarity over 3.4M answer embeddings on A100).
- HB1 vs ZS end-to-end overhead: within 3 s / query for four of six LLMs.
- Per-1k-queries API cost: ~$0.23 (local open-source), ~$1.17 (DeepSeek-r1), ~$4.84 (GPT-4.1).
