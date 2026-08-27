# RQ2 — HB1 with adaptive thresholding on Unseen Stack Overflow 2025

RQ2 evaluates HB1 with adaptive thresholding on 3,376 Stack Overflow questions
created in 2025 (after every generator's training cutoff and after the KB
freeze). HB1 retrieves useful context and reduces hallucination rate
significantly for all four small open-source LLMs.

## Scripts (`scripts/`)

`config.py` is fully standalone (no cross-folder imports). Set `KB_ROOT` and
`OPENROUTER_KEY_FILE` via environment variables.

| Script | Purpose |
|--------|---------|
| `config.py`                  | Paths, thresholds, encoder, PIPELINES, per-LLM optimal pipeline |
| `kb.py`                      | KB loaders + encoder (mirror of RQ1 for self-containment) |
| `llm_clients.py`             | Unified OpenRouter / Ollama client + `MODELS` registry |
| `retrieval.py`               | Retrieval primitives (`retrieve_hb1`, `retrieve_hyb`, `generate_hyde`) |
| `build_testset_crawl.py`     | Stack Overflow API crawler for 2025 posts |
| `build_testset_dedup.py`     | Dedup against the KB (post-id + title normalization) |
| `build_testset_assemble.py`  | Assemble the final unseen 2025 test set |
| `hyde_cache.py`              | Cache GPT-4o hypothetical answers per query (idempotent) |
| `retrieval_adaptive.py`      | Adaptive threshold retrieval (0.9 → 0.1, floor 0.1) for HB1 / HYB |
| `generation.py`              | Final-answer generation (`--mode rag \| zeroshot`) with sharding support |
| `merge_shards.py`            | Merge sharded generation outputs |
| `judge.py`                   | GPT-4o LLM-as-a-Judge |
| `hallucination.py`           | 3-class hallucination taxonomy (`check` + `aggregate` subcommands) |

## Data (`data/`)

- `unseen_2025.csv` — 3,376-question Unseen Stack Overflow 2025 test set.

## Reproduce

```bash
export KB_ROOT=~/Adaptive_HyDe_RAG
export OPENROUTER_KEY_FILE=~/openrouter_api.txt
cd scripts

# 1. Precompute HyDE answers (idempotent cache)
python hyde_cache.py --input ../data/unseen_2025.csv --out ../data/hyde_cache.json

# 2. Adaptive retrieval for HB1 (and HYB, for Mistral-7B)
python retrieval_adaptive.py --pipeline HB1 --gpu 0
python retrieval_adaptive.py --pipeline HYB --gpu 0

# 3. Generation per LLM
python generation.py --mode rag       --model llama-3.1-8b
python generation.py --mode zeroshot  --model llama-3.1-8b
# (repeat per model)

# 4. Judge answer quality
python judge.py --mode rag --model llama-3.1-8b --all

# 5. Hallucination v2 taxonomy
python hallucination.py check
python hallucination.py aggregate     # writes hallucination_summary.csv
```

## Key result

Paired Wilcoxon hallucination-rate reduction (HB1 vs ZS on unseen 2025, n=3,376):

| LLM             | HB1   | ZS    | Δ      | p        |
|-----------------|-------|-------|--------|----------|
| LLaMa-3.1-8B    | 0.277 | 0.362 | −0.086 | < 10⁻³  |
| Mistral-7B      | 0.323 | 0.400 | −0.078 | < 10⁻³  |
| Qwen3-8B        | 0.129 | 0.143 | −0.014 | < 10⁻³  |
| Granite-3.1-8B  | 0.231 | 0.258 | −0.027 | < 10⁻³  |

Judge-vs-human alignment sample (68 rows): Cohen's κ = 0.465, 91.2% raw agreement.

## Data-leakage protocol

All Unseen 2025 questions were crawled with `creation_date ≥ 2025-01-01`. The KB
was frozen before 2025-01-01, so no `post_id` overlap is possible. Stack
Overflow duplicates (`closed_reason = "Duplicate"`) are excluded via the
accepted-answer requirement, and all 3,376 titles are unique within the set.
See `../datasets/README.md` for the full protocol.
