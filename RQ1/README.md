# RQ1 — Optimal RAG pipeline on the Synthetic Question Set

HB1 (HyDE-based direct retrieval over full accepted answers) has the highest
mean answer quality across six LLMs on 666 synthetic questions sampled and
paraphrased from the Stack Overflow knowledge base. HB1 significantly
outperforms all four ML-venue baselines and improves over zero-shot for
five of six generators.

## Scripts (`scripts/`)

Set `KB_ROOT` to your local knowledge-base build and `OPENROUTER_KEY_FILE` to
your API key file. Defaults live in `config.py`.

| Script | Purpose |
|--------|---------|
| `config.py`              | Paths, thresholds, encoder, LLM IDs, PIPELINES table |
| `kb.py`                  | Knowledge-base loaders + encoder wrapper |
| `build_synthetic_set.py` | Build the 666-question synthetic set (99% CI, 5% margin) — sample and paraphrase |
| `retrieval.py`           | Run any of the 7 internal pipelines (QB1–QB4, HB1, HB2, HYB) at a fixed threshold |
| `baseline_bm25.py`       | BM25 baseline |
| `baseline_rag_fusion.py` | RAG-Fusion baseline |
| `generation.py`          | Final-answer generation (`--mode rag \| zeroshot`) |
| `judge.py`               | GPT-4o LLM-as-a-Judge (`--mode rag \| zeroshot`) |
| `evaluation.py`          | Table II builder + HB1-vs-baselines significance (`significance \| table-ii`) |
| `hyde_robustness.py`     | HyDE-generator robustness check (Qwen3-8B vs GPT-4o) |

## Baselines (`baselines/`)

BM25 and RAG-Fusion are implemented directly in `scripts/`. Self-RAG and
Adaptive-RAG use the official upstream repos — clone them per
`baselines/adaptive_rag/UPSTREAM.md` and `baselines/self_rag/UPSTREAM.md`, then
follow `baselines/README.md` for the setup and
training / prediction commands.

Aggregated CSVs used to build Table II are shipped alongside:
`table_hb1_vs_all_stats.csv`.

## Data (`data/`)

- `synthetic_questions_n666.csv` — the 666-question synthetic test set.

## Manual retrieval-usefulness sub-study (`manual_retrieval_eval/`)

Numbered scripts run in order (`1_…` → `14_…`) to sample queries, LLM-precheck
retrieval, aggregate usefulness, correlate with judge scores, build annotator
audit spreadsheets, compute Cohen's κ, run the HB1 threshold sweep (0.1 → 0.9),
and export final xlsx / kappa tables. Data snapshots and audit CSVs live in
`manual_retrieval_eval/data/`.

## Reproduce

```bash
export KB_ROOT=~/Adaptive_HyDe_RAG
export OPENROUTER_KEY_FILE=~/openrouter_api.txt
cd scripts

# 1. Retrieval — sweep all 7 internal pipelines × 9 thresholds
python retrieval.py --pipeline HB1
# (repeat for QB1, QB2, QB3, QB4, HB2, HYB)
python baseline_bm25.py
python baseline_rag_fusion.py

# 2. Generation per LLM (rag mode reads retrieval CSVs; zeroshot reads test set)
python generation.py --mode rag       --model llama-3.1-8b --all
python generation.py --mode zeroshot  --model llama-3.1-8b
# (repeat for each of the 6 LLMs)

# 3. Judge answer quality
python judge.py --mode rag       --model llama-3.1-8b --all
python judge.py --mode zeroshot  --all

# 4. Table II + significance
python evaluation.py table-ii
python evaluation.py significance
```

## Key results

HB1 mean judge score across 6 generators = **7.50** (highest of all 12
pipelines and baselines). Best on 5 of 6 generators; statistically tied with
HYB for Mistral-7B (Wilcoxon p = 0.86).

Manual retrieval-usefulness (62 Java/Python queries × 11 methods, consensus labels):

| Method       | Usefulness |
|--------------|------------|
| HB1          | 0.768      |
| HYB          | 0.765      |
| QB4          | 0.753      |
| RAG-Fusion   | 0.604      |
| Self-RAG     | 0.498      |
| Adaptive-RAG | 0.378      |
| BM25         | 0.362      |

Inter-annotator Cohen's κ = 0.462 (moderate, 78.6% raw agreement).
