# Never Come Up Empty: Adaptive Retrieval for Improving LLM Developer Support

Replication package for the paper. It reproduces all four research questions on a
Stack Overflow knowledge base of **3,428,217** Java/Python posts with accepted
answers (created before 2025-01-01) across **six** LLMs.

## Overview

We study Retrieval-Augmented Generation (RAG) for answering developer questions.
We evaluate **11 retrieval methods** — 7 internal pipelines (`QB1`–`QB4`, `HB1`,
`HB2`, `HYB`) and 4 baselines (**BM25, RAG-Fusion, Self-RAG, Adaptive-RAG**) — and
propose **Adaptive HyDE** (pipeline **`HB1`**): the LLM writes a hypothetical
answer with GPT-4o, we embed it with `all-mpnet-base-v2`, retrieve the top-10
accepted answers whose cosine similarity clears an **adaptive threshold**
(start 0.9, decrement 0.1, floor 0.1), and feed them to the final LLM. If nothing
clears the floor, the LLM falls back to zero-shot generation.

Six answer LLMs: **LLaMa-3.1-8B, Mistral-7B, Granite-3.1-8B, Qwen3-8B**
(small, open-source, run locally via Ollama) and **DeepSeek-r1-70B, GPT-4.1**
(frontier, via OpenRouter). GPT-4o generates the HyDE answers and serves as the
LLM-as-a-Judge.

## Workflow



## Repository layout and mapping to the paper's RQs

The folders are organized by **experiment stage**. Because the paper's RQ2 is
retrieval usefulness (a manual study) and RQ3 is generalization (both 2025 sets),
the folder numbers do **not** line up one-to-one with the paper's RQ numbers:

| Folder | Contents | Paper RQ |
|--------|----------|----------|
| `RQ1/` | Pipeline selection on the Synthetic Set (n=666): 11 methods × 6 LLMs, answer quality | **RQ1** |
| `RQ1/manual_retrieval_eval/`, `manual_evaluation/` | Retrieval-usefulness annotation (62 queries × 11 methods) + threshold sweep | **RQ2** |
| `RQ2/` | Adaptive HB1 on Unseen Stack Overflow 2025 (n=3,376): answer quality + hallucination | **RQ3** |
| `RQ3/` | Cross-platform generalization on GitHub Discussions (n=1,732) | **RQ3** |
| `RQ4/` | Deployment cost and latency | **RQ4** |

```
replication_package_v2/
├── RQ1/                     Pipeline selection, Synthetic Set (n=666)  [paper RQ1]
│   ├── scripts/             retrieval / generation / judge / evaluation / hyde_robustness
│   ├── baselines/           BM25, RAG-Fusion, Self-RAG, Adaptive-RAG wrappers
│   ├── data/                synthetic_questions_n666.csv
│   └── manual_retrieval_eval/  retrieval-usefulness study (62 queries × 11 methods)  [paper RQ2]
├── RQ2/                     Adaptive HB1 on Unseen SO 2025 (n=3,376)   [paper RQ3]
│   ├── scripts/             retrieval_adaptive / generation / judge / hallucination / build_testset_*
│   └── data/                unseen_2025.csv
├── RQ3/                     GitHub Discussions generalization (n=1,732) [paper RQ3]
│   └── data/                github_discussions_2025_balanced.{csv,json}   (reuses RQ2/scripts/)
├── RQ4/                     Cost + latency  [paper RQ4]
│   └── scripts/             parse_log_timings.py, benchmark_retrieval_step.py, build_cost_table.py
├── manual_evaluation/       4 annotator spreadsheets + agreement + judge-consistency  [paper RQ2/RQ3]
├── datasets/                Stack Overflow crawler + offline embedding scripts
├── LICENSE
└── README.md
```

Each folder has its own `README.md` with per-step commands.

## Datasets and data availability

Small artifacts (test-set CSVs, aggregated results, manual-eval spreadsheets,
figures) are included. The full KB and embeddings are too large to ship:

| Artifact | Size | How to get it |
|----------|------|---------------|
| Stack Overflow raw dump (Jan 2008 – Dec 2024) | ~90 GB | https://archive.org/details/stackexchange |
| `stack_overflow_python_java_kb.json` | 5.3 GB | Regenerate via `datasets/Embedding_Offline_code.py` |
| Answer + sentence embeddings (`.pt`) | 24 GB | Regenerate via `datasets/Embedding_Offline_code.py` |
| Synthetic Question Set (n=666) | shipped | `RQ1/data/synthetic_questions_n666.csv` |
| Unseen Stack Overflow 2025 (n=3,376) | shipped | `RQ2/data/unseen_2025.csv` |
| GitHub Discussions 2025 (n=1,732) | shipped | `RQ3/data/github_discussions_2025_balanced.csv` |

Rebuilding the KB and embeddings on an A100 80GB takes ~14 hours.

## Requirements

- Python 3.10+; NVIDIA A100 80GB as reported (24 GB VRAM suffices for the small open-source generators only).
- `transformers`, `sentence-transformers`, `torch`, `huggingface_hub`, `pandas`, `numpy`, `nltk`, `scipy`, `tqdm`, `openai`, `openpyxl`.
- **Ollama** for the four small open-source LLMs.
- **OpenRouter** (or OpenAI) API key: GPT-4o (HyDE + judge) and the two frontier answer LLMs (GPT-4.1, DeepSeek-r1-70B). Set `OPENROUTER_KEY_FILE` to a file holding your key.

## Reproducing each research question

| Paper RQ | Where | Entry points | Headline result |
|----------|-------|--------------|-----------------|
| **RQ1** — highest-quality pipeline | `RQ1/scripts/` | `retrieval.py` → `generation.py` → `judge.py` → `evaluation.py` | HB1 has the highest mean answer quality (7.50) across the 6 LLMs |
| **RQ2** — retrieval usefulness | `RQ1/manual_retrieval_eval/scripts/` | `1_sample_queries.py` … `14_kappa_annotators.py` | HB1 retrieves the most useful context (0.77 vs 0.60 for RAG-Fusion, 0.36 for BM25) |
| **RQ3** — generalization | `RQ2/scripts/` (+ `RQ3/data/`) | `retrieval_adaptive.py` → `generation.py` → `judge.py` → `hallucination.py` | Exceeds the accepted answer; hallucination 29.1% → 24.0% on the 4 small LLMs; identical per-LLM ranking on both platforms (Spearman ρ = 1.00) |
| **RQ4** — cost and latency | `RQ4/scripts/` | `parse_log_timings.py` → `build_cost_table.py` | +0.48 s/query (HyDE), 22 ms retrieval; <½ cent/query API cost |

## Manual evaluation

`manual_evaluation/` holds the annotator spreadsheets (two PhD annotators; each
sample sized for 90% confidence, 10% margin of error):

| File | Size | Study | Inter-annotator κ |
|------|------|-------|-------------------|
| `01_usefulness_pipelines.xlsx` | 62 × 11 methods | Retrieval usefulness | 0.46 |
| `02_usefulness_thresholds.xlsx` | 62 × 9 thresholds | HB1 threshold sweep (0.1→0.9) | 0.34 |
| `03_final_answer_usefulness.xlsx` | 68 × 6 LLMs | Developer helpfulness of HB1 answers | 0.35 |
| `04_hallucination.xlsx` | 68 | LLM-vs-human hallucination-judge alignment | 0.46 |

`annotator_agreement_summary.csv` reports κ per study. `judge_consistency/`
validates GPT-4o as the judge against **four** code-specialized judges
(Qwen2.5-Coder-32B, DeepSeek-Coder-33B, DeepSeek-Coder-v2-16B, StarCoder2-15B):
GPT-4o aligns best with the human consensus, and only Qwen2.5-Coder is
competitive.

## Cite

```
@article{NeverComeUpEmpty2026,
  title   = {Never Come Up Empty: Adaptive Retrieval for Improving LLM Developer Support},
  journal = {IEEE Transactions on Software Engineering},
  year    = {2026},
  note    = {Under review}
}
```
