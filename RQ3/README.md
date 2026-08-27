# RQ3 — Cross-platform generalization on GitHub Discussions

RQ3 tests whether the HB1 findings on Stack Overflow generalize to a second,
independent developer Q&A platform. We crawl 1,732 GitHub Discussions threads
(866 Java + 866 Python) from the top-100 Java and top-100 Python public
repositories by stars, all created in 2025 with a marked answer of at least
100 characters.

The retrieval, generation, and evaluation code is identical to RQ2; only the
input CSV changes.

## Contents

| Path | Purpose |
|------|---------|
| `data/github_discussions_2025_balanced.csv`  | 1,732-thread balanced test set |
| `data/github_discussions_2025_balanced.json` | Same set with full discussion metadata |
| `data/github_disc_2025.csv`                  | Balanced GitHub Discussions set (1,732 threads) |

RQ3 has no scripts of its own — it reuses `../RQ2/scripts/` verbatim, with the
GitHub Discussions CSV as input.

## Reproduce

RQ3 reuses `../RQ2/scripts/`:

```bash
export KB_ROOT=~/Adaptive_HyDe_RAG
export OPENROUTER_KEY_FILE=~/openrouter_api.txt
cd ../RQ2/scripts

python hyde_cache.py --input ../../RQ3/data/github_discussions_2025_balanced.csv \
                     --out ../../RQ3/data/hyde_cache.json

python retrieval_adaptive.py --input ../../RQ3/data/github_discussions_2025_balanced.csv \
                             --hyde-cache ../../RQ3/data/hyde_cache.json \
                             --out ../../RQ3/data/retrieval

python generation.py --mode rag --model <LLM>     # loop over 6 LLMs
python judge.py     --mode rag --all
```

## Key result

Per-LLM ranking on GitHub Discussions matches the ranking on Unseen SO 2025
exactly (Spearman ρ = 1.00). Coverage: 77% of GitHub Discussions queries retain
at least one candidate above threshold 0.6 (vs 96% for Unseen SO 2025).

## Crawl filters

1. Marked answer ≥ 100 characters (excludes one-liners like "fixed in PR #123").
2. Thread title or body contains a question signal (*what*, *how*, *why*, or `?`).
3. Answer is not solely a URL (link-only answers add no standalone content).
4. Duplicate titles removed, majority class undersampled to balance Java and Python.
