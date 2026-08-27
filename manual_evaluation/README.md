# Manual evaluation

Four annotator spreadsheets and their agreement scripts, plus the LLM-as-a-Judge
consistency study.

All sample sizes target 90% confidence with a 10% margin of error. Two
independent annotators (both PhD students in CS with > 10 years of Java + Python
experience) labelled every row and discussed disagreements to a consensus label.

## Files

| File                              | Population        | Sample | Study                                             |
|-----------------------------------|-------------------|--------|---------------------------------------------------|
| `01_usefulness_pipelines.xlsx`    | Synthetic n=666 (Java/Python-only after filtering) | 62 × 11 pipelines | Retrieval-usefulness across 7 internal pipelines + 4 ML-venue baselines |
| `02_usefulness_thresholds.xlsx`   | Synthetic n=666 (Java/Python-only) | 62 × HB1 × 9 thresholds | HB1 threshold sweep (0.1 → 0.9) |
| `03_final_answer_usefulness.xlsx` | Unseen SO 2025 (n=3,376) | 68 × 6 LLMs | Developer helpfulness of final HB1 answers |
| `04_hallucination.xlsx`        | Unseen SO 2025 (n=3,376) | 68 | LLM-vs-human alignment for hallucination judge |
| `annotator_agreement_summary.csv` | — | — | Cohen's κ and raw agreement per study |
| `compute_gwet_ac1_all.py`         | — | — | Recompute Gwet's AC1 across studies |
| `compute_gwet_ac1_threshold.py`   | — | — | Threshold-specific agreement |

## Inter-annotator agreement

| Study                              | Cohen's κ | Raw agreement |
|------------------------------------|-----------|---------------|
| 01 — usefulness across pipelines   | 0.462     | 78.6%         |
| 02 — HB1 threshold sweep           | 0.339     | 80.6%         |
| 03 — final-answer helpfulness      | 0.345     | 70.6%         |
| 04 — hallucination-v2 alignment    | 0.465     | 91.2%         |

## Judge-consistency sub-study

`judge_consistency/` re-runs GPT-4o's judgments through three additional
code-specialized LLM judges:

- DeepSeek-Coder-33B
- Qwen2.5-Coder-32B
- StarCoder2-15B

Aligned on the 62-answer manual sample from RQ1 (see the same-name xlsx in that
folder). GPT-4o remains the reported judge because it has the highest agreement
with the human consensus label (κ = 0.59).
