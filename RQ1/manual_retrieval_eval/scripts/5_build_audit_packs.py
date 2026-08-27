"""
Build human audit packs for Tier-2 validation of LLM Usefulness judgments.

Focal methods: HB1 (proposed best), HYB (close runner-up), BM25 (lexical baseline).
Sample: uniform random subset of 30 queries from the 84 paired sample. The same
30 queries are audited on all 3 methods so the human can compare side-by-side.

Each audit pack is a CSV with one row per (query, retrieved unit), with the
LLM's decision and rationale **hidden** in a separate "answer key" file.
The human assigns yes/no based only on (question, unit_text).

Outputs (under ../audit/):
  audit_pack_HB1.csv          (human form: blank `human_useful` column)
  audit_pack_HYB.csv
  audit_pack_BM25.csv
  audit_answer_key.csv        (LLM decisions - DO NOT show to annotator)
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
AUDIT = os.path.abspath(os.path.join(HERE, "..", "audit"))
os.makedirs(AUDIT, exist_ok=True)

FOCAL = ["HB1", "HYB", "BM25"]
N_AUDIT = 30
SEED = 17


def main():
    sample = pd.read_csv(os.path.join(DATA, "sample_queries.csv"))
    decisions = pd.read_csv(os.path.join(DATA, "usefulness_decisions.csv"))

    # Uniform random subsample of N_AUDIT queries
    rng = np.random.default_rng(SEED)
    all_ids = sample["test_idx"].to_numpy()
    picked = np.sort(rng.choice(all_ids, size=N_AUDIT, replace=False))
    print(f"Audit subsample: {len(picked)} queries")

    qmap = dict(zip(sample["test_idx"].astype(int), sample["Paraphrased Question"].astype(str)))

    # Answer key (all LLM decisions)
    key_rows = []
    for method in FOCAL:
        sub = decisions[(decisions["method"] == method) &
                        (decisions["test_idx"].astype(int).isin(picked))].copy()
        sub = sub.sort_values(["test_idx", "unit_id"])

        # Human form: same rows, no LLM decision shown, blank `human_useful` col
        human = sub[["test_idx", "unit_id", "unit_text"]].copy()
        human.insert(1, "question", human["test_idx"].astype(int).map(qmap))
        human["human_useful"] = ""
        human["human_notes"] = ""
        out_path = os.path.join(AUDIT, f"audit_pack_{method}.csv")
        human.to_csv(out_path, index=False)
        print(f"  {method}: {len(human)} rows -> {out_path}")

        # Answer key
        for _, r in sub.iterrows():
            key_rows.append({
                "method": method,
                "test_idx": int(r["test_idx"]),
                "unit_id": int(r["unit_id"]),
                "llm_useful": str(r["useful"]),
                "llm_reason": str(r.get("reason", "")),
            })

    pd.DataFrame(key_rows).to_csv(os.path.join(AUDIT, "audit_answer_key.csv"), index=False)
    print(f"\nAnswer key: audit/audit_answer_key.csv  ({len(key_rows)} rows)")

    # Instructions file
    instr = """\
HUMAN AUDIT INSTRUCTIONS
========================

You are validating LLM-assigned Usefulness labels on retrieved content.

Files:
  audit_pack_HB1.csv    (HB1 retrievals, blank human_useful column)
  audit_pack_HYB.csv    (HYB retrievals)
  audit_pack_BM25.csv   (BM25 retrievals)
  audit_answer_key.csv  (LLM labels — DO NOT open until done)

For each row:
  1. Read the `question` and the `unit_text`.
  2. Decide: does this unit contain information that helps answer the question?
     - Mark `human_useful` = TRUE if yes (any concrete signal toward an answer
       counts; partial usefulness counts; same-topic boilerplate does NOT count).
     - Mark `human_useful` = FALSE if no (off-topic, generic noise, or filler).
  3. Optionally add a one-line note in `human_notes`.
  4. Save the file.

Tips:
  - Stay calibrated: an answer like "Use sorted()" for a sorting question is TRUE.
    "Hope this helps!" is FALSE.
  - Code that does the same thing as what's needed counts as TRUE.
  - Don't peek at the LLM labels before finishing — that biases agreement.

After all 3 packs are filled in, run:
  python3 scripts/6_kappa.py
to compute Cohen's κ between LLM and human.
"""
    with open(os.path.join(AUDIT, "INSTRUCTIONS.txt"), "w") as f:
        f.write(instr)
    print("Audit instructions written to audit/INSTRUCTIONS.txt")


if __name__ == "__main__":
    main()
