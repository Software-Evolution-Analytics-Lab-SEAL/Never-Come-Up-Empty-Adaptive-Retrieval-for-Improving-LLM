"""
Build the four manual-evaluation audit files.

File 1 — Usefulness across pipelines (full, no subsample)
   Source: ../data/usefulness_decisions.csv  (5,062 LLM decisions, 11 methods)
   Output: ../manual_evaluation/01_usefulness_pipelines.csv

File 2 — Usefulness across thresholds for HB1 + HYB (full, no subsample)
   Source: ../data/usefulness_threshold_decisions.csv
   Output: ../manual_evaluation/02_usefulness_thresholds.csv

File 3 — Final-answer usefulness for HB1 (sampled n=68; 90% CI, 10% margin)
   Sources: New_Experiments_Code/RQ2/data/generation/unseen_2025/<model>/adaptive.csv
   Models: 4 small LLMs (llama, mistral, qwen3, granite)
   Output: ../manual_evaluation/03_final_answer_usefulness.csv
   (Note: actual LLM-binary-judge labels are filled by the helper script
   `11_judge_final_answer_usefulness.py`; this script just builds the sample.)

File 4 — Hallucination manual verification (sampled n=68 queries × 4 models × 2 pipelines)
   Source: New_Experiments_Code/RQ2/data/evaluation/unseen_2025/<model>/<pipeline>_hallu.csv
   Output: ../manual_evaluation/04_hallucination.csv
   (Run after hallucination_check.py completes; otherwise rows have empty LLM labels.)

Each file uses the same audit-friendly schema:
    visible_first      ... question + content
    primary_label      ... LLM judgment (positioned as the user's initial label)
    primary_reason     ... LLM rationale
    annotator2_label   ... blank (second annotator fills)
    annotator2_notes   ... blank (free-text)
    metadata_columns   ... model/method/pipeline/query_idx etc.
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
OUT = os.path.abspath(os.path.join(HERE, "..", "manual_evaluation"))
GEN_UNSEEN = os.path.join(ROOT, "..", "RQ2", "data", "generation", "unseen_2025")
HALLU_DIR = os.path.join(ROOT, "..", "RQ2", "data", "evaluation", "unseen_2025")

SMALL_MODELS = ["llama-3.1-8b", "mistral-7b", "qwen3-8b", "granite-3.1-8b"]
ALL_MODELS = ["llama-3.1-8b", "mistral-7b", "qwen3-8b", "granite-3.1-8b",
              "deepseek-r1-70b", "gpt-4.1"]
SAMPLE_N_UNSEEN = 68     # 90% CI, 10% margin from N=3376
SAMPLE_N_SYNTH = 62      # 90% CI, 10% margin from N=666
SEED = 42

os.makedirs(OUT, exist_ok=True)


def add_blank_annotator(df, primary_col, reason_col):
    """Insert blank columns annotator2_label and annotator2_notes after primary."""
    df["annotator2_label"] = ""
    df["annotator2_notes"] = ""
    return df


# ------------------------------------------------------------------------------
def _synth_subsample_idxs():
    """Sample size: n=62 (90% CI, 10% margin from population N=666 synthetic).
    Uniform random subsample."""
    jp_path = os.path.join(DATA, "sample_queries.csv")
    if not os.path.exists(jp_path):
        raise FileNotFoundError(f"Missing {jp_path}")
    sample = pd.read_csv(jp_path)
    if len(sample) > SAMPLE_N_SYNTH:
        rng = np.random.default_rng(SEED)
        pool = sample["test_idx"].astype(int).to_numpy()
        pick = rng.choice(pool, size=SAMPLE_N_SYNTH, replace=False)
        keep = set(int(i) for i in pick)
        sample = sample[sample["test_idx"].astype(int).isin(keep)].reset_index(drop=True)
    else:
        keep = set(int(i) for i in sample["test_idx"].astype(int))
    return keep, sample


def build_file1():
    """Pipelines: 11 methods × 62 queries × multiple retrieved units per query."""
    keep_idx, sample = _synth_subsample_idxs()
    qmap = dict(zip(sample["test_idx"].astype(int),
                    sample["Paraphrased Question"].astype(str)))
    df = pd.read_csv(os.path.join(DATA, "usefulness_decisions.csv"))
    df = df[df["test_idx"].astype(int).isin(keep_idx)].copy()
    df["question"] = df["test_idx"].astype(int).map(qmap)
    df = df.rename(columns={"useful": "primary_label", "reason": "primary_reason"})
    df["annotator2_label"] = ""
    df["annotator2_notes"] = ""
    cols = ["question", "method", "unit_id", "unit_text",
            "primary_label", "primary_reason",
            "annotator2_label", "annotator2_notes",
            "test_idx"]
    df[cols].sort_values(["test_idx", "method", "unit_id"]).to_csv(
        os.path.join(OUT, "01_usefulness_pipelines.csv"), index=False)
    print(f"[File 1] Saved 01_usefulness_pipelines.csv  "
          f"rows={len(df)}  queries={len(keep_idx)} (90% CI, 10% margin from N=666)")


# ------------------------------------------------------------------------------
def build_file2():
    """HB1 only across thresholds 0.1 to 0.9, restricted to the 62-query subsample."""
    keep_idx, sample = _synth_subsample_idxs()
    qmap = dict(zip(sample["test_idx"].astype(int),
                    sample["Paraphrased Question"].astype(str)))
    df = pd.read_csv(os.path.join(DATA, "usefulness_threshold_decisions.csv"))
    df = df[df["test_idx"].astype(int).isin(keep_idx) & (df["method"] == "HB1")].copy()
    df["question"] = df["test_idx"].astype(int).map(qmap)
    df = df.rename(columns={"useful": "primary_label", "reason": "primary_reason"})
    df["annotator2_label"] = ""
    df["annotator2_notes"] = ""
    cols = ["question", "method", "threshold", "unit_id", "unit_text",
            "primary_label", "primary_reason",
            "annotator2_label", "annotator2_notes",
            "test_idx", "from_cache"]
    df[cols].sort_values(["test_idx", "method", "threshold", "unit_id"]).to_csv(
        os.path.join(OUT, "02_usefulness_thresholds.csv"), index=False)
    print(f"[File 2] Saved 02_usefulness_thresholds.csv  "
          f"rows={len(df)}  queries={len(keep_idx)} (90% CI, 10% margin from N=666)")


# ------------------------------------------------------------------------------
def build_file3_sample():
    """Sample n=68 queries from unseen_2025, prepare audit rows for each of all
    6 generator LLMs. The primary_label column is filled by the companion
    script 11_judge_final_answer_usefulness.py with a binary LLM judge call.
    If the output file already exists, existing primary_label / primary_reason
    values are preserved (keyed on model + query_idx)."""
    out = os.path.join(OUT, "03_final_answer_usefulness.csv")
    existing = {}
    if os.path.exists(out):
        try:
            prev = pd.read_csv(out)
            for _, r in prev.iterrows():
                key = (str(r.get("model", "")), int(r.get("query_idx", -1)))
                existing[key] = (str(r.get("primary_label", "") or ""),
                                 str(r.get("primary_reason", "") or ""),
                                 str(r.get("annotator2_label", "") or ""),
                                 str(r.get("annotator2_notes", "") or ""))
            print(f"[File 3] Will preserve {len(existing)} existing labelled rows")
        except Exception:
            existing = {}

    rng = np.random.default_rng(SEED)
    base_path = os.path.join(GEN_UNSEEN, ALL_MODELS[0], "adaptive.csv")
    base = pd.read_csv(base_path)
    n_pop = len(base)
    picks = sorted(rng.choice(n_pop, size=SAMPLE_N_UNSEEN, replace=False).tolist())
    print(f"[File 3] Sampled {SAMPLE_N_UNSEEN} queries from unseen_2025 (population={n_pop})")

    rows = []
    for m in ALL_MODELS:
        p = os.path.join(GEN_UNSEEN, m, "adaptive.csv")
        if not os.path.exists(p):
            print(f"  WARN: missing {p}; skipping model {m}")
            continue
        df_m = pd.read_csv(p)
        sub = df_m.iloc[picks].reset_index(drop=True)
        for i in range(len(sub)):
            qid = int(sub["query_idx"].iloc[i])
            prev = existing.get((m, qid), ("", "", "", ""))
            rows.append({
                "question": str(sub["Title"].iloc[i]),
                "model": m,
                "generated_answer": str(sub["generated_response"].iloc[i]),
                "primary_label": prev[0],
                "primary_reason": prev[1],
                "annotator2_label": prev[2],
                "annotator2_notes": prev[3],
                "query_idx": qid,
                "Catalog": str(sub.get("Catalog", pd.Series([""] * len(sub))).iloc[i]),
            })
    pd.DataFrame(rows).to_csv(out, index=False)
    n_filled = sum(1 for r in rows if r["primary_label"] in ("True", "False"))
    print(f"[File 3] Saved {out}  rows={len(rows)} (=68 queries × {len(ALL_MODELS)} models)  "
          f"filled={n_filled}  todo={len(rows)-n_filled}")
    print(f"         Run scripts/11_judge_final_answer_usefulness.py to label remaining rows.")


# ------------------------------------------------------------------------------
def build_file4_sample(force=False):
    """LLM-vs-human alignment sample: n=68 (model, query) pairs sampled from
    the combined HB1 pool of 4 small LLMs × 3,376 queries = 13,504 HB1
    generations. ZS is excluded since alignment is the goal — a single 68-row
    sample is enough to compute Cohen's kappa.

    90% confidence, 10% margin (z=1.645, p=0.5, e=0.10) from N=13,504 gives
    n0 ≈ 68; the finite-population correction shifts this only slightly so we
    keep n=68.
    """
    rng = np.random.default_rng(SEED + 1)
    # Build the combined (model, query_idx) pool from each model's adaptive_hallucination.csv
    pool = []
    for m in SMALL_MODELS:
        p = os.path.join(HALLU_DIR, m, "adaptive_hallucination.csv")
        if not os.path.exists(p):
            print(f"[File 4] Hallu v2 output not found for {m}: {p}")
            if not force:
                return
            continue
        df = pd.read_csv(p)
        for i in range(len(df)):
            pool.append((m, i))
    if not pool:
        print(f"[File 4] No data available."); return
    print(f"[File 4] Population: {len(pool)} (model, query) HB1 pairs across "
          f"{len(SMALL_MODELS)} small LLMs")
    pick_idx = rng.choice(len(pool), size=min(SAMPLE_N_UNSEEN, len(pool)), replace=False)
    picks = [pool[i] for i in sorted(pick_idx.tolist())]

    rows = []
    cache = {}   # model -> loaded dataframe (to avoid reloading)
    for model_name, row_idx in picks:
        if model_name not in cache:
            cache[model_name] = pd.read_csv(os.path.join(HALLU_DIR, model_name, "adaptive_hallucination.csv"))
        df = cache[model_name]
        if row_idx >= len(df):
            continue
        r = df.iloc[row_idx]
        rows.append({
            "question": str(r.get("Title") or r.get("Paraphrased Question") or ""),
            "model": model_name,
            "pipeline": "HB1",
            "generated_answer": str(r.get("generated_response") or ""),
            "accepted_answer": str(r.get("Accepted_Answer") or r.get("Accepted Answer") or ""),
            "primary_label": str(r.get("hallu_status") or ""),
            "primary_hallucination_rate": float(r["hallucination_rate"]) if pd.notna(r.get("hallucination_rate")) else float("nan"),
            "primary_supported": int(r["supported"]) if pd.notna(r.get("supported")) else 0,
            "primary_compatible": int(r["compatible"]) if pd.notna(r.get("compatible")) else 0,
            "primary_contradicts": int(r["contradicts"]) if pd.notna(r.get("contradicts")) else 0,
            "primary_claims_json": str(r.get("claims_json") or "")[:4000],
            "primary_reason": str(r.get("hallu_reason") or ""),
            "annotator2_label": "",
            "annotator2_notes": "",
            "query_idx": int(r["query_idx"]) if pd.notna(r.get("query_idx")) else row_idx,
        })
    out = os.path.join(OUT, "04_hallucination.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    n_per_model = pd.Series([r["model"] for r in rows]).value_counts().to_dict()
    print(f"[File 4] Saved {out}  rows={len(rows)}  per-model: {n_per_model}")


# ------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", default=["1", "2", "3", "4"])
    ap.add_argument("--force", action="store_true", help="proceed for file 4 even if hallu csvs missing")
    args = ap.parse_args()
    if "1" in args.files: build_file1()
    if "2" in args.files: build_file2()
    if "3" in args.files: build_file3_sample()
    if "4" in args.files: build_file4_sample(force=args.force)


if __name__ == "__main__":
    main()
