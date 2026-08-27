"""
Merge existing zero-shot generations into Path-A rows of each model's
ADAPTIVERAG generation CSV.

Adaptive-RAG's classifier sends 151 of the 666 SE questions down Path A
(no retrieval). Those rows have empty `generated_response` after running
generate_responses.py (which skips rows with num_retrieved == 0). Path A's
intended behavior is zero-shot generation, and we already have zero-shot
outputs at data/baseline_zeroshot/<model>.csv for all 6 models.

This script copies the zero-shot `generated_response` into the Path-A rows
of each data/generation/<model>/ADAPTIVERAG.csv, keyed on
(post_idx, Paraphrased Question).

Idempotent — safe to re-run after each model's generation completes.

Usage:
    python3 merge_zeroshot_path_a.py            # all 6 models
    python3 merge_zeroshot_path_a.py --model llama-3.1-8b   # one model
"""
import os
import sys
import argparse
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RQ1_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "RQ1_n666"))
sys.path.insert(0, RQ1_DIR)
import config

ZEROSHOT_DIR = os.path.join(config.DATA_DIR, "baseline_zeroshot")
GEN_DIR = os.path.join(config.DATA_DIR, "generation")
MODELS = ["llama-3.1-8b", "mistral-7b", "granite-3.1-8b",
          "qwen3-8b", "deepseek-r1-70b", "gpt-4.1"]


def merge_for_model(model):
    gen_path = os.path.join(GEN_DIR, model, "ADAPTIVERAG.csv")
    zs_path = os.path.join(ZEROSHOT_DIR, f"{model}.csv")
    if not os.path.exists(gen_path):
        print(f"[{model}] generation CSV not found: {gen_path} — skipping (still running?)")
        return False
    if not os.path.exists(zs_path):
        print(f"[{model}] zero-shot CSV not found: {zs_path} — skipping")
        return False

    g = pd.read_csv(gen_path)
    z = pd.read_csv(zs_path)
    z_lookup = {(int(r["post_idx"]), str(r["Paraphrased Question"])): str(r["generated_response"])
                for _, r in z.iterrows()}

    # Identify Path-A rows that still need generation
    a_mask = (g.get("complexity_label", pd.Series(dtype=object)) == "A")
    empty_mask = g["generated_response"].fillna("").astype(str).str.strip() == ""
    target_mask = a_mask & empty_mask
    n_target = target_mask.sum()
    if n_target == 0:
        print(f"[{model}] no Path-A rows need merging — already done")
        return True

    n_filled = 0
    for idx in g[target_mask].index:
        row = g.loc[idx]
        key = (int(row["post_idx"]), str(row["Paraphrased Question"]))
        if key in z_lookup:
            g.at[idx, "generated_response"] = z_lookup[key]
            n_filled += 1

    g.to_csv(gen_path, index=False)
    print(f"[{model}] filled {n_filled}/{n_target} Path-A rows with zero-shot responses")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="Single model slug; default = all 6")
    args = parser.parse_args()
    targets = [args.model] if args.model else MODELS
    for m in targets:
        merge_for_model(m)


if __name__ == "__main__":
    main()
