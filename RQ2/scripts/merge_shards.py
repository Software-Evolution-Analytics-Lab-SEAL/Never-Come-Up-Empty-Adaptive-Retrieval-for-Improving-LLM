"""
Merge sharded generation outputs into a single CSV per (model, mode).

For each model × mode in data/generation/{model}/{mode}.shard*_of*.csv:
  - Concatenate rows in query_idx order
  - Write data/generation/{model}/{mode}.csv
  - Sanity-check: should end up with 5510 rows per model

Usage:
    python3 merge_shards.py                                   # merge all 6 models
    python3 merge_shards.py --model llama-3.1-8b --mode adaptive
"""
import os
import sys
import glob
import argparse
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config


MODELS = ["llama-3.1-8b", "gpt-4.1", "qwen3-8b", "mistral-7b", "deepseek-r1-70b", "granite-3.1-8b"]
MODES = ["adaptive", "zeroshot"]


def merge(model, mode):
    d = os.path.join(config.DATA_DIR, "generation", model)
    shards = sorted(glob.glob(os.path.join(d, f"{mode}.shard*_of*.csv")))
    if not shards:
        print(f"[{model}/{mode}] no shards found")
        return
    parts = [pd.read_csv(s) for s in shards]
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values("query_idx").reset_index(drop=True)
    out = os.path.join(d, f"{mode}.csv")
    df.to_csv(out, index=False)
    n = len(df)
    r = df.get("generated_response", pd.Series([""])).astype(str).fillna("")
    filled = ((r.str.strip() != "") & (r.str.strip().str.lower() != "nan")).sum()
    print(f"[{model}/{mode}] merged {len(shards)} shards -> {out}  ({filled}/{n} filled)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--mode", default=None, choices=MODES)
    args = ap.parse_args()

    models = [args.model] if args.model else MODELS
    modes = [args.mode] if args.mode else MODES
    for m in models:
        for md in modes:
            merge(m, md)


if __name__ == "__main__":
    main()
