"""
RQ2 answer generation — RAG (adaptive context) and zero-shot in one entry point.

    --mode rag       Uses the per-model optimal pipeline's adaptive retrieval CSV
                     (data/retrieval/{HB1|HYB}_adaptive.csv) as context.
                     Output: data/generation/{model}/adaptive.csv
    --mode zeroshot  No retrieval. Uses raw Titles from data/unseen_5510.csv
                     (override with --input).
                     Output: data/generation/{model}/zeroshot.csv

Both modes are resumable and support --num-shards / --shard for parallel runs.

Usage:
    python generation.py --mode rag      --model llama-3.1-8b
    python generation.py --mode zeroshot --model llama-3.1-8b
"""
import argparse
import os
import sys

import pandas as pd
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config
from llm_clients import MODELS, call_model


RAG_SYSTEM = """You are a senior software engineer answering a developer's question. You have been given retrieved reference material from authoritative Stack Overflow posts. Your job is to write an answer that is comprehensive, technically precise, and directly useful.

Strict requirements:
- Ground every factual claim in the retrieved context where the context applies; supplement with your own expert knowledge only when the context is incomplete or does not cover a specific detail.
- If a code example is appropriate, include a COMPLETE, self-contained, runnable snippet (with imports, proper variable setup, and output where relevant). DO NOT provide partial, truncated, or placeholder code.
- Explain the underlying reasoning, relevant concepts, and the why — not just the mechanical steps.
- Anticipate and address common pitfalls, edge cases, or misconceptions the asker is likely to encounter.
- Write in clear, professional prose as one or two connected paragraphs. Do NOT use bullet points, numbered lists, or markdown headers.
- Be thorough but focused: no filler, no off-topic tangents, no meta-commentary about the question itself.

Output a single self-contained answer that a senior engineer would consider authoritative, correct, and genuinely helpful."""

RAG_USER_TEMPLATE = """### QUESTION:
{question}

### RETRIEVED REFERENCE CONTEXT:
{context}

### YOUR ANSWER:
"""

ZS_SYSTEM = (
    "You are an expert programmer answering Stack Overflow questions. "
    "Provide clear, accurate, and concise answers to the Question "
    "and end with 'END_OF_ANSWER'."
)


def _run(model, mode, in_path, out_path, num_shards, shard):
    df = pd.read_csv(in_path)
    if num_shards > 1:
        df = df.iloc[shard::num_shards].reset_index(drop=True)
    label = f"{model}/{mode}"
    print(f"[{label} shard {shard}/{num_shards}] Loaded {len(df)} rows from {in_path}")

    if os.path.exists(out_path):
        prior = pd.read_csv(out_path)
        if "generated_response" in prior.columns and len(prior) == len(df):
            df["generated_response"] = prior["generated_response"].fillna("").astype(str)
            print(f"[{label}] Resuming from {out_path}")
    if "generated_response" not in df.columns:
        df["generated_response"] = ""
    df["generated_response"] = df["generated_response"].fillna("").astype(str)

    todo_mask = df["generated_response"].str.strip() == ""
    todo = df[todo_mask]
    print(f"[{label}] {len(todo)}/{len(df)} rows to generate")

    for idx, row in tqdm(todo.iterrows(), total=len(todo), desc=label):
        question = str(row["Title"])
        if not question.strip():
            continue
        if mode == "rag":
            context = str(row.get("retrieved_context", "") or "")
            user = RAG_USER_TEMPLATE.format(question=question, context=context)
            ans = call_model(model, RAG_SYSTEM, user)
        else:                                            # zeroshot
            ans = call_model(model, ZS_SYSTEM, question)
        df.at[idx, "generated_response"] = ans
        if (todo.index.get_loc(idx) + 1) % 20 == 0:
            df.to_csv(out_path, index=False)

    df.to_csv(out_path, index=False)
    print(f"[{label}] Saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rag", "zeroshot"], required=True)
    ap.add_argument("--model", required=True, choices=list(MODELS.keys()))
    ap.add_argument("--input", default=None, help="Override input CSV path")
    ap.add_argument("--output", default=None, help="Override output CSV path")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    if args.mode == "rag":
        pipeline = config.OPTIMAL_PIPELINE[args.model]
        in_path = args.input or os.path.join(config.DATA_DIR, "retrieval", f"{pipeline}_adaptive.csv")
        out_name = "adaptive"
    else:
        in_path = args.input or os.path.join(config.DATA_DIR, "unseen_5510.csv")
        out_name = "zeroshot"

    out_dir = args.output_dir or os.path.join(config.DATA_DIR, "generation", args.model)
    os.makedirs(out_dir, exist_ok=True)
    tag = "" if args.num_shards == 1 else f".shard{args.shard}_of{args.num_shards}"
    out_path = args.output or os.path.join(out_dir, f"{out_name}{tag}.csv")

    _run(args.model, args.mode, in_path, out_path, args.num_shards, args.shard)


if __name__ == "__main__":
    main()
