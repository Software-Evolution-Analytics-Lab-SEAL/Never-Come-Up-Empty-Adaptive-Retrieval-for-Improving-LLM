"""
Final-answer generation for the six evaluated LLMs — for both RAG and zero-shot.

Modes:
    --mode rag       Reads retrieval CSVs from data/retrieval_results/,
                     writes data/generation/{model}/{csv_name}.csv
    --mode zeroshot  Answers the 666 test questions with no context,
                     writes data/baseline_zeroshot/{model}.csv

Both modes are resumable and share the same OpenRouter/Ollama clients and
model registry (`MODELS`).

Usage:
    python generation.py --mode rag       --model llama-3.1-8b --csv HB1_0.7
    python generation.py --mode rag       --model granite-3.1-8b --pipeline HB1
    python generation.py --mode rag       --model llama-3.1-8b --all
    python generation.py --mode zeroshot  --model llama-3.1-8b
    python generation.py --mode zeroshot  --all
"""
import argparse
import glob
import os
import sys
import time

import pandas as pd
import requests
from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


# --- Prompts ---------------------------------------------------------------

RAG_SYSTEM_PROMPT = """
You are a knowledgeable and helpful assistant. The user has asked a question on Stack Overflow.
Use the provided context to craft an accurate, concise, and highly relevant response.
Present your answer in a clear and well-structured paragraph format, avoiding the use of bullet points or lists.
DO NOT GENREATE INCOMPLETE CODE AND EXCESSIVE CODE TO DISTRACT PEOPLE!
"""

RAG_USER_TEMPLATE = """\
### QUESTION:
{question}

### CONTEXT:
{context}

Please provide your best answer below:

"""

ZS_SYSTEM_PROMPT = (
    "You are an expert programmer answering Stack Overflow questions. "
    "Provide clear, accurate, and concise answers to the Question "
    "and end with 'END_OF_ANSWER'."
)


# --- Model registry --------------------------------------------------------

# (endpoint, identifier, fallback_endpoint, fallback_id)
#   endpoint: "openrouter" | "ollama"
# Local models are served from an in-house Ollama instance on the A100 host.
# Only the two frontier LLMs (GPT-4.1 and DeepSeek-r1-70B) are accessed via OpenRouter.
MODELS = {
    "llama-3.1-8b":    ("ollama",     "llama3.1:8b-instruct-fp16",   None, None),
    "mistral-7b":      ("ollama",     "mistral:7b-instruct-v0.3",    None, None),
    "granite-3.1-8b":  ("ollama",     "granite3.1:8b-instruct",      None, None),
    "qwen3-8b":        ("ollama",     "qwen3:8b",                    None, None),
    "deepseek-r1-70b": ("openrouter", "deepseek/deepseek-r1",        None, None),
    "gpt-4.1":         ("openrouter", "openai/gpt-4.1",              None, None),
}

MAX_TOKENS = 512
RETRIES = 3


# --- Clients ---------------------------------------------------------------

_or_client = None
def or_client():
    global _or_client
    if _or_client is None:
        _or_client = OpenAI(base_url=config.OPENROUTER_BASE_URL,
                             api_key=config.get_openrouter_api_key())
    return _or_client


def _call_openrouter(model_id, system, user, retries=RETRIES):
    for attempt in range(retries):
        try:
            r = or_client().chat.completions.create(
                model=model_id,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=0.7, max_tokens=MAX_TOKENS,
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return f"[ERROR] {e}"


def _call_ollama(model_id, system, user, retries=RETRIES):
    payload = {"model": model_id, "stream": False,
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": user}],
               "options": {"temperature": 0.7, "num_predict": MAX_TOKENS}}
    for attempt in range(retries):
        try:
            r = requests.post("http://localhost:11434/api/chat", json=payload, timeout=900)
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return f"[ERROR] {e}"


def _dispatch(model_slug, system, user):
    endpoint, model_id, fallback_ep, fallback_id = MODELS[model_slug]
    if endpoint == "openrouter":
        return _call_openrouter(model_id, system, user)
    if endpoint == "ollama":
        out = _call_ollama(model_id, system, user)
        if out.startswith("[ERROR]") and fallback_ep == "openrouter":
            print(f"  [fallback] ollama failed, trying OpenRouter ({fallback_id})")
            return _call_openrouter(fallback_id, system, user)
        return out
    raise ValueError(f"Unknown endpoint {endpoint}")


def call_model(model_slug, question, context):
    """RAG-mode call (kept public — imported by scripts like retrieval.py)."""
    return _dispatch(model_slug, RAG_SYSTEM_PROMPT,
                     RAG_USER_TEMPLATE.format(question=question, context=context))


def call_zeroshot(model_slug, question):
    return _dispatch(model_slug, ZS_SYSTEM_PROMPT, question)


# --- RAG mode --------------------------------------------------------------

def _process_rag_csv(model_slug, csv_name):
    in_path = os.path.join(config.RETRIEVAL_DIR, f"{csv_name}.csv")
    if not os.path.exists(in_path):
        print(f"[{model_slug}/{csv_name}] ERROR: {in_path} does not exist"); return
    out_dir = os.path.join(config.DATA_DIR, "generation", model_slug)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{csv_name}.csv")

    df = pd.read_csv(in_path)
    if os.path.exists(out_path):
        prior = pd.read_csv(out_path)
        if len(prior) == len(df):
            df["generated_response"] = prior["generated_response"].fillna("").astype(str)
            print(f"[{model_slug}/{csv_name}] Resuming — existing has {len(prior)} rows")
    if "generated_response" not in df.columns:
        df["generated_response"] = ""
    df["generated_response"] = df["generated_response"].fillna("").astype(str)

    todo_mask = (df["generated_response"].str.strip() == "") & (df["num_retrieved"] > 0)
    todo = df[todo_mask]
    print(f"[{model_slug}/{csv_name}] {len(todo)}/{len(df)} rows to generate")

    for idx, row in tqdm(todo.iterrows(), total=len(todo), desc=f"{model_slug}/{csv_name}"):
        q = str(row.get("Paraphrased Question", "") or "")
        ctx = str(row.get("retrieved_context", "") or "")
        if not q or not ctx:
            continue
        df.at[idx, "generated_response"] = call_model(model_slug, q, ctx)
        if (todo.index.get_loc(idx) + 1) % 20 == 0:
            df.to_csv(out_path, index=False)
    df.to_csv(out_path, index=False)
    print(f"[{model_slug}/{csv_name}] Saved {out_path}")


def _all_rag_csv_names():
    paths = sorted(glob.glob(os.path.join(config.RETRIEVAL_DIR, "*.csv")))
    return [os.path.basename(p)[:-4] for p in paths]


# --- Zero-shot mode --------------------------------------------------------

def _process_zeroshot(model_slug):
    out_dir = os.path.join(config.DATA_DIR, "baseline_zeroshot")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{model_slug}.csv")

    df = pd.read_csv(config.TEST_SET_PATH)
    print(f"[{model_slug}] Loaded {len(df)} test queries")
    if os.path.exists(out_path):
        prior = pd.read_csv(out_path)
        if len(prior) == len(df):
            df["generated_response"] = prior["generated_response"].fillna("").astype(str)
            print(f"[{model_slug}] Resuming from existing output")
    if "generated_response" not in df.columns:
        df["generated_response"] = ""
    df["generated_response"] = df["generated_response"].fillna("").astype(str)

    todo_mask = df["generated_response"].str.strip() == ""
    todo = df[todo_mask]
    print(f"[{model_slug}] {len(todo)}/{len(df)} rows to generate")

    for idx, row in tqdm(todo.iterrows(), total=len(todo), desc=f"zs/{model_slug}"):
        q = str(row.get("Paraphrased Question", "") or "")
        if not q.strip():
            continue
        df.at[idx, "generated_response"] = call_zeroshot(model_slug, q)
        if (todo.index.get_loc(idx) + 1) % 20 == 0:
            df.to_csv(out_path, index=False)
    df.to_csv(out_path, index=False)
    print(f"[{model_slug}] Saved {out_path}")


# --- Main ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rag", "zeroshot"], default="rag")
    ap.add_argument("--model", choices=list(MODELS.keys()), default=None)
    ap.add_argument("--csv", default=None, help="Single retrieval CSV (rag mode).")
    ap.add_argument("--pipeline", default=None, help="Run all thresholds of a pipeline (rag mode).")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Used with --pipeline for a specific threshold.")
    ap.add_argument("--all", action="store_true", help="Run all CSVs / all models.")
    args = ap.parse_args()

    if args.mode == "zeroshot":
        models = list(MODELS.keys()) if args.all else [args.model]
        if not models[0]:
            ap.error("--model required in zeroshot mode (or use --all)")
        for m in models:
            _process_zeroshot(m)
        return

    if not args.model:
        ap.error("--model required in rag mode")
    if args.all:
        names = _all_rag_csv_names()
    elif args.csv:
        names = [args.csv]
    elif args.pipeline:
        if args.threshold is not None:
            names = [f"{args.pipeline}_{args.threshold:.1f}"]
        elif args.pipeline in ("BM25", "RAGFUSION"):
            names = [args.pipeline]
        else:
            names = [f"{args.pipeline}_{t:.1f}" for t in config.THRESHOLDS
                     if os.path.exists(os.path.join(config.RETRIEVAL_DIR, f"{args.pipeline}_{t:.1f}.csv"))]
    else:
        ap.error("Provide --all, --csv, or --pipeline")

    print(f"Model: {args.model}  endpoint: {MODELS[args.model][0]}")
    print(f"Will process {len(names)} CSV(s): {names[:5]}{' ...' if len(names) > 5 else ''}")
    for name in names:
        _process_rag_csv(args.model, name)


if __name__ == "__main__":
    main()
