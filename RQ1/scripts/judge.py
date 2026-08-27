"""
LLM-as-a-Judge — GPT-4o LLM-as-a-Judge.

Scores generated answers on helpfulness, technical correctness, and level of
detail. Resumable — skips rows that already have a judge_score.

Two input layouts:

    --mode rag       Reads data/generation/{model}/{csv_name}.csv
                     Writes data/evaluation/{model}/{csv_name}.csv
    --mode zeroshot  Reads data/baseline_zeroshot/{model}.csv
                     Writes data/evaluation/zeroshot/{model}.csv

Usage:
    python judge.py --mode rag      --model mistral-7b --csv HB1_0.7
    python judge.py --mode rag      --model mistral-7b --all
    python judge.py --mode zeroshot --all
"""
import argparse
import glob
import os
import sys
import time

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


JUDGE_MODEL = "openai/gpt-4o"
EVAL_DIR = os.path.join(config.DATA_DIR, "evaluation")

SYSTEM_PROMPT = """You are an expert programming evaluator. Rate the following answer to a Stack Overflow question.

Please evaluate on these criteria:
1. Helpfulness - Does the answer address the question?
2. Technical Correctness - Is the answer technically accurate?
3. Level of Detail - Does the answer provide sufficient detail and explanation?

If there is incomplete code snippet in the response, deduct the overall score.

Output a single integer score from 1 to 10, where 10 is the best.
Output ONLY the number, nothing else."""

USER_TEMPLATE = """[QUESTION]
{question}

[ANSWER TO EVALUATE]
{answer}

[REFERENCE ACCEPTED ANSWER]
{accepted_answer}"""


_client = None
def _get_client():
    global _client
    if _client is None:
        api_key = config.get_openrouter_api_key()
        _client = OpenAI(base_url=config.OPENROUTER_BASE_URL, api_key=api_key)
    return _client


def judge_response(question, answer, accepted_answer, retries=3):
    prompt = USER_TEMPLATE.format(question=question, answer=answer,
                                   accepted_answer=accepted_answer)
    for attempt in range(retries):
        try:
            r = _get_client().chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=20,
            )
            text = r.choices[0].message.content.strip()
            for token in text.split():
                try:
                    s = int(token)
                    if 1 <= s <= 10:
                        return s
                except ValueError:
                    continue
            try:
                s = int(float(text))
                if 1 <= s <= 10:
                    return s
            except ValueError:
                pass
            return -1
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  [API error] {e}")
                return -1


def _score_dataframe(df, out_path, label):
    """Fill judge_score column in `df`, saving `out_path` every 20 rows."""
    if "judge_score" not in df.columns:
        df["judge_score"] = -1
    if os.path.exists(out_path):
        prior = pd.read_csv(out_path)
        if "judge_score" in prior.columns and len(prior) == len(df):
            df["judge_score"] = prior["judge_score"]
            print(f"[{label}] resumed")

    responses = df["generated_response"].astype(str).fillna("")
    todo_mask = ((df["judge_score"] == -1)
                 & (responses.str.strip() != "")
                 & (~responses.str.startswith("[ERROR]")))
    todo = df[todo_mask]
    df.loc[(df["judge_score"] == -1) & (~todo_mask), "judge_score"] = 0

    print(f"[{label}] {len(todo)} rows to judge")
    for i, (idx, row) in enumerate(tqdm(todo.iterrows(), total=len(todo), desc=label)):
        q = str(row.get("Paraphrased Question", "") or "")
        ans = str(row.get("generated_response", "") or "")
        gold = str(row.get("Accepted Answer", "") or "")
        df.at[idx, "judge_score"] = judge_response(q, ans, gold)
        if (i + 1) % 20 == 0:
            df.to_csv(out_path, index=False)

    df.to_csv(out_path, index=False)
    n_valid = (df["judge_score"] > 0).sum()
    mean = df.loc[df["judge_score"] > 0, "judge_score"].mean() if n_valid > 0 else 0
    print(f"[{label}] saved {out_path}  mean={mean:.2f}  n_valid={n_valid}/{len(df)}")


def _process_rag(model_slug, csv_name):
    in_path = os.path.join(config.DATA_DIR, "generation", model_slug, f"{csv_name}.csv")
    if not os.path.exists(in_path):
        print(f"[{model_slug}/{csv_name}] missing: {in_path}"); return
    out_dir = os.path.join(EVAL_DIR, model_slug)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{csv_name}.csv")
    _score_dataframe(pd.read_csv(in_path), out_path, f"{model_slug}/{csv_name}")


def _process_zeroshot(model_slug):
    in_path = os.path.join(config.DATA_DIR, "baseline_zeroshot", f"{model_slug}.csv")
    if not os.path.exists(in_path):
        print(f"[{model_slug}] no zero-shot file at {in_path}"); return
    out_dir = os.path.join(EVAL_DIR, "zeroshot")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{model_slug}.csv")
    _score_dataframe(pd.read_csv(in_path), out_path, f"zs/{model_slug}")


def _all_csvs(model_slug):
    paths = sorted(glob.glob(os.path.join(config.DATA_DIR, "generation", model_slug, "*.csv")))
    return [os.path.basename(p)[:-4] for p in paths]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rag", "zeroshot"], default="rag")
    ap.add_argument("--model", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--csvs", default=None, help="Comma-separated CSV names (rag mode)")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if args.mode == "zeroshot":
        # Import MODELS lazily to avoid loading heavy libs when not needed
        from generation import MODELS
        models = list(MODELS.keys()) if args.all else [args.model]
        for m in models:
            _process_zeroshot(m)
        return

    if not args.model:
        ap.error("--model required in rag mode")
    if args.all:
        names = _all_csvs(args.model)
    elif args.csvs:
        names = [x.strip() for x in args.csvs.split(",")]
    elif args.csv:
        names = [args.csv]
    else:
        ap.error("need --all, --csv, or --csvs")

    print(f"Judging {args.model} on {len(names)} CSVs with {JUDGE_MODEL}")
    for name in names:
        _process_rag(args.model, name)


if __name__ == "__main__":
    main()
