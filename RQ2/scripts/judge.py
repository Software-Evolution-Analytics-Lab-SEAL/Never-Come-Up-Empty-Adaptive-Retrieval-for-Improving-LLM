"""
LLM-as-a-Judge for RQ2 — GPT-4o.

Scores generated responses against the accepted Stack Overflow answer for a
given (model, mode) pair. Modes:
    - adaptive : data/generation/{model}/adaptive.csv
    - zeroshot : data/generation/{model}/zeroshot.csv
    - accepted : data/unseen_5510.csv (self-scored; answer == accepted)

Output: data/evaluation/{model}/{mode}.csv with extra `judge_score` column.

Usage:
    python judge.py --model llama-3.1-8b --mode adaptive
    python judge.py --mode accepted
"""
import argparse
import os
import re
import sys
import time

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config
from llm_clients import MODELS


JUDGE_MODEL = "openai/gpt-4o"
EVAL_DIR = os.path.join(config.DATA_DIR, "evaluation")

JUDGE_SYSTEM_PROMPT = """You are an expert programming evaluator. Rate the following answer to a Stack Overflow question.

Please evaluate on these criteria:
1. Helpfulness - Does the answer address the question?
2. Technical Correctness - Is the answer technically accurate?
3. Level of Detail - Does the answer provide sufficient detail and explanation?

If there is incomplete code snippet in the response, deduct the overall score.

Output a single integer score from 1 to 10, where 10 is the best.
Output ONLY the number, nothing else."""

JUDGE_USER_TEMPLATE = """[QUESTION]
{question}

[ANSWER TO EVALUATE]
{answer}

[REFERENCE ACCEPTED ANSWER]
{accepted_answer}"""


_client = None
def client():
    global _client
    if _client is None:
        _client = OpenAI(base_url=config.OPENROUTER_BASE_URL,
                         api_key=config.get_openrouter_api_key())
    return _client


def judge_response(question, answer, gold, retries=3):
    prompt = JUDGE_USER_TEMPLATE.format(question=question, answer=answer,
                                        accepted_answer=gold)
    for attempt in range(retries):
        try:
            r = client().chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                          {"role": "user", "content": prompt}],
                temperature=0.3, max_tokens=20,
            )
            text = r.choices[0].message.content.strip()
            for token in re.findall(r"\d+", text):
                s = int(token)
                if 1 <= s <= 10:
                    return s
            return -1
        except Exception:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return -1


def load_source(model, mode, dataset=None):
    """Return df with columns: Title, Accepted_Answer, answer_to_score."""
    if mode == "accepted":
        path = os.path.join(config.DATA_DIR, f"{dataset}.csv") if dataset \
            else os.path.join(config.DATA_DIR, "unseen_5510.csv")
        df = pd.read_csv(path)
        df["answer_to_score"] = df["Accepted_Answer"].astype(str)
        return df
    gen_dir = os.path.join(config.DATA_DIR, "generation")
    if dataset:
        gen_dir = os.path.join(gen_dir, dataset)
    if mode == "adaptive":
        df = pd.read_csv(os.path.join(gen_dir, model, "adaptive.csv"))
    elif mode == "zeroshot":
        df = pd.read_csv(os.path.join(gen_dir, model, "zeroshot.csv"))
    else:
        raise ValueError(f"Unknown mode: {mode}")
    df["answer_to_score"] = df["generated_response"].astype(str).fillna("")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, choices=list(MODELS.keys()))
    ap.add_argument("--mode", required=True, choices=["adaptive", "zeroshot", "accepted"])
    ap.add_argument("--dataset", default=None,
                    help="Optional dataset tag (e.g. unseen_2025)")
    args = ap.parse_args()

    if args.mode in ("adaptive", "zeroshot") and not args.model:
        raise SystemExit("--model is required for adaptive/zeroshot modes")

    model_tag = args.model or "_global"
    ds_tag = args.dataset + "/" if args.dataset else ""
    out_dir = (os.path.join(EVAL_DIR, args.dataset, model_tag)
               if args.dataset else os.path.join(EVAL_DIR, model_tag))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.mode}.csv")

    df = load_source(args.model, args.mode, dataset=args.dataset)
    print(f"[{ds_tag}{model_tag}/{args.mode}] Loaded {len(df)} rows")

    if "judge_score" not in df.columns:
        df["judge_score"] = -1

    if os.path.exists(out_path):
        prior = pd.read_csv(out_path)
        if "judge_score" in prior.columns and len(prior) == len(df):
            df["judge_score"] = prior["judge_score"]
            print(f"[{model_tag}/{args.mode}] Resumed")

    answers = df["answer_to_score"].astype(str).fillna("")
    usable = ((answers.str.strip() != "")
              & (~answers.str.strip().str.lower().isin(["nan", "none"]))
              & (~answers.str.startswith("[ERROR]")))
    df.loc[(df["judge_score"] == -1) & (~usable), "judge_score"] = 0

    pending = df[(df["judge_score"] == -1) & usable]
    print(f"[{model_tag}/{args.mode}] {len(pending)} rows to judge")

    for i, (idx, row) in enumerate(tqdm(pending.iterrows(), total=len(pending),
                                        desc=f"{model_tag}/{args.mode}")):
        q = str(row.get("Title", ""))
        ans = str(row["answer_to_score"])
        gold = str(row.get("Accepted_Answer", row.get("Accepted Answer", "")))
        df.at[idx, "judge_score"] = judge_response(q, ans, gold)
        if (i + 1) % 20 == 0:
            df.to_csv(out_path, index=False)

    df.to_csv(out_path, index=False)
    n_valid = (df["judge_score"] > 0).sum()
    mean = df.loc[df["judge_score"] > 0, "judge_score"].mean() if n_valid else 0
    print(f"[{model_tag}/{args.mode}] Saved {out_path}  mean={mean:.2f}  n_valid={n_valid}/{len(df)}")


if __name__ == "__main__":
    main()
