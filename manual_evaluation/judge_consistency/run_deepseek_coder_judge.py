"""
Run deepseek-coder-v2:16b as an LLM judge on the 54 annotated rows of
random_sample_for_alignment_testing_54.xlsx, using the same v1 prompt as
step4_v1prompt_mini.py. Save per-row scores for downstream consistency
analysis.

Prompt template (identical to the v1 production judge):
  System: "You are an expert programming evaluator. Rate the following
           answer to a Stack Overflow question. ... Output a single
           integer score from 1 to 10, where 10 is the best.
           Output ONLY the number, nothing else."
  User:   [QUESTION] ... [ANSWER TO EVALUATE] ... [REFERENCE ACCEPTED ANSWER] ...

Output: deepseek_coder_v2_16b_scores.csv
"""
import os
import sys
import re
import time
import json

import pandas as pd
from tqdm import tqdm
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(HERE, "random_sample_for_alignment_testing_54.xlsx")
OUTPUT = os.path.join(HERE, "deepseek_coder_v2_16b_scores.csv")

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "deepseek-coder-v2:16b"

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


def judge_ollama(question, answer, accepted_answer, retries=3):
    prompt = JUDGE_USER_TEMPLATE.format(
        question=question, answer=answer, accepted_answer=accepted_answer
    )
    payload = {
        "model": MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "options": {"temperature": 0.3, "num_predict": 200},
    }
    for attempt in range(retries):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=1800)
            r.raise_for_status()
            data = r.json()
            text = data.get("message", {}).get("content", "").strip()
            # Parse 1-10 integer
            for token in re.findall(r"\d+", text):
                s = int(token)
                if 1 <= s <= 10:
                    return s, text
            return -1, text
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  [API error] {e}")
                return -1, str(e)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", type=str, default="",
                    help="Row slice like '0:18', '18:36', '36:54' (default: all)")
    ap.add_argument("--shard", type=str, default="",
                    help="Tag to add to output filename, e.g. 'shard0'")
    args = ap.parse_args()

    df = pd.read_excel(INPUT)
    annotated = df[df["p1"].notna() & df["p2"].notna()].reset_index(drop=True)
    print(f"Loaded {len(df)} rows; {len(annotated)} have annotations")

    if args.slice:
        a, b = args.slice.split(":")
        a = int(a) if a else 0
        b = int(b) if b else len(annotated)
        annotated = annotated.iloc[a:b].reset_index(drop=True)
        print(f"Using slice {args.slice}: {len(annotated)} rows")

    out_path = OUTPUT.replace(".csv", f"_{args.shard}.csv") if args.shard else OUTPUT

    if os.path.exists(out_path):
        prior = pd.read_csv(out_path)
        # Only treat rows with valid scores (>0) as done; retry -1 / timeouts
        done_idx = set(prior.loc[prior["deepseek_score"] > 0, "row_idx"].tolist())
        print(f"Resuming: {len(prior)} previously attempted, {len(done_idx)} valid — retrying the rest")
    else:
        prior = None
        done_idx = set()

    # Keep only already-valid rows from prior; the -1s will be re-attempted
    if prior is not None:
        records = prior[prior["deepseek_score"] > 0].to_dict("records")
    else:
        records = []

    for i, row in tqdm(annotated.iterrows(), total=len(annotated), desc="deepseek-coder"):
        if i in done_idx:
            continue
        q = str(row["Question2"]) if pd.notna(row.get("Question2")) else str(row.get("Question1", ""))
        ans = str(row.get("Selected_Answer", ""))
        gold = str(row.get("Accepted_Answer", ""))
        score, raw = judge_ollama(q, ans, gold)
        records.append({
            "row_idx": int(i),
            "Question": q,
            "Selected_RAG": row.get("Selected_RAG", ""),
            "Selected_Score": int(row.get("Selected_Score", 0)) if pd.notna(row.get("Selected_Score")) else -1,
            "p1": float(row.get("p1")) if pd.notna(row.get("p1")) else None,
            "p2": float(row.get("p2")) if pd.notna(row.get("p2")) else None,
            "deepseek_score": score,
            "deepseek_raw": raw,
        })
        # Incremental save every 5
        if (i + 1) % 5 == 0:
            pd.DataFrame(records).to_csv(out_path, index=False)

    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"Saved {out_path} with {len(records)} rows")


if __name__ == "__main__":
    main()
