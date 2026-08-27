"""
Generic Ollama-based LLM judge runner.

Uses the same v1 prompt as the production judge (identical to
run_deepseek_coder_judge.py). Runs on the 54 annotated rows and saves
per-row scores to {tag}_scores_{shard}.csv.

Usage:
    python3 run_ollama_judge.py --model deepseek-coder:33b --tag deepseek_coder_33b \
        --slice 0:18 --shard shard0
"""
import os
import re
import time
import argparse

import pandas as pd
from tqdm import tqdm
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(HERE, "random_sample_for_alignment_testing_54.xlsx")
OLLAMA_URL = "http://localhost:11434/api/chat"

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


def judge_ollama(model, question, answer, accepted_answer, retries=3):
    prompt = JUDGE_USER_TEMPLATE.format(
        question=question, answer=answer, accepted_answer=accepted_answer
    )
    payload = {
        "model": model,
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
            for token in re.findall(r"\d+", text):
                s = int(token)
                if 1 <= s <= 10:
                    return s, text
            return -1, text
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return -1, str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Ollama model id, e.g. deepseek-coder:33b")
    ap.add_argument("--tag", required=True, help="Output file tag (used in filename)")
    ap.add_argument("--slice", default="", help="Row slice like '0:18'")
    ap.add_argument("--shard", default="", help="Shard tag for filename")
    args = ap.parse_args()

    df = pd.read_excel(INPUT)
    annotated = df[df["p1"].notna() & df["p2"].notna()].reset_index(drop=True)

    if args.slice:
        a, b = args.slice.split(":")
        a = int(a) if a else 0
        b = int(b) if b else len(annotated)
        annotated = annotated.iloc[a:b].reset_index(drop=True)

    suffix = f"_{args.shard}" if args.shard else ""
    out_path = os.path.join(HERE, f"{args.tag}_scores{suffix}.csv")

    prior = pd.read_csv(out_path) if os.path.exists(out_path) else None
    done_idx = set()
    records = []
    if prior is not None:
        done_idx = set(prior.loc[prior["score"] > 0, "row_idx"].tolist())
        records = prior[prior["score"] > 0].to_dict("records")
        print(f"Resuming: {len(prior)} attempted, {len(done_idx)} valid — retrying the rest")

    for i, row in tqdm(annotated.iterrows(), total=len(annotated), desc=args.tag):
        if i in done_idx:
            continue
        q = str(row["Question2"]) if pd.notna(row.get("Question2")) else str(row.get("Question1", ""))
        ans = str(row.get("Selected_Answer", ""))
        gold = str(row.get("Accepted_Answer", ""))
        score, raw = judge_ollama(args.model, q, ans, gold)
        records.append({
            "row_idx": int(i),
            "Question": q,
            "Selected_RAG": row.get("Selected_RAG", ""),
            "Selected_Score": int(row.get("Selected_Score", 0)) if pd.notna(row.get("Selected_Score")) else -1,
            "p1": float(row.get("p1")) if pd.notna(row.get("p1")) else None,
            "p2": float(row.get("p2")) if pd.notna(row.get("p2")) else None,
            "score": score,
            "raw": raw,
        })
        if (i + 1) % 5 == 0:
            pd.DataFrame(records).to_csv(out_path, index=False)

    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"Saved {out_path} with {len(records)} rows")


if __name__ == "__main__":
    main()
