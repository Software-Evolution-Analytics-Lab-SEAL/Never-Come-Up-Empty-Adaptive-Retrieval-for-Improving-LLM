"""
Fill the primary_label column in 03_final_answer_usefulness.csv with a
binary LLM-as-judge call: "Is this generated answer useful for answering
the question?" (yes/no).

The audit task is binary, so the LLM judge mirrors the human task exactly.
Resumable: rows already labelled are skipped.

Usage:
    python3 11_judge_final_answer_usefulness.py
"""
import os
import sys
import re
import json
import time
import argparse

import pandas as pd
from tqdm import tqdm
from openai import OpenAI

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
import config

CSV = os.path.abspath(os.path.join(HERE, "..", "manual_evaluation",
                                  "03_final_answer_usefulness.csv"))
JUDGE_MODEL = "openai/gpt-4o"

PROMPT_SYSTEM = (
    "You are evaluating whether a generated answer is helpful for answering "
    "a Stack Overflow programming question. Make a binary decision.\n\n"
    "Output a single line of JSON, no other text:\n"
    "{\"helpful\": true|false, \"reason\": \"<one short sentence>\"}\n\n"
    "Helpful means the answer addresses the question with information that "
    "would actually help the developer. Off-topic, vague boilerplate, or "
    "unrelated content is NOT helpful."
)
PROMPT_USER = "[QUESTION]\n{question}\n\n[GENERATED ANSWER]\n{answer}"


_client = None
def client():
    global _client
    if _client is None:
        _client = OpenAI(base_url=config.OPENROUTER_BASE_URL,
                         api_key=config.get_openrouter_api_key())
    return _client


def judge(q, a, retries=3):
    prompt = PROMPT_USER.format(question=str(q)[:1500], answer=str(a)[:3500])
    for attempt in range(retries):
        try:
            r = client().chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "system", "content": PROMPT_SYSTEM},
                          {"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=80,
            )
            text = r.choices[0].message.content.strip()
            m = re.search(r"\{.*\}", text, re.S)
            if not m:
                return None, f"no_json: {text[:120]}"
            obj = json.loads(m.group(0))
            v = obj.get("helpful")
            if isinstance(v, bool):
                return v, str(obj.get("reason", ""))[:240]
            if isinstance(v, str):
                return v.strip().lower() in ("true", "yes", "1"), str(obj.get("reason", ""))[:240]
            return None, f"bad_value: {v}"
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None, f"error: {e}"


def main():
    df = pd.read_csv(CSV)
    todo = df[df["primary_label"].astype(str).str.strip().isin(["", "nan", "None"])].index.tolist()
    print(f"Total rows: {len(df)}.  To judge: {len(todo)}")
    for i in tqdm(todo, desc="judge"):
        q = df.at[i, "question"]
        a = df.at[i, "generated_answer"]
        v, reason = judge(q, a)
        df.at[i, "primary_label"] = "" if v is None else ("True" if v else "False")
        df.at[i, "primary_reason"] = reason
        if (todo.index(i) + 1) % 25 == 0:
            df.to_csv(CSV, index=False)
    df.to_csv(CSV, index=False)
    nv = df["primary_label"].isin(["True", "False"]).sum()
    print(f"Filled {nv}/{len(df)} primary_label cells. Saved {CSV}")


if __name__ == "__main__":
    main()
