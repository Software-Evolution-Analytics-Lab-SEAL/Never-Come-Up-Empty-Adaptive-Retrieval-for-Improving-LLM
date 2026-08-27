"""
Build the Synthetic Question Set (n=666) by sampling directly from the Stack
Overflow knowledge base and paraphrasing each question with GPT-4o. The sample
size corresponds to a 99% confidence level and a 5% margin of error.

Output: data/synthetic_questions_n666.csv with columns:
    Question              — original Stack Overflow question title
    Accepted Answer       — raw text of the accepted answer
    Paraphrased Question  — GPT-4o paraphrase used as the input query
    post_idx              — index into the KB (for traceability)

Usage:
    python build_synthetic_set.py
"""
import os
import random
import sys
import time

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import kb


PARAPHRASE_SYSTEM = (
    "You are an expert at rephrasing technical Stack Overflow questions. "
    "Given a programming question, produce a single paraphrased version that "
    "preserves the technical meaning but rewords it. Output ONLY the paraphrased "
    "question, nothing else."
)


def paraphrase_question(client, question, retries=3):
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=config.GPT4O_MODEL,
                messages=[{"role": "system", "content": PARAPHRASE_SYSTEM},
                          {"role": "user", "content": question}],
                temperature=0.7,
                max_tokens=150,
            )
            text = r.choices[0].message.content.strip()
            for prefix in ("Paraphrased question:", "Paraphrased:", "Q:", "Question:"):
                if text.lower().startswith(prefix.lower()):
                    text = text[len(prefix):].strip()
            return text.strip('"').strip("'").strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  [paraphrase ERROR] {e}")
                return question


def main():
    out_path = config.TEST_SET_PATH
    if os.path.exists(out_path):
        print(f"[SKIP] {out_path} already exists")
        return

    print(f"[1/3] Loading KB ...")
    ovo = kb.load_ovo_data()
    raw_qs = kb.raw_questions_list()
    print(f"  total posts: {len(ovo)}")

    print(f"[2/3] Sampling {config.N_TARGET} posts (seed={config.SEED}) ...")
    rng = random.Random(config.SEED)
    idxs = rng.sample(range(len(ovo)), config.N_TARGET)

    print(f"[3/3] Paraphrasing {config.N_TARGET} questions with {config.GPT4O_MODEL} ...")
    client = OpenAI(base_url=config.OPENROUTER_BASE_URL,
                    api_key=config.get_openrouter_api_key())

    rows = []
    for idx in tqdm(idxs, desc="paraphrasing"):
        original_q = raw_qs[idx]
        para = paraphrase_question(client, original_q)
        rows.append({
            "Question": original_q,
            "Accepted Answer": ovo[idx]["raw_accepted_answer"],
            "Paraphrased Question": para,
            "post_idx": idx,
        })
        time.sleep(0.1)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
