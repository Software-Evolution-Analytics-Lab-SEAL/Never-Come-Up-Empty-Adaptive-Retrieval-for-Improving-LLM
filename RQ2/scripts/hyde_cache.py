"""
Parallel HyDE precomputation for the 5510 RQ2 unseen queries.

Fills data/hyde_cache_unseen.json using concurrent OpenRouter calls.
Resumable: skips any Title that already has a non-empty cached answer.

Usage:
    python3 hyde_cache.py --workers 8
"""
import os
import sys
import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import pandas as pd
from tqdm import tqdm
from openai import OpenAI

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config  # RQ2 config

HYDE_CACHE_PATH = os.path.join(config.DATA_DIR, "hyde_cache_unseen.json")
HYDE_SYSTEM = (
    "You are an expert programmer. Given a programming question, "
    "generate a hypothetical answer that would be helpful. "
    "Be specific and technical."
)

_CACHE_LOCK = Lock()


def load_cache():
    if os.path.exists(HYDE_CACHE_PATH):
        with open(HYDE_CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_cache(cache):
    tmp = HYDE_CACHE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f, indent=2)
    os.replace(tmp, HYDE_CACHE_PATH)


def generate_hyde(client, question, retries=3):
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=config.GPT4O_MODEL,
                messages=[
                    {"role": "system", "content": HYDE_SYSTEM},
                    {"role": "user", "content": question},
                ],
                temperature=0.7,
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(config.DATA_DIR, "unseen_5510.csv"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--save-every", type=int, default=50,
                    help="Flush cache to disk every N completions")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    titles = df["Title"].astype(str).tolist()
    titles = [t for t in titles if t.strip()]
    titles = list(dict.fromkeys(titles))  # preserve order, dedupe
    print(f"Loaded {len(titles)} unique non-empty titles")

    cache = load_cache()
    todo = [t for t in titles if not cache.get(t)]
    print(f"Cache has {sum(1 for v in cache.values() if v)} non-empty; {len(todo)} to compute")

    if not todo:
        print("Nothing to do.")
        return

    api_key = config.get_openrouter_api_key()
    client = OpenAI(base_url=config.OPENROUTER_BASE_URL, api_key=api_key)

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex, \
         tqdm(total=len(todo), desc="HyDE") as pbar:
        futures = {ex.submit(generate_hyde, client, q): q for q in todo}
        for fut in as_completed(futures):
            q = futures[fut]
            try:
                ans = fut.result()
            except Exception:
                ans = ""
            with _CACHE_LOCK:
                cache[q] = ans
                done += 1
                if done % args.save_every == 0:
                    save_cache(cache)
            pbar.update(1)

    save_cache(cache)
    non_empty = sum(1 for v in cache.values() if v)
    print(f"Done. cache={len(cache)} ({non_empty} non-empty)")


if __name__ == "__main__":
    main()
