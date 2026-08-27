"""
Convert the SO knowledge base (OVO_data pickle) into the TSV format expected
by Self-RAG's `generate_passage_embeddings.py` and `passage_retrieval.py`.

Format (per src/data.py:load_passages):
    First line:  id<TAB>text<TAB>title    (header)
    Each row:    <post_idx><TAB><accepted_answer><TAB><question_title>

We map:
    id    = OVO_data post index (0-indexed string)
    text  = raw_accepted_answer
    title = raw_question

We use csv.writer with QUOTE_MINIMAL so tabs / newlines / quotes inside
text or title are properly escaped.

Output: ../data/so_kb_passages.tsv  (~7-10 GB expected for 3.4M posts).
"""
import os
import sys
import csv
import pickle

HERE = os.path.dirname(os.path.abspath(__file__))
RQ1_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "RQ1_n666"))
sys.path.insert(0, RQ1_DIR)
import config

OUT_DIR = os.path.join(HERE, "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.abspath(os.path.join(OUT_DIR, "so_kb_passages.tsv"))


def main():
    print(f"[build_so_kb_tsv] Loading OVO_data from {config.OVO_PATH} ...")
    with open(config.OVO_PATH, "rb") as f:
        ovo = pickle.load(f)
    print(f"[build_so_kb_tsv] OVO_data loaded: {len(ovo):,} posts")

    if os.path.exists(OUT_PATH):
        print(f"[build_so_kb_tsv] {OUT_PATH} already exists, skipping. Delete to rebuild.")
        return

    n_kept = 0
    n_skipped = 0
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        w.writerow(["id", "text", "title"])
        for i, post in enumerate(ovo):
            q = post.get("raw_question", "") or ""
            a = post.get("raw_accepted_answer", "") or ""
            if not isinstance(q, str):
                q = str(q)
            if not isinstance(a, str):
                a = str(a)
            q = q.strip()
            a = a.strip()
            if not a:
                n_skipped += 1
                continue
            w.writerow([str(i), a, q])
            n_kept += 1
            if (i + 1) % 500000 == 0:
                print(f"  wrote {i+1:,} ...")

    size_gb = os.path.getsize(OUT_PATH) / 1e9
    print(f"[build_so_kb_tsv] Wrote {n_kept:,} passages ({n_skipped:,} skipped, no answer). "
          f"File size: {size_gb:.2f} GB")
    print(f"[build_so_kb_tsv] Output: {OUT_PATH}")


if __name__ == "__main__":
    main()
