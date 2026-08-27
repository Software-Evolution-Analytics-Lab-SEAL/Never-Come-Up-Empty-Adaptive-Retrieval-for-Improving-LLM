"""
Replace SELFRAG.csv's retrieved_context with the clean text from
answers_context.pkl (which is the source the BM25, RAG-Fusion, and HB1
baselines use). The Contriever-built TSV used OVO_data's
raw_accepted_answer which contains HTML markup (<p>, <pre>, <code>); the
post indices in OVO_data are 1:1 aligned with answers_context.pkl, so we
can remap text without re-encoding or re-retrieving.

The retrieved_post_ids and retrieval_scores columns are unchanged. Only
retrieved_context is updated. This makes the SELFRAG context fair to
compare against the other baselines (same text source).
"""
import os
import sys
import pickle
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RQ1_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "RQ1_n666"))
sys.path.insert(0, RQ1_DIR)
import config

OUT_PATH = os.path.join(config.RETRIEVAL_DIR, "SELFRAG.csv")
BACKUP_PATH = OUT_PATH + ".html_version.bak"


def main():
    if not os.path.exists(OUT_PATH):
        raise FileNotFoundError(OUT_PATH)
    print(f"[fix_text] Backing up {OUT_PATH} -> {BACKUP_PATH}")
    if not os.path.exists(BACKUP_PATH):
        import shutil
        shutil.copy(OUT_PATH, BACKUP_PATH)

    print("[fix_text] Loading answers_context.pkl ...")
    with open(config.ANSWERS_CONTEXT_PATH, "rb") as f:
        ac = pickle.load(f)
    print(f"[fix_text] answers_context.pkl: {len(ac):,} entries")

    df = pd.read_csv(OUT_PATH)
    print(f"[fix_text] SELFRAG.csv: {len(df)} rows")

    new_contexts = []
    for _, row in df.iterrows():
        ids_raw = str(row.get("retrieved_post_ids", "") or "").strip()
        if not ids_raw:
            new_contexts.append(row.get("retrieved_context", ""))
            continue
        ids = [int(x) for x in ids_raw.split(",") if x.strip()]
        lines = []
        for k, i in enumerate(ids):
            if 0 <= i < len(ac):
                lines.append(f"{k+1}. {ac[i]}")
            else:
                lines.append(f"{k+1}. [missing post_id={i}]")
        new_contexts.append("\n".join(lines))

    df["retrieved_context"] = new_contexts
    df.to_csv(OUT_PATH, index=False)
    print(f"[fix_text] Updated retrieved_context in {OUT_PATH}")
    sample = df.iloc[0]["retrieved_context"][:300]
    print(f"[fix_text] Sample new ctx (row 0, 300 chars): {sample!r}")


if __name__ == "__main__":
    main()
