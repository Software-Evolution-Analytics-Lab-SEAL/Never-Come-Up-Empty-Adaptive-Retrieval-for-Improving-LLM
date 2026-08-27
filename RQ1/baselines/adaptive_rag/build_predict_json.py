"""
Build the predict.json file for the trained Adaptive-RAG T5 classifier,
using our 666-question SE test set.

Adaptive-RAG's run_classifier.py expects validation/predict input as a JSON
list of dicts shaped like their `predict.json`:
    {
      "answer": "",            # empty - to be predicted
      "dataset_name": "...",
      "id": "<unique-id>",
      "question": "<text>",
      "total_answer": []
    }

We write to:
    upstream/classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/predict_n666.json

Then `run_classifier.py --do_eval --validation_file ...predict_n666.json`
will produce predictions in <output_dir>/eval_predictions.json (or similar).
"""
import os
import sys
import json
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RQ1_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "RQ1_n666"))
sys.path.insert(0, RQ1_DIR)
import config

UPSTREAM = os.path.join(HERE, "upstream")
OUT_PATH = os.path.join(
    UPSTREAM, "classifier", "data", "musique_hotpot_wiki2_nq_tqa_sqd",
    "predict_n666.json"
)


def main():
    df = pd.read_csv(config.TEST_SET_PATH)
    print(f"[build_predict] Loaded {len(df)} test queries from {config.TEST_SET_PATH}")

    records = []
    for i, row in df.iterrows():
        q = row.get("Paraphrased Question", "")
        if not isinstance(q, str) or not q.strip():
            continue
        records.append({
            "answer": "",
            "dataset_name": "stack_overflow_n666",
            "id": f"so_n666__{int(row.get('post_idx', i))}__{i}",
            "question": q.strip(),
            "total_answer": [],
        })

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(records, f, indent=2)
    print(f"[build_predict] Wrote {len(records)} records to {OUT_PATH}")


if __name__ == "__main__":
    main()
