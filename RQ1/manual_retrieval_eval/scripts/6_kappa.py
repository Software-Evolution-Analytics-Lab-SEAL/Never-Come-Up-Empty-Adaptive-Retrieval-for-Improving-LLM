"""
Cohen's kappa between LLM and human Usefulness labels on the audit subset.

Reads:
  audit/audit_pack_{HB1,HYB,BM25}.csv  (human's filled-in `human_useful`)
  audit/audit_answer_key.csv           (LLM labels)

Outputs:
  audit/kappa_results.csv  (per method + overall, kappa, agreement, n)
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIT = os.path.abspath(os.path.join(HERE, "..", "audit"))

FOCAL = ["HB1", "HYB", "BM25"]


def to_bool(s):
    return s.astype(str).str.strip().str.lower().isin(["true", "yes", "y", "1", "t"])


def main():
    key = pd.read_csv(os.path.join(AUDIT, "audit_answer_key.csv"))
    rows = []
    all_h, all_l = [], []
    for method in FOCAL:
        path = os.path.join(AUDIT, f"audit_pack_{method}.csv")
        if not os.path.exists(path):
            continue
        h = pd.read_csv(path)
        h = h[h["human_useful"].astype(str).str.strip() != ""]
        if h.empty:
            continue
        sub = key[key["method"] == method].copy()
        merged = h.merge(sub, on=["test_idx", "unit_id"], how="inner")
        if merged.empty:
            continue
        merged["human_b"] = to_bool(merged["human_useful"])
        merged["llm_b"] = to_bool(merged["llm_useful"])
        ka = cohen_kappa_score(merged["llm_b"], merged["human_b"])
        agr = float((merged["human_b"] == merged["llm_b"]).mean())
        cm = confusion_matrix(merged["llm_b"], merged["human_b"], labels=[True, False])
        rows.append({"method": method, "kappa": ka, "agreement": agr,
                     "n": int(len(merged)),
                     "tt_TT": int(cm[0, 0]), "tt_TF": int(cm[0, 1]),
                     "tt_FT": int(cm[1, 0]), "tt_FF": int(cm[1, 1])})
        all_h.extend(merged["human_b"].tolist())
        all_l.extend(merged["llm_b"].tolist())

    if all_h:
        ka = cohen_kappa_score(all_l, all_h)
        agr = float((np.array(all_h) == np.array(all_l)).mean())
        rows.append({"method": "ALL", "kappa": ka, "agreement": agr,
                     "n": len(all_h)})
    pd.DataFrame(rows).to_csv(os.path.join(AUDIT, "kappa_results.csv"), index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
