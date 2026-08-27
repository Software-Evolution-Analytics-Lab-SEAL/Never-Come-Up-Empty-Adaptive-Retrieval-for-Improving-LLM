"""
Cohen's kappa between annotator2 and annotator3 across all four manual-
evaluation files in Manual_Evaluation_Results/.

Reports per-file kappa + agreement, plus per-tab breakdown where relevant.

File 1, 2, 3: binary True/False labels for usefulness.
File 4: hallucination judge labels — annotator2/3 also use True/False (True
= hallucinated, False = not hallucinated, or vice versa per the annotation
guideline; we treat the labels as categorical and let kappa be invariant).
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

RESULTS = os.environ.get("MANUAL_EVAL_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "manual_evaluation")))
OUT = os.path.join(RESULTS, "annotator_agreement_summary.csv")

FILES = [
    ("01", "01_usefulness_pipelines.xlsx",      "method",    "pipeline"),
    ("02", "02_usefulness_thresholds.xlsx",     "threshold", "threshold"),
    ("03", "03_final_answer_usefulness.xlsx",   "model",     "model"),
    ("04", "04_hallucination.xlsx",          None,        "alignment"),
]


def norm(s):
    """Normalize True/False strings to canonical form."""
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    if s in ("true", "yes", "y", "1", "t"):
        return "True"
    if s in ("false", "no", "n", "0", "f"):
        return "False"
    return s


def kappa_for(df, tag):
    a2 = df["annotator2_label"].apply(norm)
    a3 = df["annotator3_label"].apply(norm)
    mask = (a2 != "") & (a3 != "")
    a2 = a2[mask].tolist()
    a3 = a3[mask].tolist()
    if len(a2) < 5:
        return None
    k = cohen_kappa_score(a2, a3)
    agree = float(np.mean([x == y for x, y in zip(a2, a3)]))
    labels = sorted(set(a2) | set(a3))
    cm = confusion_matrix(a2, a3, labels=labels)
    cm_dict = {f"a2={la2},a3={la3}": int(cm[i, j])
               for i, la2 in enumerate(labels) for j, la3 in enumerate(labels)}
    return {"tag": tag, "n": len(a2), "kappa": k, "agreement": agree,
            "labels": labels, **cm_dict}


def label_kappa(k):
    if k is None or np.isnan(k): return "-"
    if k < 0.20: return "poor"
    if k < 0.40: return "fair"
    if k < 0.60: return "moderate"
    if k < 0.80: return "substantial"
    return "almost perfect"


def main():
    summary_rows = []
    for label, fname, sub_col, sub_label in FILES:
        path = os.path.join(RESULTS, fname)
        if not os.path.exists(path):
            print(f"[{label}] missing: {fname}"); continue
        print(f"\n=== File {label} — {fname} ===")
        xls = pd.ExcelFile(path)
        all_a2, all_a3 = [], []
        for sheet in xls.sheet_names:
            df = pd.read_excel(path, sheet_name=sheet)
            res = kappa_for(df, sheet)
            if res is None:
                print(f"  [{sheet}] insufficient data")
                continue
            a2 = df["annotator2_label"].apply(norm)
            a3 = df["annotator3_label"].apply(norm)
            mask = (a2 != "") & (a3 != "")
            all_a2.extend(a2[mask].tolist())
            all_a3.extend(a3[mask].tolist())
            print(f"  {sheet:<22}  n={res['n']:>4}  κ={res['kappa']:>+0.3f} "
                  f"({label_kappa(res['kappa']):<14})  agreement={res['agreement']*100:.1f}%")
            summary_rows.append({"file": label, sub_label or "sheet": sheet,
                                 "n": res["n"], "kappa": res["kappa"],
                                 "agreement": res["agreement"],
                                 "interpretation": label_kappa(res["kappa"])})
        # File-level overall
        if all_a2:
            k_all = cohen_kappa_score(all_a2, all_a3)
            agree_all = float(np.mean([x == y for x, y in zip(all_a2, all_a3)]))
            print(f"  {'ALL (file)':<22}  n={len(all_a2):>4}  κ={k_all:>+0.3f} "
                  f"({label_kappa(k_all):<14})  agreement={agree_all*100:.1f}%")
            summary_rows.append({"file": label, sub_label or "sheet": "ALL",
                                 "n": len(all_a2), "kappa": k_all,
                                 "agreement": agree_all,
                                 "interpretation": label_kappa(k_all)})

    pd.DataFrame(summary_rows).to_csv(OUT, index=False)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
