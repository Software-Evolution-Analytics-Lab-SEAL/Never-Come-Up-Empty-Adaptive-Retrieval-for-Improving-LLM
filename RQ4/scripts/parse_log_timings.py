"""
Extract per-query end-to-end timings from existing tqdm logs for HB1
(adaptive) and zero-shot, on both the synthetic n=666 dataset and the
unseen_2025 dataset (n=3,376).

Each tqdm progress line ends with: ... [TOTAL_TIME<00:00, RATE_s/it]

Synthetic logs:
  RQ1_n666/logs/zs_<model>.log                  ZS pipeline
  RQ1_n666/logs/gen_<model>_w*.log              HB1_0.5 pipeline (one of several thresholds in each log)

Unseen_2025 logs:
  RQ2/logs/unseen_2025_adaptive_<model>_s*.log  HB1 (adaptive) pipeline, sharded
  RQ2/logs/unseen_2025_zeroshot_<model>_s*.log  ZS pipeline, sharded

For each (dataset, pipeline, model) we sum total_seconds and total_iters
across all matching logs, then report total_seconds / total_iters.
"""
import os
import re
import glob
from collections import defaultdict

RQ1_LOGS = os.environ.get("RQ1_LOGS", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "RQ1", "logs")))
RQ2_LOGS = os.environ.get("RQ2_LOGS", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "RQ2", "logs")))

MODELS = ["llama-3.1-8b", "mistral-7b", "qwen3-8b",
          "granite-3.1-8b", "deepseek-r1-70b", "gpt-4.1"]

# Pattern matching tqdm 100% line. Groups: tag, n_done, n_total, hms, rate
TQDM_RE = re.compile(
    r"([A-Za-z0-9_\-./]+):\s*100%\|[^\|]*\|\s*(\d+)/(\d+)\s*"
    r"\[(\d{1,2}:\d{2}:\d{2}|\d{1,2}:\d{2})<\d{2}:\d{2},\s*([\d.]+)\s*s/it\]"
)


def hms_to_sec(hms):
    parts = [int(p) for p in hms.split(":")]
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m, s = 0, parts[0], parts[1]
    else:
        h, m, s = 0, 0, parts[0]
    return h * 3600 + m * 60 + s


def parse_runs(path, tag_filter=None):
    """Yield (tag, n_total, total_secs) for each 100% complete tqdm progress
    bar in the file. If tag_filter is supplied, only matching tags are returned."""
    try:
        with open(path, "r", errors="ignore") as f:
            content = f.read()
    except Exception:
        return
    for tag, n_done, n_total, hms, rate in TQDM_RE.findall(content):
        if n_done != n_total:
            continue
        if tag_filter and not tag_filter(tag):
            continue
        yield tag, int(n_total), hms_to_sec(hms)


def collect():
    rows = []

    # === Synthetic n=666 ===
    # ZS
    for m in MODELS:
        log = os.path.join(RQ1_LOGS, f"zs_{m}.log")
        if not os.path.exists(log):
            continue
        for tag, n, secs in parse_runs(log, tag_filter=lambda t: t.endswith(f"/{m.replace('.','.')}") or t == f"zs/{m}"):
            rows.append({"dataset": "synthetic", "pipeline": "ZS", "model": m,
                         "n": n, "secs": secs, "log": os.path.basename(log)})

    # HB1 — extract from gen_<model>_w*.log, only HB1_<thr> tags (any threshold)
    for m in MODELS:
        for log in sorted(glob.glob(os.path.join(RQ1_LOGS, f"gen_{m}_w*.log"))):
            for tag, n, secs in parse_runs(log, tag_filter=lambda t: t.startswith(f"{m}/HB1_")):
                rows.append({"dataset": "synthetic", "pipeline": "HB1", "model": m,
                             "n": n, "secs": secs, "log": os.path.basename(log), "tag": tag})

    # === Unseen 2025 ===
    # For mistral-7b the optimal pipeline tag is HYB (per OPTIMAL map); other models use HB1
    HB1_TAG = {"mistral-7b": "HYB"}
    for m in MODELS:
        hb1_label = HB1_TAG.get(m, "HB1")
        # HB1 adaptive — only tags ending in /HB1 or /HYB
        for log in sorted(glob.glob(os.path.join(RQ2_LOGS, f"unseen_2025_adaptive_{m}_s*.log"))):
            for tag, n, secs in parse_runs(log, tag_filter=lambda t, h=hb1_label, mm=m: t == f"{mm}/{h}"):
                rows.append({"dataset": "unseen_2025", "pipeline": "HB1", "model": m,
                             "n": n, "secs": secs, "log": os.path.basename(log)})
        # ZS — only tags ending in /zs
        for log in sorted(glob.glob(os.path.join(RQ2_LOGS, f"unseen_2025_zeroshot_{m}_s*.log"))):
            for tag, n, secs in parse_runs(log, tag_filter=lambda t, mm=m: t == f"{mm}/zs"):
                rows.append({"dataset": "unseen_2025", "pipeline": "ZS", "model": m,
                             "n": n, "secs": secs, "log": os.path.basename(log)})
    return rows


def main():
    rows = collect()
    by_key = defaultdict(lambda: {"n": 0, "secs": 0, "logs": 0})
    for r in rows:
        k = (r["dataset"], r["pipeline"], r["model"])
        by_key[k]["n"] += r["n"]
        by_key[k]["secs"] += r["secs"]
        by_key[k]["logs"] += 1

    print(f"=== Per (dataset, pipeline, model) — total time / total iterations ===\n")
    print(f"{'dataset':<13}{'pipeline':<5}  {'model':<22}{'logs':>5}{'iters':>8}{'sec/query':>11}")
    print("-" * 70)
    keys = sorted(by_key.keys())
    for k in keys:
        ds, pipe, m = k
        v = by_key[k]
        sec_per = v["secs"] / max(v["n"], 1)
        print(f"{ds:<13}{pipe:<5}  {m:<22}{v['logs']:>5}{v['n']:>8}{sec_per:>11.2f}")

    print("\n=== HB1 vs ZS per-query latency (sec) ===\n")
    pivot = {}
    for (ds, pipe, m), v in by_key.items():
        sec_per = v["secs"] / max(v["n"], 1)
        pivot.setdefault((ds, m), {})[pipe] = sec_per
    print(f"{'dataset':<13}{'model':<22}{'HB1 sec/q':>11}{'ZS sec/q':>11}{'Δ overhead':>12}{'HB1/ZS':>8}")
    print("-" * 78)
    for (ds, m) in sorted(pivot.keys()):
        d = pivot[(ds, m)]
        hb1 = d.get("HB1", float("nan"))
        zs = d.get("ZS", float("nan"))
        if hb1 == hb1 and zs == zs:
            print(f"{ds:<13}{m:<22}{hb1:>11.2f}{zs:>11.2f}{(hb1-zs):>+12.2f}{(hb1/zs):>8.2f}x")
        else:
            print(f"{ds:<13}{m:<22}{hb1 if hb1==hb1 else 'n/a':>11}"
                  f"{zs if zs==zs else 'n/a':>11}    n/a       n/a")


if __name__ == "__main__":
    main()
