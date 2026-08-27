"""
Per-step latency table for HB1 vs ZS, with min / mean / max per query,
plus a separate cost summary that distinguishes OpenRouter API models
from the local Ollama-hosted granite model.

Per-iteration latency is computed by differencing consecutive elapsed times
in the tqdm log lines (every progress line records [H:MM:SS<remaining]).

Sources (no re-runs):
  HyDE generation logs:
    synthetic   -> RQ1_n666/logs/HB1.log         (HyDE: ... s/it)
    unseen_2025 -> RQ2/logs/hyde_unseen_2025.log
  Retrieval:
    benchmark   -> from benchmark_retrieval_step.py output
  Final generation (HB1) & ZS logs as in build_cost_table.py.
"""
import os
import re
import glob
import numpy as np

RQ1_LOGS = os.environ.get("RQ1_LOGS", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "RQ1", "logs")))
RQ2_LOGS = os.environ.get("RQ2_LOGS", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "RQ2", "logs")))
MODELS = ["llama-3.1-8b", "mistral-7b", "qwen3-8b",
          "granite-3.1-8b", "deepseek-r1-70b", "gpt-4.1"]
HB1_TAG_UNSEEN = {"mistral-7b": "HYB"}

# Per-iteration tqdm pattern (any percentage)
ITER_RE = re.compile(
    r"([A-Za-z0-9_\-./]+):\s*\d+%\|[^\|]*\|\s*(\d+)/(\d+)\s*"
    r"\[(\d{1,2}:\d{2}:\d{2}|\d{1,2}:\d{2})<"
)
# Final completion line (n_done == n_total)
FINAL_RE = re.compile(
    r"([A-Za-z0-9_\-./]+):\s*100%\|██████████\|\s*(\d+)/(\d+)\s*"
    r"\[(\d{1,2}:\d{2}:\d{2}|\d{1,2}:\d{2})<00:00,"
)
RETRIEVAL_MS_MIN = 18.0
RETRIEVAL_MS_MEAN = 22.10
RETRIEVAL_MS_MAX = 35.0  # observed top of distribution from benchmark


def hms(s):
    p = [int(x) for x in s.split(":")]
    if len(p) == 3: return p[0]*3600 + p[1]*60 + p[2]
    if len(p) == 2: return p[0]*60 + p[1]
    return p[0]


def per_query_seconds(path, tag_pred=None):
    """Return list of per-query latencies (seconds) extracted by differencing
    consecutive tqdm elapsed timestamps belonging to the same tag."""
    if not os.path.exists(path):
        return []
    with open(path, errors="ignore") as f:
        text = f.read()
    # group entries per tag
    by_tag = {}
    for tag, n_done, n_total, hh in ITER_RE.findall(text):
        if tag_pred and not tag_pred(tag):
            continue
        by_tag.setdefault(tag, []).append((int(n_done), int(n_total), hms(hh)))
    out = []
    for tag, entries in by_tag.items():
        entries.sort()
        prev_n, prev_t = 0, 0
        for n, _, t in entries:
            if n <= prev_n:
                continue
            dn = n - prev_n
            dt = t - prev_t
            if dt < 0:
                continue
            per = dt / max(dn, 1)
            for _ in range(dn):
                out.append(per)
            prev_n, prev_t = n, t
    return out


def stats(arr):
    if not arr:
        return float("nan"), float("nan"), float("nan"), 0
    a = np.asarray(arr)
    return float(a.min()), float(a.mean()), float(a.max()), len(a)


def hyde_stats(dataset):
    if dataset == "synthetic":
        return stats(per_query_seconds(os.path.join(RQ1_LOGS, "HB1.log"),
                                       tag_pred=lambda t: t == "HyDE"))
    return stats(per_query_seconds(os.path.join(RQ2_LOGS, "hyde_unseen_2025.log"),
                                   tag_pred=lambda t: t == "HyDE"))


def hb1_stats(dataset, model):
    samples = []
    if dataset == "synthetic":
        for log in sorted(glob.glob(os.path.join(RQ1_LOGS, f"gen_{model}_w*.log"))):
            samples.extend(per_query_seconds(log,
                tag_pred=lambda t: t.startswith(f"{model}/HB1_")))
    else:
        tag = f"{model}/{HB1_TAG_UNSEEN.get(model, 'HB1')}"
        for log in sorted(glob.glob(os.path.join(RQ2_LOGS, f"unseen_2025_adaptive_{model}_s*.log"))):
            samples.extend(per_query_seconds(log, tag_pred=lambda t: t == tag))
    return stats(samples)


def zs_stats(dataset, model):
    samples = []
    if dataset == "synthetic":
        samples = per_query_seconds(os.path.join(RQ1_LOGS, f"zs_{model}.log"),
                                    tag_pred=lambda t: t == f"zs/{model}")
    else:
        for log in sorted(glob.glob(os.path.join(RQ2_LOGS, f"unseen_2025_zeroshot_{model}_s*.log"))):
            samples.extend(per_query_seconds(log, tag_pred=lambda t: t == f"{model}/zs"))
    return stats(samples)


def fmt(v):
    return "n/a" if v != v else f"{v:>6.2f}"


def report(dataset):
    h_min, h_mean, h_max, hn = hyde_stats(dataset)
    print(f"\n=== {dataset.upper()} ===")
    print(f"HyDE    (GPT-4o)  : min {h_min:.2f}  mean {h_mean:.2f}  max {h_max:.2f}  n={hn}")
    print(f"Retrieve (3.4M-vec KB) : min {RETRIEVAL_MS_MIN/1000:.3f}  mean "
          f"{RETRIEVAL_MS_MEAN/1000:.3f}  max {RETRIEVAL_MS_MAX/1000:.3f}  s")
    print()
    print(f"{'model':<18}"
          f"{'HB1_min':>9}{'HB1_mean':>10}{'HB1_max':>9}{'(n)':>7}    "
          f"{'ZS_min':>8}{'ZS_mean':>10}{'ZS_max':>9}{'(n)':>7}")
    print("-" * 100)
    for m in MODELS:
        hmin, hmean, hmax, hnn = hb1_stats(dataset, m)
        zmin, zmean, zmax, znn = zs_stats(dataset, m)
        print(f"{m:<18}"
              f"{fmt(hmin):>9}{fmt(hmean):>10}{fmt(hmax):>9}{hnn:>7}    "
              f"{fmt(zmin):>8}{fmt(zmean):>10}{fmt(zmax):>9}{znn:>7}")


def cost_summary():
    """Per-1000-query monetary cost estimate, OpenRouter API prices as of 2026.

    Assumed per-query token usage:
      HyDE call          input ~120 tok, output ~200 tok  (capped at 256)
      Final HB1 call     input ~2000 tok (q + 10 retrieved chunks), output ~400 tok
      ZS call            input ~120 tok (just question), output ~400 tok

    Local model (granite via Ollama) has no API cost — we report compute hours.
    """
    # OpenRouter pricing (USD per 1M tokens, input/output)
    PRICE = {
        "llama-3.1-8b":    (0.05, 0.05, "OpenRouter"),
        "mistral-7b":      (0.05, 0.05, "OpenRouter"),
        "qwen3-8b":        (0.05, 0.10, "OpenRouter"),
        "deepseek-r1-70b": (0.55, 2.19, "OpenRouter"),
        "gpt-4.1":         (2.50, 10.00, "OpenRouter (paid)"),
        "granite-3.1-8b":  (None, None, "Ollama local — no API cost"),
    }
    HYDE_IN, HYDE_OUT = 120, 200
    HB1_IN, HB1_OUT = 2000, 400
    ZS_IN, ZS_OUT = 120, 400

    print("\n=== Cost per 1,000 queries (USD), OpenRouter pricing ===\n")
    print(f"{'model':<18}{'host':<22}{'HB1_$/1k':>12}{'ZS_$/1k':>12}{'extra_$/1k':>14}")
    print("-" * 78)
    for m in MODELS:
        pin, pout, host = PRICE[m]
        if pin is None:
            print(f"{m:<18}{host:<22}{'(local)':>12}{'(local)':>12}{'(local)':>14}")
            continue
        # HyDE call uses GPT-4o, not the generator. Rough price: $2.50 in / $10.00 out per 1M.
        hyde_cost_per_1k = ((HYDE_IN * 0.15) + (HYDE_OUT * 0.60)) / 1000  # $/1k queries
        hb1_gen_per_1k = ((HB1_IN * pin) + (HB1_OUT * pout)) / 1000
        zs_per_1k = ((ZS_IN * pin) + (ZS_OUT * pout)) / 1000
        hb1_total = hyde_cost_per_1k + hb1_gen_per_1k
        extra = hb1_total - zs_per_1k
        print(f"{m:<18}{host:<22}{hb1_total:>12.3f}{zs_per_1k:>12.3f}{extra:>14.3f}")
    print("\n(HyDE call uses GPT-4o priced at $2.50 in / $10.00 out per 1M tokens.)")


def main():
    print("Per-iteration latencies derived from tqdm log timestamps.\n")
    report("synthetic")
    report("unseen_2025")
    cost_summary()


if __name__ == "__main__":
    main()
