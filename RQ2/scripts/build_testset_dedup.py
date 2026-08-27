"""
Build the deduplicated unseen dataset.

Pipeline:
  1. Load the 12 segment CSVs from RQ1_RQ2/complete_testing_data -> 5510 queries.
  2. Embed each query Title with all-mpnet-base-v2.
  3. Compute max-cosine-similarity vs KB question embeddings (N=3.4M).
  4. Print survivor counts for each tau in [0.1, ..., 0.9].
  5. Save:
       data/unseen_raw_5510.csv           (full concatenated segments)
       data/unseen_kb_similarity.csv      (adds max_sim + nearest_kb_idx columns)

Usage:
    python3 build_unseen_dedup.py                        # runs full pipeline
    python3 build_unseen_dedup.py --report-only          # skips recomputing if the CSV exists

After you pick a threshold tau, run:
    python3 build_unseen_dedup.py --write-dedup 0.85     # writes unseen_dedup_tau0.85.csv
"""
import os
import sys
import glob
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config
import kb  # type: ignore

from sentence_transformers import SentenceTransformer, util


def concat_segments() -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(config.SEGMENT_DIR, "segment_*.csv")),
                   key=lambda p: int(Path(p).stem.split("_")[1]))
    dfs = [pd.read_csv(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(dfs)} segments -> {len(df)} queries")
    return df


def load_input(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {path} -> {len(df)} queries  (columns={list(df.columns)})")
    return df


def embed_queries(titles: list[str], device: str, batch: int = 128) -> torch.Tensor:
    print(f"Encoding {len(titles)} query titles on {device} ...")
    enc = SentenceTransformer(config.EMBED_MODEL, device=device)
    emb = enc.encode(titles, batch_size=batch, convert_to_tensor=True,
                     show_progress_bar=True, normalize_embeddings=False)
    return emb.to(device)


def kb_max_sim(query_emb: torch.Tensor, kb_emb: torch.Tensor, device: str,
               chunk: int = 200_000):
    """Compute max cosine similarity of each query against KB, in KB chunks.

    Returns (max_sim[N_q], argmax_kb[N_q]).
    """
    # Make sure both are normalized for cosine
    qn = torch.nn.functional.normalize(query_emb.to(torch.float32), dim=-1).to(device)
    N = kb_emb.shape[0]
    max_sim = torch.full((query_emb.shape[0],), -1.0, device=device, dtype=torch.float32)
    argmax = torch.zeros(query_emb.shape[0], dtype=torch.int64, device=device)
    for start in tqdm(range(0, N, chunk), desc="KB chunks"):
        end = min(start + chunk, N)
        kc = torch.nn.functional.normalize(kb_emb[start:end].to(torch.float32), dim=-1).to(device)
        # sims shape (Nq, chunk)
        sims = qn @ kc.T
        batch_max, batch_arg = sims.max(dim=1)
        mask = batch_max > max_sim
        max_sim = torch.where(mask, batch_max, max_sim)
        argmax = torch.where(mask, batch_arg + start, argmax)
        del sims, kc
    return max_sim.cpu().numpy(), argmax.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None,
                    help="Input CSV with 'Title' (+ optional 'Catalog'). Default: concat segment_*.csv.")
    ap.add_argument("--output-sim", default=None,
                    help="Where to write the per-query similarity CSV (default config.UNSEEN_SIM_CSV)")
    ap.add_argument("--report-only", action="store_true",
                    help="Skip recompute; just print the survivor report from existing CSV")
    ap.add_argument("--write-dedup", type=float, default=None,
                    help="If given, write data/unseen_dedup_tauXX.csv (queries with max_sim < tau).")
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()
    out_sim = args.output_sim or config.UNSEEN_SIM_CSV

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(config.DATA_DIR, exist_ok=True)

    if args.report_only and os.path.exists(out_sim):
        df = pd.read_csv(out_sim)
        print(f"Loaded {out_sim} ({len(df)} rows)")
    else:
        raw = load_input(args.input) if args.input else concat_segments()

        # Drop any exact-title duplicates within the unseen set itself
        before = len(raw)
        raw = raw.drop_duplicates(subset=["Title"]).reset_index(drop=True)
        print(f"Dropped {before - len(raw)} within-dataset exact-title dupes -> {len(raw)}")

        titles = raw["Title"].astype(str).tolist()
        q_emb = embed_queries(titles, device=device)

        kb_emb = kb.question_embeddings_matrix()
        print(f"KB question embedding matrix: {tuple(kb_emb.shape)}")

        max_sim, argmax = kb_max_sim(q_emb, kb_emb, device=device)

        df = raw.copy()
        df["max_sim_to_kb"] = max_sim
        df["nearest_kb_idx"] = argmax
        df.to_csv(out_sim, index=False)
        print(f"Saved {out_sim}")

    # Survivor report
    taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
    print("\n=== Survivor counts (keep rows with max_sim_to_kb < tau) ===")
    print(f"{'tau':<6} {'#kept':<8} {'% kept':<8} {'#java':<7} {'#python':<8} {'balanced-per-cat':<18}")
    for tau in taus:
        sub = df[df["max_sim_to_kb"] < tau]
        nj = (sub["Catalog"] == "JAVA").sum() if "Catalog" in sub.columns else 0
        np_ = (sub["Catalog"] == "PYTHON").sum() if "Catalog" in sub.columns else 0
        bal = min(nj, np_)
        print(f"{tau:<6.2f} {len(sub):<8d} {100*len(sub)/len(df):<7.1f}% {nj:<7d} {np_:<8d} {bal*2:<18d}")

    # Distribution
    print("\n=== max_sim_to_kb distribution ===")
    for q in [0.5, 0.75, 0.9, 0.95, 0.99, 1.0]:
        print(f"  p{int(q*100):>3}%   = {np.quantile(df['max_sim_to_kb'], q):.4f}")
    print(f"  mean    = {df['max_sim_to_kb'].mean():.4f}")
    print(f"  n (max_sim >= 0.99, near-exact) = {(df['max_sim_to_kb'] >= 0.99).sum()}")

    if args.write_dedup is not None:
        tau = args.write_dedup
        out = df[df["max_sim_to_kb"] < tau].reset_index(drop=True)
        path = config.UNSEEN_DEDUP_CSV_TPL.format(tau=f"{tau:.2f}".rstrip("0").rstrip("."))
        out.to_csv(path, index=False)
        print(f"\nWrote {path}  ({len(out)} rows, tau<{tau})")


if __name__ == "__main__":
    main()
