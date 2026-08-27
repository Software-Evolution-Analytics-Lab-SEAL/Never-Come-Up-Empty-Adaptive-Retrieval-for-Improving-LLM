"""
Concat the 12 OLD-unseen segment CSVs into one RQ2 test set.

Output: data/unseen_5510.csv with columns
    query_idx, Title, Accepted_Answer, Catalog
"""
import os, sys, glob
import pandas as pd
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config


def main():
    paths = sorted(glob.glob(os.path.join(config.SEGMENT_DIR, "segment_*.csv")),
                   key=lambda p: int(Path(p).stem.split("_")[1]))
    dfs = [pd.read_csv(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    df = df.rename(columns={"Accepted Answer Body": "Accepted_Answer"})
    df.insert(0, "query_idx", range(len(df)))
    print(f"Loaded {len(dfs)} segments -> {len(df)} rows")
    out = os.path.join(config.DATA_DIR, "unseen_5510.csv")
    df.to_csv(out, index=False)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
