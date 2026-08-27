"""
Sample 84 queries (95% conf, 10% margin from N=666) uniformly at random from
the Synthetic Question Set.

Output: ../data/sample_queries.csv — the 84 sampled rows + their test-set index.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RQ1 = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, RQ1)
import config

OUT = os.path.abspath(os.path.join(HERE, "..", "data", "sample_queries.csv"))
SEED = 42
N_SAMPLE = 84

df = pd.read_csv(config.TEST_SET_PATH).reset_index(drop=False).rename(columns={"index": "test_idx"})
print(f"Population: {len(df)}")
print(f"Sample:     {N_SAMPLE}")

rng = np.random.default_rng(SEED)
picked = np.sort(rng.choice(df.index.to_numpy(), size=N_SAMPLE, replace=False))
sample = df.loc[picked].copy()
sample.to_csv(OUT, index=False)
print(f"Saved {OUT}  n={len(sample)}")
