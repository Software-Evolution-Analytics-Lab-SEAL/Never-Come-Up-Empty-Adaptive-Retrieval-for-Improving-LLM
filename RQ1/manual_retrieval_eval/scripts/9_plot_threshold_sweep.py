"""
Two-panel figure: Usefulness and retrieval-volume vs. threshold for HB1 and HYB.

Panel A (top):  Usefulness with shaded 95% CI bands (left y-axis)
                + avg #units/query as dashed lines (right y-axis)
Panel B (bot):  Coverage (n_queries with retrievals) as bars

Output: ../data/fig_usefulness_vs_threshold.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
OUT = os.path.join(DATA, "fig_usefulness_vs_threshold.png")

COLOR = {"HB1": "#1f77b4", "HYB": "#d62728"}
MARKER = {"HB1": "o", "HYB": "s"}


def main():
    df = pd.read_csv(os.path.join(DATA, "threshold_sweep_summary.csv"))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7),
                                   gridspec_kw={"height_ratios": [3, 1]},
                                   sharex=False)

    # Panel A: Usefulness with CI bands
    for method in ["HB1", "HYB"]:
        sub = df[df["method"] == method].sort_values("threshold")
        x = sub["threshold"].to_numpy()
        y = sub["mean_usefulness"].to_numpy()
        lo = sub["ci_lo"].to_numpy()
        hi = sub["ci_hi"].to_numpy()
        c = COLOR[method]

        ax1.plot(x, y, marker=MARKER[method], color=c, lw=2.4, ms=8,
                 label=method, zorder=3)
        ax1.fill_between(x, lo, hi, color=c, alpha=0.18, zorder=1)

    ax1.set_ylabel("Usefulness  (per-unit binary judgment)",
                   fontsize=12, fontweight="bold")
    ax1.set_ylim(0.4, 1.05)
    ax1.grid(alpha=0.3)
    ax1.set_title("Usefulness vs. similarity threshold  (n=84 queries)",
                  fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlim(0.05, 0.95)
    ax1.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ax1.legend(loc="lower right", fontsize=11, framealpha=0.9)

    # Panel B: Coverage bars
    thrs = sorted(df["threshold"].unique())
    width = 0.35
    x_pos = np.arange(len(thrs))
    for offset, method in zip([-width/2, width/2], ["HB1", "HYB"]):
        cov = []
        for t in thrs:
            row = df[(df["method"] == method) & (df["threshold"] == t)]
            cov.append(int(row["n_queries"].iloc[0]) if len(row) else 0)
        ax2.bar(x_pos + offset, cov, width=width,
                color=COLOR[method], label=method, edgecolor="black", lw=0.5)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"{t:.1f}" for t in thrs])
    ax2.set_xlabel("Similarity threshold", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Coverage  (n_q)", fontsize=11, fontweight="bold")
    ax2.axhline(84, color="gray", lw=0.8, ls=":", label="full sample")
    ax2.set_ylim(0, 90)
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax2.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
