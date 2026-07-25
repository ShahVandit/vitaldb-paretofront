"""Portfolio figures from the result CSVs (decoupled from the heavy pipeline).

Colorblind-safe (Okabe-Ito) palette, one axis per chart, recessive grid, thin marks,
direct labels for the story points. Run after run_pipeline.py.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = os.path.join(os.path.dirname(__file__), "results")

# Okabe-Ito
GRAY, BLUE, ORANGE, GREEN = "#999999", "#0072B2", "#E69F00", "#009E73"

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 10, "axes.grid": True,
    "grid.color": "#E6E6E6", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#666666",
})


def _pareto_idx(rows, sens, burd):
    # non-dominated for higher-sensitivity / lower-burden
    keep = []
    for i in range(len(rows)):
        if not any(j != i and sens[j] >= sens[i] and burd[j] <= burd[i]
                   and (sens[j] > sens[i] or burd[j] < burd[i]) for j in range(len(rows))):
            keep.append(i)
    return keep


def fig_frontier(f):
    sens, burd = f["sensitivity"].to_numpy(), f["burden"].to_numpy()
    p = _pareto_idx(f, sens, burd)
    order = np.argsort(burd[p])
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.scatter(burd, sens, s=16, color=GRAY, alpha=0.45, label="candidate policies",
               edgecolor="none")
    ax.plot(burd[p][order], sens[p][order], "-o", color=BLUE, ms=5, lw=1.8,
            label="Pareto frontier")
    if "method" in f.columns and (f["method"] == "current_map").any():
        cm = f[f["method"] == "current_map"].iloc[0]
        ax.scatter([cm.burden], [cm.sensitivity], marker="D", s=70, color=ORANGE,
                   zorder=5, label="current-MAP rule")
        ax.annotate("current-MAP rule\n(low warning time)",
                    (cm.burden, cm.sensitivity), textcoords="offset points",
                    xytext=(10, -6), color=ORANGE, fontsize=8.5)
    ax.set_xlabel("alert burden  (alerts / case-hour)")
    ax.set_ylabel("event sensitivity")
    ax.set_title("Alert-policy trade-off: sensitivity vs clinician burden")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(R, "fig_frontier.png")); plt.close(fig)


def fig_not_an_roc(f):
    """Warning time vs burden at ~matched sensitivity: the ROC-invisible axis."""
    band = f[(f["sensitivity"] >= 0.9)].copy()
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    sc = ax.scatter(band["burden"], band["warning_time"], c=band["false_rate"],
                    cmap="viridis", s=26, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("alert burden  (alerts / case-hour)")
    ax.set_ylabel("median warning time  (min)")
    ax.set_title("Policies at ≥ 0.90 sensitivity differ on burden & warning time\n"
                 "(same ROC point, different temporal behavior — not an ROC)")
    fig.colorbar(sc, label="false-alert rate / case-hour")
    fig.tight_layout(); fig.savefig(os.path.join(R, "fig_not_an_roc.png")); plt.close(fig)


def fig_bandit(f, b):
    sens, burd = f["sensitivity"].to_numpy(), f["burden"].to_numpy()
    p = _pareto_idx(f, sens, burd); order = np.argsort(burd[p])
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(burd[p][order], sens[p][order], "-o", color=BLUE, ms=4, lw=1.6,
            label="grid Pareto frontier")
    bb = b.sort_values("burden")
    ax.plot(bb["burden"], bb["sensitivity"], "-s", color=GREEN, ms=5, lw=1.8,
            label="learned bandit policy")
    ax.set_xlabel("alert burden  (alerts / case-hour)")
    ax.set_ylabel("event sensitivity")
    ax.set_title("Learned offline bandit policy vs the interpretable frontier")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(R, "fig_bandit.png")); plt.close(fig)


def fig_leakage(lk):
    order = ["trend", "naive_currentMAP", "leak"]
    labels = ["trend only\n(true early warning)", "current-MAP\nrule", "trend + current MAP\n(leaky)"]
    vals = [float(lk[k]) for k in order]
    colors = [GREEN, ORANGE, GRAY]
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    ax.bar(labels, vals, color=colors, width=0.62)
    ax.axhline(0.5, color="#444444", lw=1, ls="--")
    ax.text(2.35, 0.51, "chance", fontsize=8, color="#444444", ha="right")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylim(0.4, max(0.85, max(vals) + 0.05))
    ax.set_ylabel("test AUC")
    ax.set_title("The leakage trap: apparent 'prediction' is mostly current MAP")
    fig.tight_layout(); fig.savefig(os.path.join(R, "fig_leakage.png")); plt.close(fig)


def main():
    f = pd.read_csv(os.path.join(R, "frontier.csv"))
    fig_frontier(f)
    fig_not_an_roc(f)
    if os.path.exists(os.path.join(R, "bandit_sweep.csv")):
        fig_bandit(f, pd.read_csv(os.path.join(R, "bandit_sweep.csv")))
    lk_p = os.path.join(R, "leakage.csv")
    if os.path.exists(lk_p):
        lk = pd.read_csv(lk_p, index_col=0).iloc[:, 0]
        fig_leakage(lk)
    print("[plots] written to", R)


if __name__ == "__main__":
    main()
