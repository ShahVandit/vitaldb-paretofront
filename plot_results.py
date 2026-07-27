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


def _split(f):
    meth = f["method"].astype(str) if "method" in f.columns else pd.Series("", index=f.index)
    is_base = meth.str.startswith("MAP<")
    return f[~is_base].reset_index(drop=True), f[is_base].sort_values("burden")


def fig_frontier(f, false_cap=2.0, ppv_min=0.40, gap_max=0.10):
    """Joint constrained frontier: Pareto over feasible MODEL + BASELINE policies."""
    model, base = _split(f)
    feas = f[(f["false_rate"] <= false_cap) & (f["ppv"] >= ppv_min)
             & (f["disparity"] <= gap_max)].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.scatter(model["burden"], model["utility"], s=16, color=GRAY, alpha=0.35,
               edgecolor="none", label="model policies")
    if len(base):
        ax.scatter(base["burden"], base["utility"], marker="D", s=40, color=ORANGE,
                   label="MAP baselines")
    if len(feas):
        u, b = feas["utility"].to_numpy(), feas["burden"].to_numpy()
        p = _pareto_idx(feas, u, b)
        order = np.argsort(b[p])
        ax.plot(b[p][order], u[p][order], "-o", color=BLUE, ms=5, lw=1.8, zorder=5,
                label=f"feasible frontier (false≤{false_cap:g}/hr, PPV≥{ppv_min:g})")
    ax.set_xlabel("alert burden  (alerts / case-hour)")
    ax.set_ylabel("clinical utility  (detection × timeliness)")
    ax.set_title("Constrained utility-vs-burden frontier (model + MAP baselines)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(R, "fig_frontier.png")); plt.close(fig)


def fig_not_an_roc(f):
    """Warning time vs burden at ~matched sensitivity: the ROC-invisible axis."""
    model, _ = _split(f)
    band = model[(model["sensitivity"] >= 0.9)].copy()
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    sc = ax.scatter(band["burden"], band["warning_time"], c=band["false_rate"],
                    cmap="viridis", s=26, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("alert burden  (alerts / case-hour)")
    ax.set_ylabel("median warning time  (min)")
    ax.set_title("Policies at ≥ 0.90 sensitivity differ on burden & warning time\n"
                 "(same ROC point, different temporal behavior — not an ROC)")
    fig.colorbar(sc, label="false-alert rate / case-hour")
    fig.tight_layout(); fig.savefig(os.path.join(R, "fig_not_an_roc.png")); plt.close(fig)


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


def fig_roc(roc):
    """ROC of the risk *score* (classifier) — contrast with the policy frontier."""
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    colors = {"trend": GREEN, "naive": ORANGE, "leak": GRAY}
    names = {"trend": "trend model", "naive": "current-MAP rule", "leak": "with current MAP (leaky)"}
    for name in ["trend", "naive", "leak"]:
        d = roc[roc["model"] == name].sort_values("fpr")
        if len(d):
            auc = np.trapz(d["tpr"], d["fpr"])
            ax.plot(d["fpr"], d["tpr"], color=colors[name], lw=1.8,
                    label=f"{names[name]} (AUC {auc:.2f})")
    ax.plot([0, 1], [0, 1], ls="--", color="#444444", lw=1, label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate (sensitivity)")
    ax.set_title("ROC of the risk score\n(a classifier lives here; the alerting policy does not)")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(R, "fig_roc.png")); plt.close(fig)


def main():
    f = pd.read_csv(os.path.join(R, "frontier.csv"))
    fig_frontier(f)
    fig_not_an_roc(f)
    roc_p = os.path.join(R, "roc_points.csv")
    if os.path.exists(roc_p):
        fig_roc(pd.read_csv(roc_p))
    lk_p = os.path.join(R, "leakage.csv")
    if os.path.exists(lk_p):
        lk = pd.read_csv(lk_p, index_col=0).iloc[:, 0]
        fig_leakage(lk)
    print("[plots] written to", R)


if __name__ == "__main__":
    main()
