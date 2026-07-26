"""End-to-end: cohort -> risk model -> alert-policy Pareto frontier -> bandit ->
robustness -> plots. Run after (or alongside) the cached download.

  python run_pipeline.py --limit 300      # fast smoke run
  python run_pipeline.py                   # full cached cohort
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import bandit  # noqa: E402
import risk_model  # noqa: E402
import robustness  # noqa: E402
import vitaldb_load as V  # noqa: E402
from frontier import (bootstrap_policy, evaluate_grid, not_an_roc_pairs,  # noqa: E402
                      pareto_indices)
from labels import hypotension_onsets  # noqa: E402
from metrics import CaseEval  # noqa: E402
from policies import naive_threshold_policy, policy_grid  # noqa: E402

OBJ = ["sensitivity", "warning_time", "false_rate", "burden", "disparity"]
RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)


def age_band(a):
    return "<50" if a < 50 else ("50-65" if a < 65 else "65+")


def build_cases(meta, risk_by_case, thr=None):
    cases, frames = [], []
    for row in meta.itertuples(index=False):
        cid = int(row.caseid)
        df = V.load_cached_case(cid)
        sg = {"asa": str(row.asa), "age_band": age_band(row.age), "sex": row.sex}
        ce = CaseEval.from_frame(df, risk_by_case[cid], sg)
        if thr is not None:
            ce.onsets = hypotension_onsets(df["map"].to_numpy(float), thr=thr)
        cases.append(ce)
        frames.append(df)
    return cases, frames


def make_grid():
    return policy_grid(
        taus=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        ms=[1, 2, 3],
        cs=[0, 3, 5, 10],
        trends=[None, 0.02],
    )


def plot_frontier(rows, pareto, path):
    sens = [r["sensitivity"] for r in rows]
    burd = [r["burden"] for r in rows]
    warn = [r["warning_time"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sc = ax.scatter(burd, sens, c=warn, cmap="viridis", s=18, alpha=0.5)
    pb = [rows[i]["burden"] for i in pareto]
    ps = [rows[i]["sensitivity"] for i in pareto]
    order = np.argsort(pb)
    ax.plot(np.array(pb)[order], np.array(ps)[order], "r.-", lw=1.2, label="Pareto")
    ax.set_xlabel("alert burden (alerts / case-hour)")
    ax.set_ylabel("event sensitivity")
    fig.colorbar(sc, label="median warning time (min)")
    ax.legend()
    ax.set_title("Alert-policy frontier (color = warning time -> not an ROC)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--boot", type=int, default=200)
    args = ap.parse_args()

    print("[1] cohort cache")
    meta = V.build_cache(limit=args.limit)          # resumable; uses existing cache

    print("[2] risk model (leakage check)")
    models, aucs, roc_data = risk_model.train(meta)
    print("    test AUC  trend=%.3f  leak=%.3f  naive_currentMAP=%.3f  (event rate %.3f)"
          % (aucs["trend"], aucs["leak"], aucs["naive_currentMAP"], aucs["event_rate"]))
    pd.Series(aucs).to_csv(os.path.join(RESULTS, "leakage.csv"))
    # ROC of the risk score (classifier) vs the pre-event label, on held-out patients
    from sklearn.metrics import roc_curve
    roc_rows = []
    for name in ("trend", "leak", "naive"):
        fpr, tpr, _ = roc_curve(roc_data["y"], roc_data[name])
        roc_rows += [{"model": name, "fpr": f, "tpr": t} for f, t in zip(fpr, tpr)]
    pd.DataFrame(roc_rows).to_csv(os.path.join(RESULTS, "roc_points.csv"), index=False)

    print("[3] risk streams + cases")
    risk_by_case = risk_model.risk_streams(models["trend"], meta)
    cases, frames = build_cases(meta, risk_by_case)

    print("[4] policy grid + Pareto")
    grid = make_grid()
    rows = evaluate_grid(cases, grid, OBJ)
    # clinically-honest strawman: alert on current MAP only (near-zero lead time)
    map_risk = {int(r.caseid):
                np.clip((70 - V.load_cached_case(int(r.caseid))["map"].to_numpy(float))
                        / 20.0, 0, 1)
                for r in meta.itertuples(index=False)}
    cm_cases, _ = build_cases(meta, map_risk)
    from metrics import objectives as _obj
    cm = dict(_obj(cm_cases, naive_threshold_policy(0.5)))
    cm.update(tau=0.5, m=1, C=0, trend=None, method="current_map")
    rows.append(cm)
    pareto = pareto_indices(rows, OBJ)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "frontier.csv"), index=False)
    plot_frontier(rows, pareto, os.path.join(RESULTS, "frontier.png"))
    witness = not_an_roc_pairs(rows)
    print("    grid=%d  pareto=%d  not-an-ROC witness pairs=%d"
          % (len(grid), len(pareto), len(witness)))

    print("[5] bootstrap CIs on Pareto policies")
    ci = {}
    for i in pareto[:6]:
        p = grid[i] if i < len(grid) else naive_threshold_policy(0.5)
        ci[i] = bootstrap_policy(cases, p, OBJ, n_boot=args.boot)
    pd.DataFrame({k: {f"{o}_mean": v[o][0] for o in OBJ} for k, v in ci.items()}).T \
        .to_csv(os.path.join(RESULTS, "pareto_ci.csv"))

    print("[6] bandit (offline policy, patient-level train/test, exact replay)")
    risk_list = [risk_by_case[int(c)] for c in meta.caseid]
    idx = np.arange(len(cases))
    np.random.default_rng(0).shuffle(idx)
    cut = int(0.7 * len(idx))
    tr_i, te_i = idx[:cut], idx[cut:]
    clf = bandit.train_bandit([frames[i] for i in tr_i], [risk_list[i] for i in tr_i])
    brows = bandit.sweep([cases[i] for i in te_i], clf,
                         [frames[i] for i in te_i], [risk_list[i] for i in te_i],
                         biases=np.linspace(-2, 2, 9), C=3)
    pd.DataFrame(brows).to_csv(os.path.join(RESULTS, "bandit_sweep.csv"), index=False)

    print("[7] robustness")
    sub = robustness.subgroup_frontiers(cases, grid, OBJ, "asa")
    pd.DataFrame({v: {"n": d["n"], "n_pareto": len(d["pareto"])}
                  for v, d in sub.items()}).T.to_csv(
        os.path.join(RESULTS, "subgroup_asa.csv"))

    best = grid[pareto[0]] if pareto and pareto[0] < len(grid) else naive_threshold_policy(0.4)
    ts = robustness.threshold_sensitivity(
        lambda thr: build_cases(meta, risk_by_case, thr=thr)[0], best, [60, 65, 70])
    pd.DataFrame(ts).T.to_csv(os.path.join(RESULTS, "threshold_sensitivity.csv"))

    print("[done] results in", RESULTS)


def objectives_naive(cases):
    from metrics import objectives
    o = dict(objectives(cases, naive_threshold_policy(0.4)))
    o.update(tau=0.4, m=1, C=0, trend=None, method="naive")
    return o


if __name__ == "__main__":
    main()
