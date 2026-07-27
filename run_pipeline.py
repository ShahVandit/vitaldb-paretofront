"""End-to-end: cohort -> risk model -> constrained utility/burden frontier ->
robustness. (Part A: 2-objective constrained frontier; the GBM risk stays here.
Part B swaps in the multimodal model + subject-level, test-only evaluation.)

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

import risk_model  # noqa: E402
import robustness  # noqa: E402
import vitaldb_load as V  # noqa: E402
from frontier import bootstrap_policy, constrained_pareto, evaluate_grid  # noqa: E402
from labels import exclude_mask, hypotension_episodes  # noqa: E402
from metrics import CaseEval  # noqa: E402
from metrics import objectives as obj_of  # noqa: E402
from policies import Policy, policy_grid  # noqa: E402

OBJ = ["utility", "burden"]                 # the two Pareto axes (review C5)
FALSE_CAPS = [1.0, 2.0, 4.0]               # constrained-frontier sweep (alerts/hr)
MAP_THRESHOLDS = [65, 70, 72, 75]          # literature-standard baseline (MAP<72 RCT)
RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)


def age_band(a):
    return "<50" if a < 50 else ("50-65" if a < 65 else "65+")


def _subgroup(row):
    return {"asa": str(row.asa), "age_band": age_band(row.age), "sex": row.sex}


def build_cases(meta, risk_by_case, thr=None):
    cases, frames = [], []
    for row in meta.itertuples(index=False):
        cid = int(row.caseid)
        df = V.load_cached_case(cid)
        sg = _subgroup(row)
        if thr is None:
            ce = CaseEval.from_frame(df, risk_by_case[cid], sg)
        else:                                # threshold-sensitivity: re-label at thr
            m = df["map"].to_numpy(float)
            eps = hypotension_episodes(m, thr=thr)
            ce = CaseEval(minute=np.arange(len(m)), risk=np.asarray(risk_by_case[cid],
                          float), episodes=eps, hours=len(m) / 60.0,
                          exclude=exclude_mask(len(m), eps), subgroup=sg)
        cases.append(ce)
        frames.append(df)
    return cases, frames


def map_baseline_rows(meta):
    """MAP<T alarm family (T in 65..75, cooldown 0/5) as its own set of policies."""
    rows = []
    prel = []
    for r in meta.itertuples(index=False):
        m = V.load_cached_case(int(r.caseid))["map"].to_numpy(float)
        eps = hypotension_episodes(m)
        prel.append((m, _subgroup(r), eps, exclude_mask(len(m), eps)))
    for T in MAP_THRESHOLDS:
        for C in (0, 5):
            cs = [CaseEval(minute=np.arange(len(m)), risk=(m < T).astype(float),
                           episodes=eps, hours=len(m) / 60.0, exclude=ex, subgroup=sg)
                  for m, sg, eps, ex in prel]
            o = dict(obj_of(cs, Policy(tau=0.5, m=1, C=C)))
            o.update(tau=0.5, m=1, C=C, trend=None, method=f"MAP<{T}")
            rows.append(o)
    return rows


def make_grid():
    return policy_grid(taus=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                       ms=[1, 2, 3], cs=[0, 3, 5, 10], trends=[None, 0.02])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--boot", type=int, default=200)
    args = ap.parse_args()

    print("[1] cohort cache")
    meta = V.build_cache(limit=args.limit)

    print("[2] risk model (GBM leakage check)")
    models, aucs, roc_data = risk_model.train(meta)
    print("    test AUC  trend=%.3f  leak=%.3f  naive_currentMAP=%.3f  (event rate %.3f)"
          % (aucs["trend"], aucs["leak"], aucs["naive_currentMAP"], aucs["event_rate"]))
    pd.Series(aucs).to_csv(os.path.join(RESULTS, "leakage.csv"))
    from sklearn.metrics import roc_curve
    roc_rows = []
    for name in ("trend", "leak", "naive"):
        fpr, tpr, _ = roc_curve(roc_data["y"], roc_data[name])
        roc_rows += [{"model": name, "fpr": f, "tpr": t} for f, t in zip(fpr, tpr)]
    pd.DataFrame(roc_rows).to_csv(os.path.join(RESULTS, "roc_points.csv"), index=False)

    print("[3] risk streams + cases")
    risk_by_case = risk_model.risk_streams(models["trend"], meta)
    cases, _ = build_cases(meta, risk_by_case)

    print("[4] constrained utility/burden frontier")
    grid = make_grid()
    rows = evaluate_grid(cases, grid, OBJ)
    rows += map_baseline_rows(meta)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "frontier.csv"), index=False)
    headline = None
    for cap in FALSE_CAPS:
        nd, feas = constrained_pareto(rows, OBJ, false_cap=cap)
        print("    false<=%.0f/hr:  feasible=%3d  pareto=%2d" % (cap, len(feas), len(nd)))
        if cap == 2.0:
            headline = nd
    pd.Series({"false_cap": 2.0, "pareto_idx": headline}).to_json(
        os.path.join(RESULTS, "headline_frontier.json"))

    print("[5] bootstrap CIs on headline (false<=2/hr) frontier")
    ci = {}
    for i in (headline or [])[:6]:
        if i < len(grid):
            ci[i] = bootstrap_policy(cases, grid[i], OBJ, n_boot=args.boot)
    if ci:
        pd.DataFrame({k: {f"{o}_mean": v[o][0] for o in OBJ}
                      for k, v in ci.items()}).T.to_csv(
            os.path.join(RESULTS, "pareto_ci.csv"))

    print("[6] robustness")
    sub = robustness.subgroup_frontiers(cases, grid, OBJ, "asa")
    pd.DataFrame({v: {"n": d["n"], "n_pareto": len(d["pareto"])}
                  for v, d in sub.items()}).T.to_csv(
        os.path.join(RESULTS, "subgroup_asa.csv"))
    best = grid[headline[0]] if headline and headline[0] < len(grid) else grid[0]
    ts = robustness.threshold_sensitivity(
        lambda thr: build_cases(meta, risk_by_case, thr=thr)[0], best, [60, 65, 70])
    pd.DataFrame(ts).T.to_csv(os.path.join(RESULTS, "threshold_sensitivity.csv"))

    print("[done] results in", RESULTS)


if __name__ == "__main__":
    main()
