"""End-to-end (Part B): subject-level splits -> multimodal risk model + ablations ->
constrained utility/burden frontier on TEST subjects -> subject-level bootstrap.

  python run_pipeline.py --limit 400 --epochs 4   # fast smoke
  python run_pipeline.py                            # full cohort (GPU server)
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import mm_risk  # noqa: E402
import robustness  # noqa: E402
import vitaldb_load as V  # noqa: E402
from frontier import bootstrap_policy, constrained_pareto, evaluate_grid, is_feasible  # noqa: E402
from labels import MAP_THR, exclude_mask, hypotension_episodes  # noqa: E402
from metrics import CaseEval  # noqa: E402
from metrics import objectives as obj_of  # noqa: E402
from policies import Policy, policy_grid  # noqa: E402

OBJ = ["utility", "burden"]
FALSE_CAPS = [1.0, 2.0, 4.0]
MAP_THRESHOLDS = [65, 70, 72, 75]
RESULTS = os.path.join(os.path.dirname(__file__), "results")
HEADLINE_FALSE_CAP = 2.0
HEADLINE_BUDGET = 4.0
HEADLINE_PPV_MIN = 0.40
HEADLINE_GAP_MAX = 0.10
os.makedirs(RESULTS, exist_ok=True)


def age_band(a):
    return "<50" if a < 50 else ("50-65" if a < 65 else "65+")


def _subgroup(row):
    return {"asa": str(row.asa), "age_band": age_band(row.age), "sex": row.sex}


def build_cases(meta, risk_by_case, thr=None):
    cases, subs = [], []
    for row in meta.itertuples(index=False):
        cid = int(row.caseid)
        df = V.load_cached_case(cid)
        sg = _subgroup(row)
        if thr is None:
            ce = CaseEval.from_frame(df, risk_by_case[cid], sg)
        else:
            m = df["map"].to_numpy(float)
            eps = hypotension_episodes(m, thr=thr)
            ex = exclude_mask(len(m), eps)
            ce = CaseEval(minute=np.arange(len(m)), risk=np.asarray(risk_by_case[cid],
                          float), episodes=eps,
                          hours=float(np.count_nonzero(~ex)) / 60.0,
                          exclude=ex, subgroup=sg)
        cases.append(ce)
        subs.append(int(row.subjectid))
    return cases, subs


def map_baseline_rows(meta):
    rows, prel = [], []
    for r in meta.itertuples(index=False):
        m = V.load_cached_case(int(r.caseid))["map"].to_numpy(float)
        eps = hypotension_episodes(m)
        prel.append((m, _subgroup(r), eps, exclude_mask(len(m), eps)))
    for T in MAP_THRESHOLDS:
        for C in (0, 5):
            cs = [CaseEval(minute=np.arange(len(m)), risk=(m < T).astype(float),
                           episodes=eps,
                           hours=float(np.count_nonzero(~ex)) / 60.0,
                           exclude=ex, subgroup=sg)
                  for m, sg, eps, ex in prel]
            o = dict(obj_of(cs, Policy(tau=0.5, m=1, C=C)))
            o.update(tau=0.5, m=1, C=C, trend=None, method=f"MAP<{T}")
            rows.append(o)
    return rows


def gbm_and_currentmap(train_meta, test_meta):
    """Reference test AUC/Brier for the ablation table: GBM (trend), GBM (leaky), and
    the current-MAP rule."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import brier_score_loss, roc_auc_score

    import risk_model
    from features import LEAK, TREND
    Xtr, ytr, _ = risk_model.build_dataset(train_meta)
    Xte, yte, _ = risk_model.build_dataset(test_meta)

    def fit(cols):
        g = GradientBoostingClassifier(max_depth=3, n_estimators=150, random_state=0)
        g.fit(Xtr[cols], ytr)
        p = g.predict_proba(Xte[cols])[:, 1]
        return {"auc": roc_auc_score(yte, p), "brier_raw": brier_score_loss(yte, p)}

    pm = np.clip((MAP_THR + 5 - Xte["map_now"]) / 20.0, 0, 1).to_numpy()
    return {"gbm_trend": fit(TREND), "gbm_leaky": fit(LEAK),
            "current_map": {"auc": roc_auc_score(yte, pm),
                            "brier_raw": brier_score_loss(yte, pm)}}


def make_grid():
    return policy_grid(taus=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                       ms=[1, 2, 3], cs=[0, 3, 5, 10], trends=[None, 0.02])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--boot", type=int, default=200)
    args = ap.parse_args()

    print("[1] cohort + subject-level 4-way split")
    meta = V.build_cache(limit=args.limit)
    splits = V.subject_split(meta, seed=0)
    print("    splits (cases):", {k: len(v) for k, v in splits.items()})
    sub_meta = {k: meta[meta.caseid.isin(v)] for k, v in splits.items()}

    print("[2] multimodal risk model + modality ablations (TEST metrics)")
    fusion, iso, norm, static_X, abl = mm_risk.run(splits, epochs=args.epochs)
    ref = gbm_and_currentmap(sub_meta["train"], sub_meta["test"])
    abl = pd.concat([abl, pd.DataFrame(ref).T])
    abl.to_csv(os.path.join(RESULTS, "ablations.csv"))
    print(abl.round(3).to_string())

    print("[3] calibrated risk streams for validation and TEST cases")
    val_risk = mm_risk.risk_streams(fusion, iso, splits["val"], norm, static_X)
    val_cases, _ = build_cases(sub_meta["val"], val_risk)
    risk_by_case = mm_risk.risk_streams(fusion, iso, splits["test"], norm, static_X)
    cases, subjects = build_cases(sub_meta["test"], risk_by_case)

    print("[4] select policy on validation; evaluate frontier on TEST")
    grid = make_grid()
    val_rows = evaluate_grid(val_cases, grid, OBJ)
    pd.DataFrame(val_rows).to_csv(
        os.path.join(RESULTS, "validation_frontier.csv"), index=False)
    candidates = [i for i, row in enumerate(val_rows)
                  if is_feasible(row, HEADLINE_FALSE_CAP, HEADLINE_PPV_MIN,
                                 HEADLINE_GAP_MAX)
                  and row["burden"] <= HEADLINE_BUDGET]
    selected = (max(candidates, key=lambda i: val_rows[i]["utility"])
                if candidates else None)

    model_rows = evaluate_grid(cases, grid, OBJ)
    for i, row in enumerate(model_rows):
        row["selected_on_val"] = i == selected
    baseline_rows = map_baseline_rows(sub_meta["test"])
    for row in baseline_rows:
        row["selected_on_val"] = False
    rows = model_rows + baseline_rows
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "frontier.csv"), index=False)
    headline = None
    for cap in FALSE_CAPS:
        nd, feas = constrained_pareto(rows, OBJ, false_cap=cap)
        print("    false<=%.0f/hr:  feasible=%3d  pareto=%2d" % (cap, len(feas), len(nd)))
        if cap == 2.0:
            headline = nd

    print("[5] subject-level bootstrap CI for validation-selected policy")
    if selected is not None:
        ci = bootstrap_policy(cases, grid[selected], OBJ, subjects=subjects,
                              n_boot=args.boot)
        pd.DataFrame({f"{o}_{b}": [ci[o][j]] for o in OBJ
                      for j, b in enumerate(("mean", "lo", "hi"))}).to_csv(
            os.path.join(RESULTS, "selected_policy_ci.csv"), index=False)
    else:
        print("    no policy met validation constraints")

    print("[6] robustness (TEST)")
    sub = robustness.subgroup_frontiers(cases, grid, OBJ, "asa")
    pd.DataFrame({v: {"n": d["n"], "n_pareto": len(d["pareto"])}
                  for v, d in sub.items()}).T.to_csv(
        os.path.join(RESULTS, "subgroup_asa.csv"))
    best = grid[selected] if selected is not None else grid[0]
    ts = robustness.threshold_sensitivity(
        lambda thr: build_cases(sub_meta["test"], risk_by_case, thr=thr)[0],
        best, [60, 65, 70])
    pd.DataFrame(ts).T.to_csv(os.path.join(RESULTS, "threshold_sensitivity.csv"))

    print("[done] results in", RESULTS)


if __name__ == "__main__":
    main()
