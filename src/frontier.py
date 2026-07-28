"""Pareto non-dominated set over policies + bootstrap confidence bands.

Also exposes the 'not-an-ROC' witness: pairs of policies at (near-)equal
sensitivity and false-rate that differ on a temporal axis (warning time / burden).
"""
from __future__ import annotations

import numpy as np

from metrics import DIRECTIONS, objectives


def _signed(row, names):
    return np.array([DIRECTIONS[n] * row[n] for n in names], dtype=float)


def pareto_indices(rows: list[dict], names: list[str]) -> list[int]:
    """Indices of non-dominated rows (higher-is-better after DIRECTIONS sign)."""
    V = [_signed(r, names) for r in rows]
    nd = []
    for i in range(len(V)):
        if not any(j != i and np.all(V[j] >= V[i]) and np.any(V[j] > V[i])
                   for j in range(len(V))):
            nd.append(i)
    return nd


def is_feasible(row: dict, false_cap: float, ppv_min: float = 0.40,
                gap_max: float = 0.10) -> bool:
    """Clinical safety constraints on an alerting policy (review C5)."""
    ppv = row.get("ppv", float("nan"))
    return (row.get("false_rate", float("inf")) <= false_cap
            and np.isfinite(ppv) and ppv >= ppv_min
            and row.get("disparity", float("inf")) <= gap_max)


def constrained_pareto(rows: list[dict], names: list[str], false_cap: float,
                       ppv_min: float = 0.40, gap_max: float = 0.10):
    """Pareto set among policies that satisfy the feasibility constraints.

    Returns (pareto_indices_into_rows, feasible_indices_into_rows)."""
    feas = [i for i, r in enumerate(rows)
            if is_feasible(r, false_cap, ppv_min, gap_max)]
    if not feas:
        return [], []
    nd = pareto_indices([rows[i] for i in feas], names)
    return [feas[k] for k in nd], feas


def evaluate_grid(cases, policies, names) -> list[dict]:
    """Objective vector + policy params for each policy in the grid."""
    rows = []
    for p in policies:
        o = dict(objectives(cases, p))
        o.update(tau=p.tau, m=p.m, C=p.C, trend=p.trend)
        rows.append(o)
    return rows


def bootstrap_policy(cases, policy, names, subjects=None, n_boot=200, seed=0) -> dict:
    """Mean and 95% CI per objective. If `subjects` (one id per case) is given, use a
    **cluster bootstrap over subjects** (correct unit of resampling; review C6);
    otherwise resample cases."""
    rng = np.random.default_rng(seed)
    n = len(cases)
    if subjects is not None:
        subjects = np.asarray(subjects)
        uniq = np.unique(subjects)
        by_sub = {s: np.where(subjects == s)[0] for s in uniq}
    samp = {k: [] for k in names}
    for _ in range(n_boot):
        if subjects is not None:
            chosen = rng.choice(uniq, len(uniq), replace=True)
            idx = np.concatenate([by_sub[s] for s in chosen])
        else:
            idx = rng.integers(0, n, n)
        o = objectives([cases[i] for i in idx], policy)
        for k in names:
            samp[k].append(o[k])
    return {k: (float(np.nanmean(v)),
                float(np.nanpercentile(v, 2.5)),
                float(np.nanpercentile(v, 97.5))) for k, v in samp.items()}


def not_an_roc_pairs(rows: list[dict], sens_tol=0.02, fr_tol=0.05) -> list[tuple]:
    """Find policy pairs at ~equal (sensitivity, false_rate) but differing on a
    temporal axis. Their existence proves the frontier is not an ROC curve."""
    out = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            if (abs(a["sensitivity"] - b["sensitivity"]) <= sens_tol and
                    abs(a["false_rate"] - b["false_rate"]) <= fr_tol and
                    (abs(a["warning_time"] - b["warning_time"]) >= 1.0 or
                     abs(a["burden"] - b["burden"]) >= 0.5)):
                out.append((i, j))
    return out
