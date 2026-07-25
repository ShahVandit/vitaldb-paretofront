"""Robustness analyses required by the JD: subgroup frontiers, missing-signal
sensitivity, and hypotension-threshold sensitivity."""
from __future__ import annotations

from frontier import evaluate_grid, pareto_indices
from metrics import objectives


def subgroup_frontiers(cases, policies, names, key: str) -> dict:
    """Pareto frontier computed separately within each subgroup value."""
    out = {}
    for v in sorted({str(c.subgroup.get(key, "NA")) for c in cases}):
        sub = [c for c in cases if str(c.subgroup.get(key, "NA")) == v]
        if len(sub) < 20:
            continue
        rows = evaluate_grid(sub, policies, names)
        out[v] = {"n": len(sub), "rows": rows, "pareto": pareto_indices(rows, names)}
    return out


def missing_signal_effect(policy, cases_full, cases_masked) -> dict:
    """Objectives with all signals vs with one input masked (risk recomputed upstream)."""
    return {"full": objectives(cases_full, policy),
            "masked": objectives(cases_masked, policy)}


def threshold_sensitivity(rebuild, policy, thrs) -> dict:
    """Re-evaluate a policy as the MAP hypotension threshold is varied.

    `rebuild(thr)` returns a fresh list[CaseEval] with events labeled at `thr`.
    """
    return {thr: objectives(rebuild(thr), policy) for thr in thrs}
