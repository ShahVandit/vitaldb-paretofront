"""The five objectives, scored at the event/case level with timing.

An alert 'detects' an event if it lands in the actionable window
[onset-W_MAX, onset-W_MIN]. Warning time credits the earliest in-window alert
(max actionable lead). A false alert is any alert not inside some event's
[onset-W_MAX, onset] span. Sensitivity, false-rate and burden are pooled across
cases; warning time is the median over detected events; disparity is the max-min
sensitivity gap across a chosen subgroup axis.

Two policies at the same (sensitivity, false_rate) can differ on warning_time and
burden because persistence/cooldown reshape the alert stream in time. That is what
makes the frontier more than an ROC curve.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from labels import hypotension_onsets

W_MIN = 1            # alert must precede onset by >= this (min)
W_MAX = 15           # ... and <= this to be actionable (min)
SUBGROUP_KEY = "asa"  # axis for the disparity objective

# objective name -> +1 higher-is-better, -1 lower-is-better
DIRECTIONS = {
    "sensitivity": +1, "warning_time": +1,
    "false_rate": -1, "burden": -1, "disparity": -1,
}


@dataclass
class CaseEval:
    minute: np.ndarray          # 0..n-1
    risk: np.ndarray            # per-minute risk in [0,1]
    onsets: list                # hypotension onset minutes
    hours: float                # analysis-window hours (for rate denominators)
    subgroup: dict = field(default_factory=dict)

    @classmethod
    def from_frame(cls, df, risk, subgroup=None):
        mapp = df["map"].to_numpy(dtype=float)
        n = len(mapp)
        return cls(
            minute=np.arange(n),
            risk=np.asarray(risk, dtype=float),
            onsets=hypotension_onsets(mapp),
            hours=n / 60.0,
            subgroup=subgroup or {},
        )


def score_case(case: CaseEval, alerts: np.ndarray) -> dict:
    """Raw per-case counts used to aggregate the objectives (vectorized)."""
    n = len(case.minute)
    a = np.sort(np.asarray(alerts, dtype=int)) if len(alerts) else np.empty(0, int)
    onsets = case.onsets

    matched, warn = 0, []
    for o in onsets:                         # detection via sorted-alert search
        lo, hi = o - W_MAX, o - W_MIN
        l = np.searchsorted(a, lo, "left")
        r = np.searchsorted(a, hi, "right")
        if r > l:
            matched += 1
            warn.append(o - int(a[l]))       # earliest in-window = max lead

    near = np.zeros(n + 1, dtype=bool)        # false = alert not near any event
    for o in onsets:
        near[max(0, o - W_MAX):o + 1] = True
    false = int(np.count_nonzero(~near[np.clip(a, 0, n - 1)])) if a.size else 0

    return {"n_events": len(onsets), "n_detected": matched,
            "warn": warn, "n_alerts": int(a.size),
            "n_false": false, "hours": case.hours}


def objectives_from_alerts(cases: list[CaseEval], alerts_list: list[np.ndarray],
                           subgroup_key: str = SUBGROUP_KEY) -> dict:
    """Aggregate the five objectives given precomputed alert times per case.

    Shared by the grid policies and the learned bandit policy so both are scored
    identically (exact retrospective replay)."""
    scored = [(c, score_case(c, a)) for c, a in zip(cases, alerts_list)]

    ev = sum(s["n_events"] for _, s in scored)
    det = sum(s["n_detected"] for _, s in scored)
    hrs = sum(s["hours"] for _, s in scored) or np.nan
    false = sum(s["n_false"] for _, s in scored)
    alerts = sum(s["n_alerts"] for _, s in scored)
    warn = [w for _, s in scored for w in s["warn"]]

    groups: dict = {}
    for c, s in scored:
        g = c.subgroup.get(subgroup_key, "NA")
        d, e = groups.get(g, (0, 0))
        groups[g] = (d + s["n_detected"], e + s["n_events"])
    gs = [d / e for d, e in groups.values() if e >= 5]
    disparity = (max(gs) - min(gs)) if len(gs) > 1 else 0.0

    return {
        "sensitivity": det / ev if ev else np.nan,           # 1 - missed-event rate
        "warning_time": float(np.median(warn)) if warn else 0.0,  # time-to-event
        "ppv": (alerts - false) / alerts if alerts else np.nan,   # positive predictive value
        "false_rate": false / hrs,
        "burden": alerts / hrs,
        "disparity": disparity,
    }


def objectives(cases: list[CaseEval], policy,
               subgroup_key: str = SUBGROUP_KEY) -> dict:
    """Aggregate the five objectives over cases for one policy object."""
    alerts_list = [policy.alert_times(c.minute, c.risk) for c in cases]
    return objectives_from_alerts(cases, alerts_list, subgroup_key)
