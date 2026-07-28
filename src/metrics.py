"""Event-level scoring of alerting policies.

Events are hypotension **episodes** (merged runs; see labels.py). Matching is
**one-to-one**: each episode is detected by at most one alert (the earliest alert in
its actionable window [onset-W_MAX, onset-W_MIN]), and each alert credits at most one
episode. An alert that lands in an episode's pre-window, inside the episode, or in its
post-episode refractory window is **not** a false alarm; any other uncredited alert is.

Objectives:
  utility      = mean over ALL episodes of episode utility, where a detected episode
                 scores BETA + (1-BETA)*timing(warn) and a missed episode scores 0.
                 So detection has intrinsic value (BETA) and earlier warning adds more;
                 missed episodes drag the mean down (can't be gamed).
  sensitivity, warning_time, ppv, false_rate, burden, disparity  -- reported.

The Pareto frontier uses only (utility, burden); the rest are reported context.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from labels import REFRACTORY, exclude_mask, hypotension_episodes

W_MIN = 1              # alert must precede onset by >= this (min)
W_MAX = 15            # ... and <= this to be actionable (min)
SUBGROUP_KEY = "asa"  # axis for the disparity objective
BETA = 0.5            # intrinsic value of detecting at all (vs timeliness)
G_LO, G_HI = 1, 10   # timing reward ramps 0->1 as warning goes G_LO->G_HI min

DIRECTIONS = {
    "utility": +1, "sensitivity": +1, "warning_time": +1, "ppv": +1,
    "false_rate": -1, "burden": -1, "disparity": -1,
}


def timing_reward(warn: float) -> float:
    return float(np.clip((warn - G_LO) / (G_HI - G_LO), 0.0, 1.0))


@dataclass
class CaseEval:
    minute: np.ndarray                 # 0..n-1
    risk: np.ndarray                   # per-minute risk in [0,1]
    episodes: list                     # list of (onset, end_exclusive)
    hours: float                       # analysis-window hours
    exclude: np.ndarray                # minutes already-hypotensive / in refractory
    subgroup: dict = field(default_factory=dict)

    @classmethod
    def from_frame(cls, df, risk, subgroup=None):
        mapp = df["map"].to_numpy(dtype=float)
        n = len(mapp)
        eps = hypotension_episodes(mapp)
        ex = exclude_mask(n, eps)
        # rate denominator = eligible monitoring hours (exclude active-event/refractory)
        return cls(minute=np.arange(n), risk=np.asarray(risk, float), episodes=eps,
                   hours=float(np.count_nonzero(~ex)) / 60.0, exclude=ex,
                   subgroup=subgroup or {})


def score_case(case: CaseEval, alerts: np.ndarray) -> dict:
    """Per-case counts with one-to-one alert<->episode matching and **actionable**
    accounting: alerts fired while already hypotensive / in refractory are suppressed
    (the system would not alert during known ongoing hypotension), and only the single
    matched alert per episode is 'useful' -- every other alert counts against PPV.
    So PPV = matched_alerts / total_alerts (not the older near-episode leniency)."""
    n = len(case.minute)
    a = np.sort(np.asarray(alerts, dtype=int)) if len(alerts) else np.empty(0, int)
    if a.size and case.exclude.size:                    # suppress excluded-time alerts
        a = a[~case.exclude[np.clip(a, 0, n - 1)]]
    used = np.zeros(len(a), dtype=bool)

    matched, warn, util = 0, [], 0.0
    for onset, _end in sorted(case.episodes):
        lo, hi = onset - W_MAX, onset - W_MIN
        l = np.searchsorted(a, lo, "left")
        r = np.searchsorted(a, hi, "right")
        j = next((k for k in range(l, r) if not used[k]), None)   # earliest unused
        if j is not None:
            used[j] = True
            matched += 1
            w = onset - int(a[j])
            warn.append(w)
            util += BETA + (1 - BETA) * timing_reward(w)

    n_alerts = int(a.size)
    n_false = n_alerts - matched          # every non-matched alert is non-actionable
    return {"n_events": len(case.episodes), "n_detected": matched, "warn": warn,
            "util": util, "n_alerts": n_alerts, "n_false": n_false,
            "hours": case.hours}


def objectives_from_alerts(cases: list[CaseEval], alerts_list: list[np.ndarray],
                           subgroup_key: str = SUBGROUP_KEY) -> dict:
    scored = [(c, score_case(c, a)) for c, a in zip(cases, alerts_list)]
    ev = sum(s["n_events"] for _, s in scored)
    det = sum(s["n_detected"] for _, s in scored)
    hrs = sum(s["hours"] for _, s in scored) or np.nan
    false = sum(s["n_false"] for _, s in scored)
    alerts = sum(s["n_alerts"] for _, s in scored)
    util = sum(s["util"] for _, s in scored)
    warn = [w for _, s in scored for w in s["warn"]]

    groups: dict = {}
    for c, s in scored:
        g = c.subgroup.get(subgroup_key, "NA")
        d, e = groups.get(g, (0, 0))
        groups[g] = (d + s["n_detected"], e + s["n_events"])
    gs = [d / e for d, e in groups.values() if e >= 5]
    # A missing comparison is unknown, not evidence of equal performance.
    disparity = (max(gs) - min(gs)) if len(gs) > 1 else np.nan

    return {
        "utility": util / ev if ev else np.nan,
        "sensitivity": det / ev if ev else np.nan,
        "warning_time": float(np.median(warn)) if warn else 0.0,
        "ppv": (alerts - false) / alerts if alerts else np.nan,
        "false_rate": false / hrs,
        "burden": alerts / hrs,
        "disparity": disparity,
    }


def objectives(cases: list[CaseEval], policy,
               subgroup_key: str = SUBGROUP_KEY) -> dict:
    alerts_list = [policy.alert_times(c.minute, c.risk) for c in cases]
    return objectives_from_alerts(cases, alerts_list, subgroup_key)
