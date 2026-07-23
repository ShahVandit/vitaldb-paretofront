"""Temporal alert policies: map a per-minute risk stream to alert times.

A policy is (tau, m, C, trend). The parameters m (persistence) and C (cooldown)
change alert timing and count WITHOUT moving the ROC operating point, which is what
makes the downstream frontier more than an ROC curve.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Policy:
    tau: float          # risk threshold
    m: int = 1          # consecutive minutes above tau required (debounce)
    C: int = 0          # cooldown minutes after an alert
    trend: float | None = None  # early trigger if risk rises faster than this/min

    def alert_times(self, times: np.ndarray, risk: np.ndarray) -> np.ndarray:
        """Return the times at which this policy fires an alert."""
        n = len(risk)
        run = 0
        last_alert = -np.inf
        fired = []
        drisk = np.diff(risk, prepend=risk[0])  # per-minute change
        for i in range(n):
            above = risk[i] >= self.tau
            run = run + 1 if above else 0
            persist_ok = run >= self.m
            trend_ok = self.trend is not None and drisk[i] >= self.trend
            if (persist_ok or trend_ok) and (times[i] - last_alert) >= self.C:
                fired.append(times[i])
                last_alert = times[i]
        return np.asarray(fired, dtype=float)


def naive_threshold_policy(tau: float) -> Policy:
    """The degenerate ROC-baseline policy: fire every minute risk>=tau."""
    return Policy(tau=tau, m=1, C=0, trend=None)


def policy_grid(taus, ms, cs, trends) -> list[Policy]:
    """Cartesian product of policy parameters -> list of Policy objects."""
    grid = []
    for tau in taus:
        for m in ms:
            for C in cs:
                for tr in trends:
                    grid.append(Policy(tau=tau, m=m, C=C, trend=tr))
    return grid
