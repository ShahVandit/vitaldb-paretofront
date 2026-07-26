"""Contextual-bandit offline alert policy.

Per minute the agent observes context x_t = (trend features, learned risk) and
chooses action a in {alert, silence}. The immediate reward of alerting is
  +w_detect  if the minute lies in an event's actionable pre-window,
  -w_false   otherwise,
with a per-alert -w_burden and a cooldown that blocks repeats. The myopic
reward-optimal action label is therefore the pre-event indicator (features.case_labels);
we fit a reward-weighted linear policy s(x)=w.x+b0 and fire when s(x) >= bias with
cooldown C.

Evaluation is EXACT retrospective replay, not estimated OPE: alerting does not alter
the patient trajectory, so a policy's score on the logged data is its true score.
This is why off-policy evaluation is valid here for alerting but not for treatment.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from features import TREND, case_features, case_labels, fill_features
from metrics import objectives_from_alerts


def threshold_cooldown(minute: np.ndarray, signal: np.ndarray,
                       thr: float, C: int) -> np.ndarray:
    """Fire when signal >= thr, respecting a C-minute cooldown."""
    last, out = -np.inf, []
    for i, t in enumerate(minute):
        if signal[i] >= thr and (t - last) >= C:
            out.append(t)
            last = t
    return np.asarray(out, dtype=float)


def _context(df, risk):
    f = fill_features(case_features(df))[TREND].to_numpy()
    return np.column_stack([f, np.asarray(risk, dtype=float)])


def train_bandit(frames, risks, w_detect=1.0, w_false=1.0, seed=0):
    """Fit a reward-weighted linear alert policy over pooled minutes."""
    X, y, sw = [], [], []
    for df, risk in zip(frames, risks):
        Xi = _context(df, risk)
        yi = case_labels(df)
        X.append(Xi)
        y.append(yi)
        sw.append(np.where(yi == 1, w_detect, w_false))
    X = np.vstack(X)
    y = np.concatenate(y)
    sw = np.concatenate(sw)
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(X, y, sample_weight=sw)
    return clf


def score_streams(clf, frames, risks) -> list[np.ndarray]:
    """Per-case decision-function score stream."""
    return [clf.decision_function(_context(df, risk)) for df, risk in zip(frames, risks)]


def sweep(cases, clf, frames, risks, biases, C=3) -> list[dict]:
    """Trace bandit operating points across a decision-bias grid."""
    scores = score_streams(clf, frames, risks)
    rows = []
    for b in biases:
        alerts = [threshold_cooldown(c.minute, s, b, C) for c, s in zip(cases, scores)]
        o = dict(objectives_from_alerts(cases, alerts))
        o.update(method="bandit", bias=float(b), C=C)
        rows.append(o)
    return rows
