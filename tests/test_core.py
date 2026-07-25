"""Core unit tests: event detection, cooldown, and exact-replay invariance."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from labels import hypotension_onsets  # noqa: E402
from metrics import CaseEval, objectives_from_alerts  # noqa: E402
from policies import Policy, naive_threshold_policy  # noqa: E402


def test_event_detection_and_nan_break():
    mapp = np.array([80, 60, 60, 80, np.nan, 60, 80])  # events at idx1 and idx5
    assert hypotension_onsets(mapp, thr=65, min_len=1) == [1, 5]
    # a NaN inside a low run must break the run
    mapp2 = np.array([60, np.nan, 60])
    assert hypotension_onsets(mapp2, thr=65, min_len=2) == []


def test_cooldown_collapses_repeats():
    minute = np.arange(10)
    risk = np.array([0, 0.9, 0.9, 0.9, 0, 0, 0.9, 0, 0, 0])
    no_cd = naive_threshold_policy(0.5).alert_times(minute, risk)
    cd = Policy(tau=0.5, C=5).alert_times(minute, risk)
    assert len(no_cd) == 4                # fires every minute above threshold
    assert list(cd) == [1, 6]             # cooldown collapses the 1-3 burst to one


def test_persistence_requires_sustained_risk():
    minute = np.arange(5)
    risk = np.array([0.9, 0, 0.9, 0.9, 0])   # only idx2-3 is a 2-min run
    assert list(Policy(tau=0.5, m=2).alert_times(minute, risk)) == [3]


def _toy_cases():
    cs = []
    for k in range(4):
        n = 30
        risk = np.zeros(n)
        risk[10:14] = 0.9
        ce = CaseEval(minute=np.arange(n), risk=risk,
                      onsets=[15], hours=n / 60.0, subgroup={"asa": str(k % 2)})
        cs.append(ce)
    return cs


def test_exact_replay_order_invariance():
    cases = _toy_cases()
    p = Policy(tau=0.5, C=3)
    a = [p.alert_times(c.minute, c.risk) for c in cases]
    o1 = objectives_from_alerts(cases, a)
    idx = [2, 0, 3, 1]
    o2 = objectives_from_alerts([cases[i] for i in idx], [a[i] for i in idx])
    for k in ["sensitivity", "false_rate", "burden"]:
        assert abs(o1[k] - o2[k]) < 1e-9


if __name__ == "__main__":
    for fn in list(globals()):
        if fn.startswith("test_"):
            globals()[fn]()
            print("ok", fn)
