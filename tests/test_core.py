"""Core unit tests: episode clustering, one-to-one matching, utility, exclusion,
policy cooldown/persistence, and exact-replay order invariance."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from labels import exclude_mask, hypotension_episodes  # noqa: E402
from metrics import BETA, CaseEval, objectives_from_alerts, score_case  # noqa: E402
from policies import Policy, naive_threshold_policy  # noqa: E402


def _case(n, episodes, risk=None, sg=None):
    return CaseEval(minute=np.arange(n), risk=np.zeros(n) if risk is None else risk,
                    episodes=episodes, hours=n / 60.0,
                    exclude=exclude_mask(n, episodes), subgroup=sg or {})


def test_episode_merging():
    # two dips 3 min apart -> ONE episode; 20 min apart -> TWO
    close = np.array([80, 80, 60, 60, 80, 80, 80, 60, 60, 80, 80])
    assert hypotension_episodes(close) == [(2, 9)]
    far = np.array([60, 60] + [80] * 20 + [60, 60])
    assert hypotension_episodes(far) == [(0, 2), (22, 24)]


def test_one_to_one_matching():
    # two episodes; a single alert sits in both windows -> detects only ONE
    case = _case(50, [(20, 21), (32, 33)])
    s = score_case(case, np.array([18]))
    assert s["n_detected"] == 1


def test_utility_rewards_detection_and_timeliness():
    ep = [(20, 21)]
    # missed -> utility 0
    assert objectives_from_alerts([_case(40, ep)], [np.array([])])["utility"] == 0.0
    # 1-min warning -> BETA (intrinsic detection value, not 0)
    u1 = objectives_from_alerts([_case(40, ep)], [np.array([19])])["utility"]
    assert abs(u1 - BETA) < 1e-9
    # 10-min warning -> 1.0 (max)
    u10 = objectives_from_alerts([_case(40, ep)], [np.array([10])])["utility"]
    assert abs(u10 - 1.0) < 1e-9


def test_already_hypotensive_alert_suppressed():
    case = _case(50, [(20, 25)])            # episode 20-24; refractory to 35
    s = score_case(case, np.array([22, 40]))  # 22 suppressed (in-episode), 40 kept+false
    assert s["n_alerts"] == 1 and s["n_false"] == 1


def test_ppv_is_actionable():
    # one episode, four pre-window alerts -> only ONE is useful; PPV = 1/4
    case = _case(40, [(20, 21)])
    s = score_case(case, np.array([8, 10, 12, 19]))
    assert s["n_detected"] == 1 and s["n_alerts"] == 4 and s["n_false"] == 3
    o = objectives_from_alerts([case], [np.array([8, 10, 12, 19])])
    assert abs(o["ppv"] - 0.25) < 1e-9


def test_cooldown_collapses_repeats():
    minute = np.arange(10)
    risk = np.array([0, 0.9, 0.9, 0.9, 0, 0, 0.9, 0, 0, 0])
    assert len(naive_threshold_policy(0.5).alert_times(minute, risk)) == 4
    assert list(Policy(tau=0.5, C=5).alert_times(minute, risk)) == [1, 6]


def test_persistence_requires_sustained_risk():
    minute = np.arange(5)
    risk = np.array([0.9, 0, 0.9, 0.9, 0])
    assert list(Policy(tau=0.5, m=2).alert_times(minute, risk)) == [3]


def test_exact_replay_order_invariance():
    cases = [_case(30, [(15, 16)], risk=np.r_[np.zeros(10), np.full(4, 0.9), np.zeros(16)],
                   sg={"asa": str(k % 2)}) for k in range(4)]
    p = Policy(tau=0.5, C=3)
    a = [p.alert_times(c.minute, c.risk) for c in cases]
    o1 = objectives_from_alerts(cases, a)
    idx = [2, 0, 3, 1]
    o2 = objectives_from_alerts([cases[i] for i in idx], [a[i] for i in idx])
    for k in ["utility", "sensitivity", "false_rate", "burden"]:
        assert abs(o1[k] - o2[k]) < 1e-9


if __name__ == "__main__":
    for fn in list(globals()):
        if fn.startswith("test_"):
            globals()[fn]()
            print("ok", fn)
