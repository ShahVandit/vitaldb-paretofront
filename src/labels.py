"""Hypotension event detection on the cleaned 1-min MAP grid.

A hypotension minute is MAP < MAP_THR. Consecutive/near-consecutive low minutes are
merged into a single **episode**: two below-threshold runs separated by a recovery
shorter than REFRACTORY minutes count as one episode (a MAP that dips, briefly
recovers, and dips again is one clinical event, not two). This prevents a single
alert from being credited for several nearby onsets.

NaN minutes (no arterial data) are treated as not-hypotensive and break runs.
"""
from __future__ import annotations

import numpy as np

MAP_THR = 65.0        # mmHg
MIN_EVENT_MIN = 1     # sustained minutes below threshold to start a run
REFRACTORY = 10       # recovery shorter than this merges runs into one episode


def _below_runs(mapp, thr, min_len):
    """Contiguous (start, end_exclusive) runs of MAP < thr lasting >= min_len."""
    below = np.isfinite(mapp) & (np.asarray(mapp, float) < thr)
    runs, start = [], None
    n = len(below)
    for i in range(n):
        if below[i] and start is None:
            start = i
        if start is not None and (not below[i] or i == n - 1):
            end = i + 1 if below[i] else i          # exclusive
            if end - start >= min_len:
                runs.append((start, end))
            start = None
    return runs


def hypotension_episodes(mapp, thr: float = MAP_THR, min_len: int = MIN_EVENT_MIN,
                         gap: int = REFRACTORY) -> list[tuple[int, int]]:
    """Merged episodes as (onset, end_exclusive); runs split by < gap min are merged."""
    merged = []
    for s, e in _below_runs(mapp, thr, min_len):
        if merged and s - merged[-1][1] < gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


def episode_onsets(mapp, **kw) -> list[int]:
    """Onset minute of each episode (convenience)."""
    return [s for s, _ in hypotension_episodes(mapp, **kw)]


def exclude_mask(n: int, episodes: list[tuple[int, int]],
                 refractory: int = REFRACTORY) -> np.ndarray:
    """Boolean per-minute mask of times to EXCLUDE from risk training / false-alert
    accounting: minutes already hypotensive (inside an episode) or within `refractory`
    minutes after one ends."""
    m = np.zeros(n, dtype=bool)
    for s, e in episodes:
        m[s:min(n, e + refractory)] = True
    return m


def hypotension_onsets(mapp, thr: float = MAP_THR,
                       min_len: int = MIN_EVENT_MIN) -> list[int]:
    """Raw (un-merged) onset of each below-threshold run. Kept for callers/tests that
    want the pre-clustering behavior; scoring uses `hypotension_episodes`."""
    return [s for s, _ in _below_runs(mapp, thr, min_len)]
