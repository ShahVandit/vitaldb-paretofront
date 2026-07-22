"""Hypotension event detection on the cleaned 1-min MAP grid.

Event = MAP < MAP_THR for >= MIN_EVENT_MIN contiguous minutes; onset is the first
minute of the run. NaN minutes (no arterial data) break runs and are never counted
as hypotension, so events are only asserted where MAP is actually observed.
"""
from __future__ import annotations

import numpy as np

MAP_THR = 65.0       # mmHg
MIN_EVENT_MIN = 1    # sustained minutes below threshold


def hypotension_onsets(mapp, thr: float = MAP_THR,
                       min_len: int = MIN_EVENT_MIN) -> list[int]:
    """Return minute indices at which hypotension events begin."""
    onsets, run = [], 0
    for i, x in enumerate(mapp):
        below = np.isfinite(x) and x < thr
        run = run + 1 if below else 0
        if below and run == min_len:
            onsets.append(i - min_len + 1)
    return onsets
