"""Per-minute features for near-term hypotension risk.

Two feature sets are defined so the risk model can demonstrate the HPI leakage trap:
  TREND   - trajectory/variability only, no absolute current MAP level
  LEAK    - TREND plus current MAP and trailing MAP level (the leaking features)

Labels: y_t = 1 if a hypotension onset falls in (t+W_MIN, t+W_MAX].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from labels import hypotension_onsets
from metrics import W_MAX, W_MIN

TREND = ["map_slope3", "map_slope5", "map_std5",
         "hr_slope3", "hr_mean5", "spo2_mean5", "minute"]
LEAK = TREND + ["map_now", "map_mean5"]


def case_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-minute features aligned to df.minute (NaNs where signal is missing)."""
    mp, hr, sp = df["map"], df["hr"], df["spo2"]
    f = pd.DataFrame(index=df.index)
    f["map_slope3"] = mp.diff().rolling(3, min_periods=1).mean()
    f["map_slope5"] = mp.diff().rolling(5, min_periods=1).mean()
    f["map_std5"] = mp.rolling(5, min_periods=2).std()
    f["hr_slope3"] = hr.diff().rolling(3, min_periods=1).mean()
    f["hr_mean5"] = hr.rolling(5, min_periods=1).mean()
    f["spo2_mean5"] = sp.rolling(5, min_periods=1).mean()
    f["minute"] = df["minute"].to_numpy(dtype=float)
    f["map_now"] = mp
    f["map_mean5"] = mp.rolling(5, min_periods=1).mean()
    return f


def case_labels(df: pd.DataFrame) -> np.ndarray:
    """y_t = 1 if a hypotension onset lies in (t+W_MIN, t+W_MAX]."""
    n = len(df)
    onsets = hypotension_onsets(df["map"].to_numpy(dtype=float))
    y = np.zeros(n, dtype=int)
    for o in onsets:
        lo, hi = o - W_MAX, o - W_MIN          # minutes from which o is foreseeable
        y[max(0, lo):max(0, hi) + 1] = 1
    return y
