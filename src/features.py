"""Per-minute features for near-term hypotension risk.

Two feature sets so the risk model can demonstrate the HPI leakage trap:
  TREND - trajectory/driver features that do NOT restate current arterial pressure:
          MAP/HR/SpO2 trends, pulse pressure (SBP-DBP), anesthetic infusion rates
          (propofol/remifentanil), BIS depth, ETCO2. These precede or cause
          hypotension without being the MAP label.
  LEAK  - TREND plus absolute pressure levels (current MAP, SBP, DBP). These track
          the label and inflate apparent performance -- the leakage demonstration.

Missing-data handling matters and differs by signal:
  - infusion rates: absent track == no infusion, so fill 0.
  - BIS / ETCO2: absent track == monitor not used (NOT a value of 0, which would be
    a flatline / apnea). Fill a clinically neutral value AND flag missingness so the
    model can tell "not measured" from "measured low".

Labels: y_t = 1 if a hypotension onset falls in (t+W_MIN, t+W_MAX].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from labels import hypotension_onsets
from metrics import W_MAX, W_MIN

TREND = ["map_slope3", "map_slope5", "map_std5",
         "hr_slope3", "hr_mean5", "spo2_mean5", "minute",
         "pp", "pp_slope3", "ppf_rate", "rftn_rate",
         "bis", "bis_slope3", "etco2", "etco2_slope3",
         "bis_missing", "etco2_missing"]
LEAK = TREND + ["map_now", "map_mean5", "sbp_now", "dbp_now"]

# neutral fill for signals where "missing" != 0 (BIS 0 = flatline, ETCO2 0 = apnea)
FILL = {"ppf_rate": 0.0, "rftn_rate": 0.0, "bis": 45.0, "etco2": 35.0}


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Column if present, else an all-NaN series (tolerates old 3-signal caches)."""
    if name in df.columns:
        return df[name]
    return pd.Series(np.nan, index=df.index, dtype=float)


def case_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-minute features aligned to df.minute (NaNs where a signal is missing)."""
    mp, hr, sp = _col(df, "map"), _col(df, "hr"), _col(df, "spo2")
    sbp, dbp = _col(df, "sbp"), _col(df, "dbp")
    ppf, rftn = _col(df, "ppf_rate"), _col(df, "rftn_rate")
    bis, etco2 = _col(df, "bis"), _col(df, "etco2")

    f = pd.DataFrame(index=df.index)
    # --- MAP/HR/SpO2 trajectory (non-leaky) ---
    f["map_slope3"] = mp.diff().rolling(3, min_periods=1).mean()
    f["map_slope5"] = mp.diff().rolling(5, min_periods=1).mean()
    f["map_std5"] = mp.rolling(5, min_periods=2).std()
    f["hr_slope3"] = hr.diff().rolling(3, min_periods=1).mean()
    f["hr_mean5"] = hr.rolling(5, min_periods=1).mean()
    f["spo2_mean5"] = sp.rolling(5, min_periods=1).mean()
    f["minute"] = df["minute"].to_numpy(dtype=float)
    # --- pulse pressure: narrows before hypotension (non-leaky morphology) ---
    pp = sbp - dbp
    f["pp"] = pp
    f["pp_slope3"] = pp.diff().rolling(3, min_periods=1).mean()
    # --- anesthetic depth drivers (non-leaky) ---
    f["ppf_rate"] = ppf
    f["rftn_rate"] = rftn
    f["bis"] = bis
    f["bis_slope3"] = bis.diff().rolling(3, min_periods=1).mean()
    f["etco2"] = etco2
    f["etco2_slope3"] = etco2.diff().rolling(3, min_periods=1).mean()
    f["bis_missing"] = bis.isna().astype(float)      # flag BEFORE filling
    f["etco2_missing"] = etco2.isna().astype(float)
    # --- absolute pressure levels (LEAKY) ---
    f["map_now"] = mp
    f["map_mean5"] = mp.rolling(5, min_periods=1).mean()
    f["sbp_now"] = sbp
    f["dbp_now"] = dbp
    return f


def fill_features(f: pd.DataFrame) -> pd.DataFrame:
    """Impute per-signal: neutral values for BIS/ETCO2, 0 for rates/trends."""
    return f.apply(lambda c: c.fillna(FILL.get(c.name, 0.0)))


def case_labels(df: pd.DataFrame) -> np.ndarray:
    """y_t = 1 if a hypotension onset lies in (t+W_MIN, t+W_MAX]."""
    n = len(df)
    onsets = hypotension_onsets(df["map"].to_numpy(dtype=float))
    y = np.zeros(n, dtype=int)
    for o in onsets:
        lo, hi = o - W_MAX, o - W_MIN          # minutes from which o is foreseeable
        y[max(0, lo):max(0, hi) + 1] = 1
    return y
