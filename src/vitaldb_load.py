"""VitalDB cohort selection, cached track download, cleaning, and 1-min resampling.

Per case we store a cleaned 1-minute grid of the signals defined in ``TRACKS``:
MAP, systolic/diastolic pressure, HR, SpO2, propofol & remifentanil infusion rates,
BIS, and ETCO2. MAP (Solar8000/ART_MBP) is required and defines the cohort
(~3,724 arterial-line cases); the other signals are project-relevant predictors of
intraoperative hypotension and are stored as NaN columns for cases that lack them.
The full clinical table (all columns, all cases) is cached separately in
``_cases.parquet``.

Data facts (verified against the live API):
  - Track `Time` and clinical anestart/aneend are seconds from casestart (=0).
  - Solar8000/ART_MBP is ~2 s sampled and artifact-laden (negative values before
    line insertion, flush spikes). Per-minute median over in-range samples both
    resamples and despikes; out-of-range samples per TRACKS ranges are dropped.
"""
from __future__ import annotations

import io
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

API = "https://api.vitaldb.net"
HERE = os.path.dirname(__file__)
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
os.makedirs(DATA, exist_ok=True)

# Signals stored per case: column name -> (VitalDB track, physiologic range).
# Out-of-range samples are dropped as artifacts. MAP is required (defines events);
# the rest are project-relevant predictors of intraoperative hypotension:
#   sbp/dbp    -> pulse pressure narrows before hypotension
#   ppf/rftn   -> anesthetic depth (propofol/remifentanil) drives hypotension
#   bis        -> depth-of-anesthesia monitor
#   etco2      -> falls when cardiac output drops
TRACKS = {
    "map":       ("Solar8000/ART_MBP", (20.0, 200.0)),
    "sbp":       ("Solar8000/ART_SBP", (30.0, 300.0)),
    "dbp":       ("Solar8000/ART_DBP", (10.0, 200.0)),
    "hr":        ("Solar8000/HR", (20.0, 220.0)),
    "spo2":      ("Solar8000/PLETH_SPO2", (50.0, 100.0)),
    "ppf_rate":  ("Orchestra/PPF20_RATE", (0.0, 1200.0)),
    "rftn_rate": ("Orchestra/RFTN20_RATE", (0.0, 1200.0)),
    "bis":       ("BIS/BIS", (0.0, 100.0)),
    "etco2":     ("Solar8000/ETCO2", (0.0, 100.0)),
}
MAP_TRK = TRACKS["map"][0]   # required track for cohort selection

SUBGROUP_COLS = ["age", "sex", "asa", "optype", "ane_type", "emop",
                 "death_inhosp", "icu_days"]


def _get_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=90)
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content))


def get_cases(refresh: bool = False) -> pd.DataFrame:
    p = os.path.join(DATA, "_cases.parquet")
    if os.path.exists(p) and not refresh:
        return pd.read_parquet(p)
    df = _get_csv(f"{API}/cases")
    df.to_parquet(p)
    return df


def get_trks(refresh: bool = False) -> pd.DataFrame:
    p = os.path.join(DATA, "_trks.parquet")
    if os.path.exists(p) and not refresh:
        return pd.read_parquet(p)
    df = _get_csv(f"{API}/trks")
    df.to_parquet(p)
    return df


def select_cohort(min_dur_min: float = 20.0, min_age: float = 18.0) -> pd.DataFrame:
    """Return one row per eligible case with subgroup metadata and analysis window."""
    cases = get_cases()
    trks = get_trks()
    art_ids = set(trks.loc[trks.tname == MAP_TRK, "caseid"])
    c = cases[cases.caseid.isin(art_ids)].copy()
    c["win_start"] = c["anestart"].clip(lower=0)
    c["win_end"] = c["aneend"]
    c["dur_min"] = (c["win_end"] - c["win_start"]) / 60.0
    keep = (c.age >= min_age) & (c.dur_min >= min_dur_min) & c.win_end.notna()
    cols = ["caseid", "win_start", "win_end", "dur_min"] + SUBGROUP_COLS
    return c.loc[keep, cols].reset_index(drop=True)


def _tids_for(trks: pd.DataFrame, caseid: int) -> dict:
    """Resolve the track id for each signal in TRACKS (None if the case lacks it)."""
    sub = trks[trks.caseid == caseid]
    out = {}
    for key, (name, _rng) in TRACKS.items():
        row = sub[sub.tname == name]
        out[key] = row.iloc[0].tid if len(row) else None
    return out


def _clean_signal(tid: str, ws: float, we: float, rng: tuple) -> pd.Series:
    """Download one track, clip to window, drop artifacts, median-resample to 1-min.

    Returns a Series indexed by integer minute (0-based from ws), possibly with gaps.
    """
    if tid is None:
        return pd.Series(dtype=float)
    df = _get_csv(f"{API}/{tid}")
    if df.shape[1] < 2 or len(df) == 0:
        return pd.Series(dtype=float)
    t = df.iloc[:, 0].to_numpy(dtype=float)
    v = df.iloc[:, 1].to_numpy(dtype=float)
    lo, hi = rng
    m = (t >= ws) & (t <= we) & np.isfinite(v) & (v >= lo) & (v <= hi)
    if not m.any():
        return pd.Series(dtype=float)
    minute = np.floor((t[m] - ws) / 60.0).astype(int)
    s = pd.Series(v[m], index=minute)
    return s.groupby(level=0).median()


def clean_case(caseid: int, ws: float, we: float, trks: pd.DataFrame) -> pd.DataFrame:
    """Build the cleaned 1-min grid for one case: [minute] + one column per TRACKS key.

    MAP is required; a case with no usable arterial MAP returns an empty frame.
    Signals the case does not have become all-NaN columns.
    """
    tids = _tids_for(trks, caseid)
    mp = _clean_signal(tids["map"], ws, we, TRACKS["map"][1])
    if mp.empty:
        return pd.DataFrame()
    n = int(np.floor((we - ws) / 60.0)) + 1
    grid = pd.RangeIndex(0, n, name="minute")
    out = pd.DataFrame(index=grid)
    out["map"] = mp.reindex(grid)
    for key, (_name, rng) in TRACKS.items():
        if key == "map":
            continue
        out[key] = _clean_signal(tids[key], ws, we, rng).reindex(grid)
    return out.reset_index()


def build_cache(limit: int | None = None, workers: int = 8,
                min_dur_min: float = 20.0, refresh: bool = False) -> pd.DataFrame:
    """Download + clean + cache each case to data/{caseid}.parquet (resumable).

    Writes data/cohort.parquet with the subgroup metadata for cached cases.
    Returns the cohort metadata frame (only successfully-cached rows).
    `refresh=True` re-downloads even if a cached file already exists (needed after
    the stored signal set changes).
    """
    cohort = select_cohort(min_dur_min=min_dur_min)
    if limit is not None:
        cohort = cohort.head(limit)
    trks = get_trks()

    def work(row):
        cid = int(row.caseid)
        fp = os.path.join(DATA, f"{cid}.parquet")
        if os.path.exists(fp) and not refresh:
            return cid, True
        try:
            df = clean_case(cid, row.win_start, row.win_end, trks)
            if df.empty or df["map"].notna().sum() < min_dur_min:
                return cid, False
            df.to_parquet(fp)
            return cid, True
        except Exception:
            return cid, False

    ok = set()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, row) for row in cohort.itertuples(index=False)]
        for i, f in enumerate(as_completed(futs), 1):
            cid, good = f.result()
            if good:
                ok.add(cid)
            if i % 100 == 0:
                print(f"  cached {i}/{len(cohort)} ({len(ok)} ok)", flush=True)

    meta = cohort[cohort.caseid.isin(ok)].reset_index(drop=True)
    meta.to_parquet(os.path.join(DATA, "cohort.parquet"))
    print(f"[done] {len(meta)} cases cached to {DATA}")
    return meta


def load_cached_case(caseid: int) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(DATA, f"{caseid}.parquet"))


def load_cohort_meta() -> pd.DataFrame:
    return pd.read_parquet(os.path.join(DATA, "cohort.parquet"))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--refresh", action="store_true",
                    help="re-download even if cached (use after changing TRACKS)")
    args = ap.parse_args()
    build_cache(limit=args.limit, workers=args.workers, refresh=args.refresh)
