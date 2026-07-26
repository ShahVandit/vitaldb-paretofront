"""Leakage-aware near-term hypotension risk model.

Fits three risk streams for comparison:
  trend  - gradient boosting on trajectory features only
  leak   - same model plus current-MAP features (demonstrates inflation)
  naive  - the pure current-MAP rule risk = clip((MAP_THR+5 - MAP)/20, 0, 1)

Split is by CASE (no within-patient leakage). The predicted per-minute risk stream
from the `trend` model is what feeds the alert policies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

import vitaldb_load as V
from features import LEAK, TREND, case_features, case_labels, fill_features
from labels import MAP_THR


def build_dataset(meta: pd.DataFrame):
    """Assemble X (all features), y, group (caseid) over cached cases."""
    frames, ys, groups = [], [], []
    for cid in meta.caseid:
        df = V.load_cached_case(int(cid))
        if df["map"].notna().sum() < 20:
            continue
        f = case_features(df)
        y = case_labels(df)
        keep = f["map_slope3"].notna().to_numpy()   # need some trajectory context
        frames.append(f[keep])
        ys.append(y[keep])
        groups.append(np.full(keep.sum(), int(cid)))
    X = fill_features(pd.concat(frames, ignore_index=True))
    return X, np.concatenate(ys), np.concatenate(groups)


def _fit(X, y, cols, seed=0):
    m = GradientBoostingClassifier(random_state=seed, max_depth=3, n_estimators=150)
    m.fit(X[cols], y)
    return m


def train(meta: pd.DataFrame, seed: int = 0, max_train_rows: int = 400_000):
    """Train trend/leak models; return models + test AUCs demonstrating leakage.

    Training rows are subsampled to `max_train_rows` to bound memory/time on the
    full cohort; the test split is untouched so AUCs stay honest.
    """
    X, y, g = build_dataset(meta)
    tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=seed).split(X, y, g))
    if len(tr) > max_train_rows:
        tr = np.random.default_rng(seed).choice(tr, max_train_rows, replace=False)
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y[tr], y[te]

    m_trend = _fit(Xtr, ytr, TREND, seed)
    m_leak = _fit(Xtr, ytr, LEAK, seed)
    s_trend = m_trend.predict_proba(Xte[TREND])[:, 1]
    s_leak = m_leak.predict_proba(Xte[LEAK])[:, 1]
    naive_te = np.clip((MAP_THR + 5 - Xte["map_now"]) / 20.0, 0, 1).to_numpy()

    aucs = {
        "trend": roc_auc_score(yte, s_trend),
        "leak": roc_auc_score(yte, s_leak),
        "naive_currentMAP": roc_auc_score(yte, naive_te),
        "event_rate": float(yte.mean()),
    }
    # held-out test arrays for the ROC figure (risk score vs pre-event label)
    roc_data = {"y": yte, "trend": s_trend, "leak": s_leak, "naive": naive_te}
    return {"trend": m_trend, "leak": m_leak}, aucs, roc_data


def risk_streams(model, meta: pd.DataFrame, cols=TREND) -> dict:
    """Per-case per-minute risk aligned to the full 1-min grid (missing -> 0)."""
    out = {}
    for cid in meta.caseid:
        cid = int(cid)
        df = V.load_cached_case(cid)
        f = fill_features(case_features(df))
        out[cid] = model.predict_proba(f[cols])[:, 1]
    return out
