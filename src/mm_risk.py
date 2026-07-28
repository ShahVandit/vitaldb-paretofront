"""Multimodal (numeric time-series + static) risk model for near-term hypotension.

Predicts calibrated P(episode onset in the next 1-15 min) per minute, from
decision-time inputs only. Branches are toggleable so we can run modality ablations
(static-only / numeric-only / fusion). Trained on TRAIN subjects, early-stopped on
VAL, isotonic-calibrated on CAL, reported on TEST (subject-level splits).

Note: this is a cumulative-window RISK model (not a conditional next-minute hazard).
Already-hypotensive / refractory minutes are excluded from training (features.valid).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

import vitaldb_load as V
from features import case_labels

CH = ["map", "sbp", "dbp", "hr", "spo2", "ppf_rate", "rftn_rate", "bis", "etco2"]
L = 15                     # trailing-window length (min)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# static feature spec (decision-time only; NO postop/outcome columns)
STAT_NUM = ["age", "bmi", "asa", "preop_hb", "preop_cr", "preop_na", "preop_k"]
STAT_BIN = ["sex", "emop", "preop_htn", "preop_dm"]   # sex handled as sex==M


# ----------------------------- data -----------------------------------------
def _case_channels(df: pd.DataFrame, mean, std) -> np.ndarray:
    arr = np.stack([df[c].to_numpy(float) if c in df else np.full(len(df), np.nan)
                    for c in CH], axis=1)              # [n, 9]
    mask = np.isnan(arr).astype(np.float32)
    z = np.nan_to_num((arr - mean) / std, nan=0.0).astype(np.float32)
    return np.concatenate([z, mask], axis=1)           # [n, 18]


def _windows(feat: np.ndarray, idxs, L=L) -> np.ndarray:
    n, C = feat.shape
    # The final channel distinguishes left padding from genuine normalized zeros.
    out = np.zeros((len(idxs), L, C + 1), np.float32)
    out[:, :, C] = 1.0
    for i, t in enumerate(idxs):
        w = feat[max(0, t - L + 1):t + 1]
        out[i, -len(w):, :C] = w
        out[i, -len(w):, C] = 0.0
    return out


def fit_norm(caseids):
    """Per-channel mean/std over TRAIN valid minutes."""
    s, s2, k = np.zeros(len(CH)), np.zeros(len(CH)), np.zeros(len(CH))
    for cid in caseids:
        df = V.load_cached_case(int(cid))
        for j, c in enumerate(CH):
            v = df[c].to_numpy(float) if c in df else np.array([])
            v = v[np.isfinite(v)]
            s[j] += v.sum(); s2[j] += (v * v).sum(); k[j] += v.size
    mean = np.where(k > 0, s / np.maximum(k, 1), 0.0)
    var = np.where(k > 0, s2 / np.maximum(k, 1) - mean ** 2, 1.0)
    std = np.sqrt(np.maximum(var, 1e-6))
    return mean, std


def build_static(train_ids, all_ids):
    """Static-feature matrix for all_ids, using TRAIN medians/vocab."""
    cases = V.get_cases().set_index("caseid")
    tr = cases.loc[cases.index.intersection([int(i) for i in train_ids])]
    med = {c: pd.to_numeric(tr[c], errors="coerce").median() for c in STAT_NUM}
    top_op = tr["optype"].value_counts().head(8).index.tolist()
    rows, miss_cols = {}, [f"{c}_miss" for c in STAT_NUM]
    for cid in all_ids:
        r = cases.loc[int(cid)]
        vec = {}
        for c in STAT_NUM:
            x = pd.to_numeric(pd.Series([r.get(c)]), errors="coerce").iloc[0]
            vec[c] = med[c] if pd.isna(x) else x
            vec[f"{c}_miss"] = float(pd.isna(x))
        vec["sex"] = float(str(r.get("sex")).upper().startswith("M"))
        for c in ["emop", "preop_htn", "preop_dm"]:
            vec[c] = float(pd.to_numeric(pd.Series([r.get(c)]), errors="coerce").fillna(0).iloc[0])
        for op in top_op:
            vec[f"op_{op}"] = float(r.get("optype") == op)
        vec["op_other"] = float(r.get("optype") not in top_op)
        rows[int(cid)] = vec
    X = pd.DataFrame(rows).T
    # normalize numeric static columns with train mean/std
    for c in STAT_NUM:
        m, sd = X.loc[[int(i) for i in train_ids], c].mean(), X.loc[[int(i) for i in train_ids], c].std() or 1.0
        X[c] = (X[c] - m) / (sd if sd else 1.0)
    return X.astype(np.float32)


def build_samples(caseids, norm, static_X, all_minutes=False):
    """Pooled samples. all_minutes=False -> only VALID (labeled) minutes for training;
    True -> every minute (for producing risk streams)."""
    Xn, Xs, y, groups, per_case_idx = [], [], [], [], {}
    mean, std = norm
    for cid in caseids:
        cid = int(cid)
        df = V.load_cached_case(cid)
        yy, valid = case_labels(df)
        feat = _case_channels(df, mean, std)
        idx = np.arange(len(df)) if all_minutes else np.where(valid)[0]
        if len(idx) == 0:
            per_case_idx[cid] = (idx, None)
            continue
        Xn.append(_windows(feat, idx))
        Xs.append(np.repeat(static_X.loc[cid].to_numpy()[None, :], len(idx), 0))
        y.append(yy[idx].astype(np.float32))
        groups.append(np.full(len(idx), cid))
        per_case_idx[cid] = (idx, None)
    if not Xn:
        return None
    return (np.concatenate(Xn), np.concatenate(Xs).astype(np.float32),
            np.concatenate(y), np.concatenate(groups), per_case_idx)


# ----------------------------- model ----------------------------------------
class MMRisk(nn.Module):
    def __init__(self, n_ch, static_dim, use_numeric=True, use_static=True, h=64):
        super().__init__()
        self.use_numeric, self.use_static = use_numeric, use_static
        d = 0
        if use_numeric:
            self.gru = nn.GRU(n_ch, h, batch_first=True)
            d += h
        if use_static:
            self.smlp = nn.Sequential(nn.Linear(static_dim, h), nn.ReLU())
            d += h
        self.head = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, xn, xs):
        parts = []
        if self.use_numeric:
            parts.append(self.gru(xn)[0][:, -1])
        if self.use_static:
            parts.append(self.smlp(xs))
        return self.head(torch.cat(parts, 1)).squeeze(1)


def _predict(model, Xn, Xs, bs=8192):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(Xn), bs):
            xn = torch.tensor(Xn[i:i + bs], device=DEV)
            xs = torch.tensor(Xs[i:i + bs], device=DEV)
            out.append(torch.sigmoid(model(xn, xs)).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def train_model(train, val, use_numeric, use_static, epochs=8, bs=4096, seed=0):
    torch.manual_seed(seed)
    Xn, Xs, y = train[0], train[1], train[2]
    model = MMRisk(Xn.shape[2], Xs.shape[1], use_numeric, use_static).to(DEV)
    pos = max(y.sum(), 1); pw = torch.tensor([(len(y) - pos) / pos], device=DEV)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    best_auc, best_state = -1, None
    idx = np.arange(len(y))
    for ep in range(epochs):
        model.train(); np.random.default_rng(ep).shuffle(idx)
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            opt.zero_grad()
            logit = model(torch.tensor(Xn[b], device=DEV), torch.tensor(Xs[b], device=DEV))
            loss = lossf(logit, torch.tensor(y[b], device=DEV))
            loss.backward(); opt.step()
        pv = _predict(model, val[0], val[1])
        auc = roc_auc_score(val[2], pv) if len(np.unique(val[2])) > 1 else 0.5
        if auc > best_auc:
            best_auc, best_state = auc, {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return model


def _cal_stats(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lr = LogisticRegression().fit(np.log(p / (1 - p)).reshape(-1, 1), y)
    return {"slope": float(lr.coef_[0, 0]), "intercept": float(lr.intercept_[0])}


def evaluate(model, cal, test):
    """Return metrics + fitted isotonic calibrator (fit on CAL, reported on TEST)."""
    pc = _predict(model, cal[0], cal[1])
    iso = IsotonicRegression(out_of_bounds="clip").fit(pc, cal[2])
    pt = _predict(model, test[0], test[1])
    pt_cal = iso.predict(pt)
    m = {
        "auc": roc_auc_score(test[2], pt) if len(np.unique(test[2])) > 1 else np.nan,
        "brier_raw": brier_score_loss(test[2], pt),
        "brier_cal": brier_score_loss(test[2], pt_cal),
    }
    m.update({f"cal_{k}": v for k, v in _cal_stats(test[2], pt_cal).items()})
    return m, iso


def risk_streams(model, iso, caseids, norm, static_X):
    """Per-case per-minute calibrated risk over ALL minutes (scoring suppresses
    excluded minutes downstream)."""
    out = {}
    for cid in caseids:
        cid = int(cid)
        df = V.load_cached_case(cid)
        feat = _case_channels(df, *norm)
        idx = np.arange(len(df))
        Xn = _windows(feat, idx)
        Xs = np.repeat(static_X.loc[cid].to_numpy()[None, :], len(idx), 0).astype(np.float32)
        out[cid] = iso.predict(_predict(model, Xn, Xs))
    return out


def run(splits, epochs=8):
    """Train the three modality variants; return (fusion_model, iso, norm, static_X,
    ablation_metrics_df). Splits: dict with train/val/cal/test caseid lists."""
    norm = fit_norm(splits["train"])
    all_ids = sum((splits[k] for k in ("train", "val", "cal", "test")), [])
    static_X = build_static(splits["train"], all_ids)
    tr = build_samples(splits["train"], norm, static_X)
    va = build_samples(splits["val"], norm, static_X)
    ca = build_samples(splits["cal"], norm, static_X)
    te = build_samples(splits["test"], norm, static_X)

    metrics, models = {}, {}
    for name, (un, us) in {"static": (False, True), "numeric": (True, False),
                           "fusion": (True, True)}.items():
        mdl = train_model(tr, va, un, us, epochs=epochs)
        m, iso = evaluate(mdl, ca, te)
        metrics[name] = m
        models[name] = (mdl, iso)
    fusion_model, fusion_iso = models["fusion"]
    return fusion_model, fusion_iso, norm, static_X, pd.DataFrame(metrics).T
