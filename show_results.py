"""Plain-language view of how well the alerting policies perform, vs the
literature-standard MAP-threshold baseline (MAP<70-75; the RCT uses MAP<72).

  python show_results.py
  python show_results.py --budget 6      # alerts/case-hour cap for the recommendation
  python show_results.py --ref "MAP<72"  # baseline to compare the model against
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

R = os.path.join(os.path.dirname(__file__), "results")


def _fmt(row) -> str:
    return (f"  sens {row['sensitivity']*100:5.1f}%   "
            f"warn {row['warning_time']:4.1f}m   "
            f"PPV {row['ppv']*100:5.1f}%   "
            f"false {row['false_rate']:5.2f}/hr   "
            f"burden {row['burden']:5.2f}/hr")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=6.0)
    ap.add_argument("--ref", default="MAP<72")
    args = ap.parse_args()

    d = pd.read_csv(os.path.join(R, "frontier.csv"))
    meth = d["method"].astype(str) if "method" in d.columns else pd.Series("", index=d.index)
    base = d[meth.str.startswith("MAP<")].copy()      # baseline family
    model = d[~meth.str.startswith("MAP<")].sort_values("burden")  # learned-risk policies

    print("=" * 74)
    print("METRICS:  sens = % events caught in time | warn = median lead (min)")
    print("          PPV = % of alerts that were real | false, burden = per case-hour")
    print("=" * 74)

    print("\nLITERATURE-STANDARD BASELINE  (alert when MAP < T):")
    for _, r in base.sort_values(["tau", "burden"]).iterrows():
        tag = f"{r['method']} (cooldown {int(r['C'])}m)"
        print(f"  {tag:22s}" + _fmt(r))

    print("\nMODEL-RISK POLICIES  (sample, low -> high burden):")
    idx = [0, len(model) // 4, len(model) // 2, 3 * len(model) // 4, len(model) - 1]
    for i in idx:
        print(_fmt(model.iloc[i]))

    # --- fair head-to-head: model vs the reference baseline at matched burden ---
    print("\n" + "=" * 74)
    print(f"HEAD-TO-HEAD vs {args.ref}  (does the model beat it at equal burden?)")
    print("=" * 74)
    ref = base[base["method"] == args.ref]
    ref = ref[ref["C"] == 0]
    if len(ref):
        ref = ref.iloc[0]
        print(f"  baseline {args.ref:8s}" + _fmt(ref))
        cand = model[model["burden"] <= ref["burden"] + 1e-9]
        if len(cand):
            best = cand.sort_values("sensitivity", ascending=False).iloc[0]
            print("  model @<=burden " + _fmt(best))
            dv = (best["sensitivity"] - ref["sensitivity"]) * 100
            print(f"  --> at <= the baseline's burden, model sensitivity is "
                  f"{dv:+.1f} points vs baseline")
        else:
            print("  (no model policy at or below the baseline's burden)")
    else:
        print(f"  reference '{args.ref}' not found in baseline rows")

    # --- recommended model operating point under an alert budget ---
    print("\n" + "=" * 74)
    print(f"RECOMMENDED MODEL POINT  (max sensitivity, burden <= {args.budget}/hr)")
    print("=" * 74)
    feasible = model[model["burden"] <= args.budget]
    if len(feasible):
        best = feasible.sort_values("sensitivity", ascending=False).iloc[0]
        print(_fmt(best))
        print("  knobs:", {k: best[k] for k in ("tau", "m", "C", "trend") if k in best})
    else:
        print(f"  no policy under {args.budget}/hr; raise --budget")

    lk = os.path.join(R, "leakage.csv")
    if os.path.exists(lk):
        s = pd.read_csv(lk, index_col=0).iloc[:, 0]
        print(f"\nrisk-model AUC:  trend={s['trend']:.3f}  leak={s['leak']:.3f}  "
              f"naive={s['naive_currentMAP']:.3f}")


if __name__ == "__main__":
    main()
