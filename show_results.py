"""Plain-language view of the constrained utility-vs-burden frontier, vs the
literature-standard MAP-threshold baseline (MAP<72 RCT).

  python show_results.py
  python show_results.py --budget 4 --false-cap 2 --ppv-min 0.4
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

R = os.path.join(os.path.dirname(__file__), "results")


def _fmt(r) -> str:
    return (f"  util {r['utility']:.2f}  sens {r['sensitivity']*100:5.1f}%  "
            f"warn {r['warning_time']:4.1f}m  PPV {r['ppv']*100:5.1f}%  "
            f"false {r['false_rate']:5.2f}/hr  burden {r['burden']:5.2f}/hr")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=4.0)
    ap.add_argument("--false-cap", type=float, default=2.0)
    ap.add_argument("--ppv-min", type=float, default=0.40)
    ap.add_argument("--gap-max", type=float, default=0.10)
    ap.add_argument("--ref", default="MAP<72")
    args = ap.parse_args()

    d = pd.read_csv(os.path.join(R, "frontier.csv"))
    meth = d["method"].astype(str) if "method" in d.columns else pd.Series("", index=d.index)
    base = d[meth.str.startswith("MAP<")].copy()
    model = d[~meth.str.startswith("MAP<")].sort_values("burden")
    selected = (model[model["selected_on_val"].astype(bool)]
                if "selected_on_val" in model else model.iloc[0:0])

    print("METRICS: util = mean episode utility (detection x timeliness) |",
          "sens/PPV/warn/false/burden as before\n")

    print("LITERATURE-STANDARD BASELINE  (alert when MAP < T):")
    for _, r in base.sort_values(["tau", "burden"]).iterrows():
        print(f"  {r['method']} (cd {int(r['C'])}m)" + _fmt(r))

    print("\nMODEL POLICIES  (sample, low -> high burden):")
    for i in [0, len(model) // 4, len(model) // 2, 3 * len(model) // 4, len(model) - 1]:
        print(_fmt(model.iloc[i]))

    # feasibility filter (constrained frontier)
    def feasible(df):
        return df[(df["false_rate"] <= args.false_cap) & (df["ppv"] >= args.ppv_min)
                  & (df["disparity"] <= args.gap_max)]

    print(f"\nHEAD-TO-HEAD vs {args.ref} (cooldown 5m) at matched burden:")
    ref = base[(base["method"] == args.ref) & (base["C"] == 5)]
    if len(ref):
        ref = ref.iloc[0]
        print("  baseline " + _fmt(ref))
        cand = feasible(selected[selected["burden"] <= ref["burden"] + 1e-9])
        if len(cand):
            best = cand.sort_values("utility", ascending=False).iloc[0]
            print("  model    " + _fmt(best) + "  [feasible]")
            print(f"  --> at <= baseline burden: model utility {best['utility']-ref['utility']:+.2f}, "
                  f"PPV {(best['ppv']-ref['ppv'])*100:+.1f} pts")
        else:
            print("  (no model policy at/below baseline burden)")

    print(f"\nRECOMMENDED (selected on validation; TEST metrics shown; "
          f"false<={args.false_cap}/hr, PPV>={args.ppv_min}, "
          f"gap<={args.gap_max}; burden<={args.budget}/hr):")
    feas = feasible(selected[selected["burden"] <= args.budget])
    if len(feas):
        best = feas.sort_values("utility", ascending=False).iloc[0]
        print(_fmt(best))
        print("  knobs:", {k: best[k] for k in ("tau", "m", "C", "trend") if k in best})
    else:
        print("  no feasible policy under these constraints; relax --false-cap/--budget")
        print("  (the MAP<72 baseline may be the best feasible option)")

    ab = os.path.join(R, "ablations.csv")
    if os.path.exists(ab):
        a = pd.read_csv(ab, index_col=0)
        print("\nMODEL ABLATIONS (test AUC):")
        for name, r in a.iterrows():
            print(f"  {name:12s} AUC {r['auc']:.3f}"
                  + (f"  Brier {r['brier_cal']:.3f}" if "brier_cal" in r
                     and pd.notna(r.get("brier_cal")) else ""))


if __name__ == "__main__":
    main()
