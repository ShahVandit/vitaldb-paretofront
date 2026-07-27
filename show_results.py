"""Plain-language view of how well the alerting policies actually perform.

Reads the CSVs written by run_pipeline.py and prints a few real operating points,
the naive current-MAP baseline, and one recommended point under an alert budget.
No plotting -- just the numbers, explained.

  python show_results.py
  python show_results.py --budget 6      # alerts/case-hour cap for the recommendation
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

R = os.path.join(os.path.dirname(__file__), "results")
COLS = ["sensitivity", "warning_time", "false_rate", "burden"]


def _fmt(row) -> str:
    return (f"  sensitivity {row['sensitivity']*100:5.1f}%   "
            f"warning {row['warning_time']:4.1f} min   "
            f"false {row['false_rate']:5.2f}/hr   "
            f"burden {row['burden']:5.2f}/hr")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=6.0,
                    help="max alerts/case-hour for the recommended operating point")
    args = ap.parse_args()

    d = pd.read_csv(os.path.join(R, "frontier.csv")).sort_values("burden")

    print("=" * 70)
    print("WHAT THE NUMBERS MEAN (per policy)")
    print("=" * 70)
    print("  sensitivity  = % of real hypotension events we alerted on IN TIME")
    print("  warning      = median minutes of lead before the event")
    print("  false        = false alarms per case-hour")
    print("  burden       = total alerts per case-hour (real + false)")
    print()

    print("=" * 70)
    print("SAMPLE OPERATING POINTS  (low -> high alert burden)")
    print("=" * 70)
    idx = [0, len(d) // 4, len(d) // 2, 3 * len(d) // 4, len(d) - 1]
    for i in idx:
        print(_fmt(d.iloc[i]))
    print()

    if "method" in d.columns and (d["method"] == "current_map").any():
        print("NAIVE CURRENT-MAP RULE  (fires when MAP already low; ~no early warning)")
        print(_fmt(d[d["method"] == "current_map"].iloc[0]))
        print()

    print("=" * 70)
    print(f"RECOMMENDED POINT  (highest sensitivity with burden <= {args.budget}/hr)")
    print("=" * 70)
    feasible = d[d["burden"] <= args.budget]
    if len(feasible):
        best = feasible.sort_values("sensitivity", ascending=False).iloc[0]
        print(_fmt(best))
        knobs = {k: best[k] for k in ("tau", "m", "C", "trend") if k in best}
        print("  policy knobs:", knobs)
    else:
        print(f"  no policy stays under {args.budget} alerts/hr; raise --budget")
    print()

    lk = os.path.join(R, "leakage.csv")
    if os.path.exists(lk):
        s = pd.read_csv(lk, index_col=0).iloc[:, 0]
        print("RISK-MODEL AUC (context):",
              f"trend={s['trend']:.3f}  leak={s['leak']:.3f} "
              f"naive={s['naive_currentMAP']:.3f}")


if __name__ == "__main__":
    main()
