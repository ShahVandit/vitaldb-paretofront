# Intraoperative Hypotension Alert-Policy Optimizer

A decision-layer prototype: given a per-minute hypotension **risk stream** for each
surgical case, compare **alerting policies** on a multi-objective Pareto frontier that
balances detection, lead time, and clinician alert burden. Built on VitalDB
(open-access perioperative data).

This is **not** a prediction project. The predictor is deliberately simple; the
contribution is the decision layer on top of it.

**RL scope.** The decision layer is (1) an interpretable multi-objective policy
frontier and (2) an **offline contextual-bandit** alert policy learned against a
composite reward and evaluated by *exact* replay — valid because alerting does not
change the patient trajectory. Treatment-level offline RL (learning a vasopressor
policy) is deliberately **out of scope**: the logged infusion actions are too sparse
(phenylephrine n≈127) to support it, and saying so is part of the deliverable
(`reports/memo.md`).

## Why the frontier is not an ROC curve

An ROC sweeps a single threshold on a risk score and plots TPR vs FPR over i.i.d.
samples at one instant. Here the object of study is a **temporal policy** over each
case's minute-by-minute stream, scored at the **event/case** level with timing. The
policy has parameters an ROC cannot represent, and two policies at the *same*
sensitivity/false-alarm point can differ on lead time and burden. The frontier
therefore lives in >=3 dimensions with at least one axis (warning time, burden) that
no ROC operating point encodes.

### Objectives (axes)
| Axis | Definition | Direction |
|---|---|---|
| Event sensitivity | fraction of hypotension events (MAP<65 for >=1 min) with >=1 alert in the actionable window `[onset-Wmax, onset-Wmin]` | higher better |
| Median warning time | for detected events, minutes of lead between first in-window alert and onset | higher better |
| False-alert rate | alerts not linked to any event, per case-hour | lower better |
| Alert burden | total alerts (incl. repeats), per case-hour | lower better |
| Subgroup disparity | max-min sensitivity gap across subgroups (ASA / age / sex) | lower better |

Sensitivity vs false-alert alone *is* an ROC. Adding **warning time** and **burden**
(both driven by temporal policy structure, not by the operating point) is what makes
this a genuine Pareto problem.

### Policy parameterization
A policy `π = (τ, m, C, trend)` maps a risk stream to alert times:
- `τ`  risk threshold to consider firing
- `m`  persistence: risk must exceed `τ` for `m` consecutive minutes (debounce)
- `C`  cooldown: no new alert for `C` minutes after one fires
- `trend`  optional early trigger: fire if risk is rising faster than `trend`/min even below `τ`

`m` and `C` change burden and warning time **without** moving the ROC point, which is
the crux. Endpoint `m=1, C=0, trend=off` recovers the naive per-sample threshold rule
(the ROC baseline) as a single degenerate policy inside the space.

## Layout
```
src/policies.py   policy parameterization -> alert times
src/metrics.py    event matching + the 5 objectives
src/frontier.py   Pareto extraction + bootstrap confidence bands
src/synth.py      synthetic cases so the pipeline runs before VitalDB is wired
src/vitaldb_load.py  VitalDB numeric-track loader (stub: resampled MAP/HR/SpO2)
run_demo.py       end-to-end on synthetic data -> results/frontier.csv + plot
```

## Run
```
pip install -r requirements.txt
python run_demo.py            # synthetic; swap in vitaldb_load once credentials/API ready
```

## Honest limitations (policy-readiness notes)
- **Single-center** (SNUH): external validity untested; a second site (INSPIRE/MOVER)
  is the natural replication. Do not claim generalization.
- **HPI/leakage trap**: IOH "prediction" can be near-trivial when features include
  current MAP (the label leaks via baseline). We evaluate every policy against a
  naive "current-MAP" comparator and use time-anchored splits; performance is
  reported *relative* to that baseline, not in absolute AUC terms.
- **Not off-policy evaluation**: alerting does not alter the logged trajectory, so
  policy scores are exact retrospective replay, not OPE. True OPE would require the
  logged clinician *treatment* actions (vasopressor/fluid) and is scoped separately.

src/vitaldb_load.py — gets the data in (done)
src/labels.py ← next: defines a hypotension event
src/policies.py — what an alert policy is (the τ, m, C, trend knobs)
src/metrics.py — how a policy is scored on the 5 objectives
src/frontier.py — picks the non-dominated policies (the Pareto part)
src/features.py then src/risk_model.py — builds the risk score (and the leakage check)
src/bandit.py — the RL part (learns an alerting policy)
src/robustness.py — subgroup / missing-signal / threshold checks
run_pipeline.py — the glue that calls everything in order
plot_results.py — makes the figures