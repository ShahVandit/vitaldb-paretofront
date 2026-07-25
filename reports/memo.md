# Policy-Readiness Memo: Intraoperative Hypotension Alerting

**Question.** Can an alerting policy for intraoperative hypotension (MAP < 65 mmHg
for ≥1 min) be chosen that balances detection, lead time, and clinician alert
burden — and is the data adequate to *learn* or *evaluate* such policies?

**Data.** VitalDB, ~3,700 non-cardiac surgical cases with an arterial line
(`Solar8000/ART_MBP`). Analysis window = anesthesia period (`anestart..aneend`);
arterial MAP cleaned (physiologic clipping + per-minute median despiking) and
resampled to a 1-minute grid. NIBP (cuff) cases are excluded: intermittent sampling
cannot support a sustained-minute event definition.

## Findings

**1. Genuine early warning is hard; most apparent "prediction" is leakage.**
A near-term-hypotension model using *trajectory features only* (MAP slope/variability,
HR/SpO2 trends) reaches test AUC ≈ 0.56–0.66. Adding the *current MAP* level jumps it
to ≈ 0.77–0.80, and a naive "current MAP" rule alone scores ≈ 0.67 — i.e. it beats the
trend model. This is the documented HPI trap: models "predict" impending hypotension
largely by detecting that MAP is *already* falling. **Consequence:** we report policy
performance against the naive current-MAP comparator, use time-anchored case splits,
and treat absolute AUC as uninformative. A high-AUC "predictor" here is not evidence of
actionable early warning.

**2. The policy choice is genuinely multi-objective, not an ROC.**
Across the policy grid we find many pairs of policies at **equal sensitivity and
false-alert rate** that differ substantially in **warning time and per-case alert
burden**, because persistence (`m`) and cooldown (`C`) reshape the alert stream in
time without moving the ROC operating point. Example: at ~0.99 sensitivity, adding
cooldown+persistence cuts burden from ~50 to ~11 alerts/case-hour. Sensitivity-vs-
false-alarm alone would be an ROC; warning time and burden are the axes that make this
a real Pareto problem, and they are where the clinically-relevant trade-off lives.

**3. A learned policy is valid to *evaluate exactly* here — but only for alerting.**
An alerting action does **not** change the patient trajectory, so any alert policy's
score on the logged data is its *true* score: off-policy evaluation is exact, with no
confounding or importance-sampling variance. We exploit this to learn an offline
contextual-bandit alert policy against a composite reward and evaluate it by exact
replay; it traces a clean operating curve on the same Pareto axes.

## Where the data is *not* adequate (the important negative)

**Treatment-level offline RL is not justified on this dataset.** Doing RL on the
clinician's *response* to hypotension (vasopressors/fluids) would require dense, timed
action logs. VitalDB's infusion tracks are sparse: phenylephrine n = 127,
norepinephrine n = 88 (of ~3,700), and bolus dosing is largely uncaptured. With actions
this sparse and unconfounded timing unavailable, off-policy value estimates for
treatment policies would be high-variance and bias-prone. **Recommendation:** do not
attempt dynamic-treatment-regime RL here; alerting (non-interfering, exactly evaluable)
is the appropriate decision problem for this data.

## Limitations
- **Single-center** (Seoul National University Hospital). External validity is untested;
  the frontier and its feasible region may shift at another site.
- Risk model is deliberately simple (the contribution is the decision layer, and the
  role consumes risk scores produced by others).
- Events defined on arterial MAP only; cuff-only cases are out of scope.

## Recommended next steps before any deployment
1. **External replication** on a second site (INSPIRE / MOVER, perioperative,
   credentialed) — report how the Pareto frontier moves.
2. **Capacity-constrained operating point**: choose the policy under an explicit
   alerts-per-hour budget rather than a fixed threshold.
3. **Silent-mode prospective evaluation**: run the chosen policy without firing, confirm
   burden and lead-time estimates hold live, then enable.
4. **Monitoring/failure points**: sensor dropout (see missing-signal analysis) and
   subgroup disparity by ASA class (see subgroup frontiers) are the two most likely
   real-world failure modes.

*Headline numbers in this memo are refreshed from the full-cohort run; see
`results/frontier.csv`, `results/bandit_sweep.csv`, and `results/*.csv`.*
