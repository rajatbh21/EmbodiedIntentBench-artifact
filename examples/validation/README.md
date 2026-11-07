# Validation experiment artifacts

Compact, committed outputs of `python -m experiments.validation.study all` and
`python -m experiments.validation.figures`. The protocol, results and limitations
are in [docs/VALIDATION.md](../../docs/VALIDATION.md). Raw per-run receipts stay
in `results/validation/` (gitignored, about 70 MB) and are regenerated
deterministically.

| File | Contents |
|---|---|
| `summary.json` | Parity/leakage receipt, config and source hashes, headline tables |
| `runs.csv` | One row per evaluated run across all experiments (reference, rescoring, reruns, causal ablation, hindsight witnesses) |
| `fig1_intent_success_heatmap.*` | Exp 1: successes per policy x contract (feasible conditions) |
| `exp1_success_rerun.csv`, `exp1_success_rescore.csv` | Exp 1 tables with recording-cluster intervals |
| `exp1_evidence_completion.csv` | Mean confirmed people and timely reports per policy x contract |
| `exp1_ranking_reversals.csv` | Every pairwise ranking reversal across contracts, with margins and per-recording support |
| `exp1_decision_changes.csv` | Runs whose executed actions changed when the contract was a public input |
| `exp1_per_recording.csv` | Successes per recording (the independent unit) |
| `fig6_sensitivity_map.*` | Exp 1c: class-vs-class success differences over the predeclared 144-setting threshold grid |
| `exp1_sensitivity_settings.csv` | All 144 settings: per-policy successes, best policies, material class wins |
| `exp1_sensitivity_classes.csv`, `exp1_sensitivity_pairs.csv`, `exp1_sensitivity_drivers.csv`, `exp1_sensitivity_per_recording.csv` | Exp 1c class wins, pairwise orderings, parameter drivers, per-recording counts |
| `fig2_feasibility_classes.*` | Exp 2: feasibility classes per contract; policy successes vs. verified schedules |
| `exp2_bounds.csv` | Sensor / detectable / model-union bounds per recording x contract |
| `exp2_condition_classes.csv` | Class of each of the 72 conditions per contract, with witness provenance |
| `exp2_policy_gap.csv` | Policy successes and policy-attributable failures |
| `fig3_causal_timeline.*` | Exp 3: capture-by-capture timelines under both protocols |
| `fig4_latency_sweep.*` | Exp 3: processed observations, confirmations, reports, successes vs. latency multiplier |
| `exp3_metrics.csv`, `exp3_protocol_disagreement.csv`, `exp3_success_flips.csv`, `exp3_timelines.json` | Exp 3 tables |
| `fig5_failure_attribution.*` | Exp 4: urgency-aware vs. blind comparison and evidence funnel |
| `exp4_unmet_patterns.csv`, `exp4_urgency_comparison.csv`, `exp4_evidence_funnel.csv`, `exp4_counterfactual_relaxations.csv`, `exp4_per_run.csv` | Exp 4 tables |

The figures use the light-mode reference palette; every figure has a CSV table view.
All counts come from six known recordings on one map. They are descriptive
replay results, not unseen-data or significance claims.
