# Benchmark validation experiments

Four experiments ask what this benchmark reveals that accuracy, latency, energy
or streaming-perception evaluation alone would not:

1. **Intent-dependent ranking.** Does the mission contract change which runtime strategy is best?
2. **Feasibility and hindsight bounds.** Is a failure caused by the policy, or was the mission infeasible?
3. **Causal execution ablation.** Does modelling blocked observations change the evaluation?
4. **Failure attribution.** Why doesn't better timeliness or energy translate into mission success?

They all reuse the released 648-cell replay (`tools/replay.py`, `data/replay/`). There
is no new flight, no new neural inference, and no retuning of any policy.
Code is in `experiments/validation/`, committed results and figures are in
`examples/validation/`, and tests are in `tests/test_validation_experiments.py`.

## Summary of findings

| Experiment | Finding |
|---|---|
| 1. Intent ranking | **No meaningful ranking reversal on this workload, and the predeclared 144-setting threshold map confirms it.** Contracts change absolute success, but not which *class* of strategy wins. The fast models (YOLO11s, CABiNet, fastest-feasible) score 0 under every Experiment 1 contract and never materially beat an accurate or adaptive strategy in any of the 144 threshold settings. Their closest approach is sightings-only with zero false alarms: YOLO11s then leads remote Mask R-CNN by 6 conditions but fails one recording entirely, and is still 4 behind the best adaptive router. Adaptive routing materially beats the accurate fixed controls in 128 of 144 settings and never loses to them. All 12 rerun reversals (18 rescored) among the five contracts are 1-condition swaps between near-tied adaptive variants. |
| 2. Feasibility | **No condition is physically infeasible, but 12/72 are certified infeasible for this model pool.** All on `sector_revisit/sector_3`: every person is visible on 11 to 16 captures, yet no model confirms two of them. Under the reference contract, 43 conditions have a successful released policy. 8 more have a verified hindsight schedule that no released policy found, and 9 are unresolved. The best policy fails 8 of 51 demonstrated-feasible conditions. |
| 3. Causal ablation | **Ignoring blocked observations inflates success and flips a ranking.** At 1× latency the no-skip ablation succeeds in 255 runs against 218 for causal replay (37 more, 0 fewer). Under causal replay, urgency-aware quality routing beats fixed remote Mask R-CNN 41 to 38. Under no-skip the order flips to 45 vs 48, and remote Mask R-CNN ties all three other adaptive routers. Going from 1× to 2× latency, causal success drops 218 to 186, and all 33 runs that flip to failure lost a confirming observation to blocking. Under no-skip, success is unchanged between 1× and 2× (255 runs at both multipliers). |
| 4. Failure attribution | **Urgency awareness removes deadline misses but not the binding requirement.** On the transfer split, urgency-aware risk routing cuts request deadline misses from 90 to 0 and task energy from 198 J to 168 J, for 25 vs 24 successes. Every one of its 23 remaining failures misses report recall, and 18 also miss repeated-evidence confirmation. None fails on battery, memory, communication or execution. The per-person funnel puts the main loss at perception: a person visible on a processed capture is never matched. |

None of these experiments shows that adaptive policies win in general, and none
is meant to. They show that the benchmark separates physical, model-pool,
execution-semantics and policy causes of mission failure.

## Units, denominators and independence

- **Recording** (the unit used for clustered analysis): 6 Gazebo recordings, 42 s each.
  They come from 3 route families (`shoreline_loop`, `wide_area_search`,
  `sector_revisit`) × 2 focal sectors (3, 7), all on **one Lake map**.
- **Condition**: recording × urgency schedule (3) × link (2) × battery (2) = **72**.
  Schedules, links and batteries are replay conditions of the same recording. They
  are not independent missions.
- **Run**: condition × policy (9) = 648 per contract, protocol and latency multiplier.
- The release labels shoreline recordings `development` (24 conditions) and
  the other four `transfer` (48). All six recordings and their earlier outcomes
  were already known, so neither split is unseen data.
- Uncertainty: each table reports per-recording successes
  (`exp1_per_recording.csv`) and the number of recordings with any success. The
  `ci_lo`/`ci_hi` columns are percentile bootstraps over recordings (seed
  20261103, 10,000 resamples). **With only 5 or 6 clusters they understate
  uncertainty; treat them as descriptive.** No significance claims are made.

## Integrity checks (all pass)

| Check | Result |
|---|---|
| Instrumented runtime vs `expected_rows.json` | 648/648 rows identical on every scientific field |
| Parametrised scorer vs frozen v2 outcome | 648/648 identical checks, failed constraints and window metrics |
| Downstream execution failures in reference runs | 0 |
| Leakage audit (every policy input, every prediction read) | 9,572 decisions and 8,888 prediction reads. No target identity or position in the intent, only the current urgency update, the link snapshot at decision time, and the current capture's RGB. Predictions are read only for the selected action, current capture, once each. |
| Calibration vs evaluation recordings | Case IDs disjoint (training: sectors 2/6 and `offset_14` clips). They still share the map and route families. |
| Determinism | Two complete regenerations produced byte-identical stage outputs |
| Hindsight witnesses | 25/25 re-executed in fresh processes with identical rows |

`examples/validation/summary.json` records the config hash, the hashes of every
validation module, and the parity receipt.

## Experiment 1: intent-dependent strategy ranking

**Contracts** (`experiments/validation/config.json`, frozen before any
non-reference outcome was computed). Each is a parameter setting of the existing
contract schema; no new requirement type is introduced.

| Contract | Definition | Source |
|---|---|---|
| `confirm` | Reference: ≥2/3 people, ≥2 observations spanning ≥2 s, report by 44 s | Frozen v2 contract |
| `survey` | ≥2/3 people, a single observation suffices | Coverage relaxation |
| `confirm_strict` | ≥3 observations spanning ≥4 s | The repository's "verified inspection" rule |
| `revisit` | Confirm ≥2/3 in [0,18) **and** again from fresh evidence in [18,36) | The rapid-survey "fresh revisit" rule |
| `time_critical` | Phase and report deadlines at 36 s (end of pass) instead of 44 s | Deadline tightening |

**Two analyses.**
- **(a) Rescoring** applies each contract's evaluator to the identical 648
  reference histories.
- **(b) Reruns** supply each contract as public input to policy *and* pipeline;
  the report verifier reads `min_hits`, span and phase windows. Each contract is
  scored only on recordings that its model-union bound does not certify
  infeasible (Experiment 2).

**Amendment (recorded in config).** The reporting pipeline enforces a
two-observation floor
(`experiments/mission_pipeline_routing/runtime.py:119`), so `survey` cannot be
a runtime input. It is rescored only. The runtime rejected it before any survey
outcome had been inspected, and the definition is unchanged.

**Successes over feasible conditions (reruns; survey rescored)**: see `fig1_intent_success_heatmap.png`.

| Policy | Survey | Confirm | Strict | Revisit | Time-critical |
|---|---:|---:|---:|---:|---:|
| Fixed local YOLO11s / CABiNet / fastest | 0/72 | 0/60 | 0/60 | 0/36 | 0/60 |
| Fixed local Mask R-CNN | 12/72 | 12/60 | 8/60 | 0/36 | 12/60 |
| Fixed remote Mask R-CNN | 38/72 | 38/60 | 34/60 | 12/36 | 38/60 |
| Quality, urgency-blind | 42/72 | 42/60 | 37/60 | 12/36 | 42/60 |
| Quality, urgency-aware | 41/72 | 41/60 | 37/60 | 13/36 | 41/60 |
| Quality+risk, urgency-blind | 42/72 | 42/60 | 37/60 | 12/36 | 42/60 |
| Quality+risk, urgency-aware | 43/72 | 43/60 | 36/60 | 13/36 | 43/60 |

**Why the rankings do not change.** These are properties of the workload, not
of the policies:

- **Survey ≡ confirm.** Captures are 2 s apart, so any two distinct observations
  already span 2 s. Reports also require two observations at the pipeline, so
  dropping the evaluator's repeated-evidence rule alone changes no outcome.
- **Time-critical ≡ confirm.** In all 218 successful reference runs, the second
  focal report arrives by 35.8 s (median 19.2 s). The tighter deadline changes
  13 urgency-blind and 2 urgency-aware schedules, but no outcome.
- **Fast models fail everywhere.** Their per-capture recall is too low: fixed
  YOLO11s matches 120 of 216 person-runs at least once but confirms only 24.
  Under no contract can they confirm two of three people. Consequently, these
  contracts do not produce a fast-model advantage with this model pool.
- **Strict and revisit contracts reduce success roughly uniformly.** Revisit is
  certified infeasible for this model pool on 3 of 6 recordings.

**Reversals.** 12 pairwise reversals in reruns, 18 in rescoring
(`exp1_ranking_reversals.csv`). Every one is between two adaptive variants, with
margin 1 condition, and holds on at most 1 recording.

**Contract as input.** The contract changed executed actions in only 54 of 1,944
rerun cells (`exp1_decision_changes.csv`): risk routing under revisit in 24, and
the quality/risk routers under time-critical in 30. Only one rerun outcome differs
from its rescored counterpart. Raising `min_hits` and span (`confirm_strict`)
changed no executed action for any policy. Only phase windows (risk routing) and
deadlines (the shared resource guard) moved decisions.

**Interpretation.** The benchmark machinery for intent variation works, and it
reports honestly that this workload does not separate strategies by intent.
Demonstrating intent-dependent ranking needs a model pool or trajectories where a
fast model can confirm targets. It also needs policies that read evidence
requirements. Retuning contracts to manufacture a reversal would be invalid.

### Experiment 1c: threshold sensitivity map

Because the contract thresholds are authored rather than measured, the
question is whether *any* reasonable setting reverses the ranking, not whether
the five chosen contracts do. The grid, the materiality rule, the strategy
classes and the headline statistics were committed
(`8f5d546`, `config.json` → `sensitivity_map`) before any setting was scored.

**Grid** (144 settings), applied to the existing contract schema:

| Parameter | Values | Reference |
|---|---|---|
| People required (of 3) | 1, 2, 3 | 2 |
| Sightings per person | 1, 2, 3 (span 2·(h−1) s) | 2 |
| False-alarm cap (raw false components) | 0, 1, 2, none | 2 |
| Report required | yes (with reference report caps), no (sightings only) | yes |
| Deadline | 36, 44 s | 44 s |

**Method.** Evaluation-only rescoring of the 648 reference histories. Sightings-only
missions cannot be runtime inputs (two-observation report floor), so policies
cannot adapt to them; the map ranks fixed execution histories. Three grid points
must reproduce known results exactly, and do: the reference outcome of all 648
runs, the time-critical rescoring and the strict-confirm rescoring.

**Materiality** (declared): A materially beats B only if A has ≥3 more successes
(of 72) *and* no recording favours B. Raw differences are reported alongside in
`exp1_sensitivity_pairs.csv`.

**Results** (`fig6_sensitivity_map.png`, `exp1_sensitivity_*.csv`):

| Best of class … materially beats every member of … | Settings (of 144) |
|---|---:|
| Fast → accurate fixed | 0 |
| Fast → adaptive | 0 |
| Accurate fixed → fast | 72 |
| Adaptive → fast | 92 |
| Adaptive → accurate fixed | 128 |
| Accurate fixed → adaptive | 0 |

- **The fast model's best region is sightings-only with a zero false-alarm cap**
  (4 settings: 1 or 2 people, either deadline). YOLO11s produces no raw false
  components in any of the 72 conditions. Fastest-feasible, which behaves
  identically here, is the only other strategy that does.
  There it raw-beats fixed remote Mask R-CNN (60 vs 54 with one person required;
  48 vs 42 with two), local Mask R-CNN, and three of the four adaptive routers.
  It still trails urgency-aware risk routing by 4.
- **That lead is not material, and the per-recording data show why.** A fixed
  local model's run doesn't depend on link, battery or urgency, so YOLO11s
  scores all-or-nothing per recording: 12/12 or 0/12. Its 72 conditions amount
  to 6 outcomes. With one person required it matches or beats remote Mask R-CNN
  on 5 recordings and scores 0 vs 4 on `shoreline_loop/sector_3`.
- **Both conditions are necessary for the fast model to come close.** Allowing
  even one false alarm widens the gap from −4 to −11, because the other
  strategies' occasional false detections stop counting against them. Requiring
  reports puts the gap at −12 or worse in every setting: YOLO11s's sightings
  rarely pair up into the two observations the report pipeline needs.
- **Fast models never raw-beat urgency-aware risk routing** in any of the 144 settings.
- **The adaptive advantage over accurate fixed controls is robust to thresholds.**
  It is material in 128 settings and never reverses materially. Of the 16
  settings without a material advantage, 14 require 3 of 3 people with 3
  sightings (difference 0 or −1). The other 2 are 2 people, 3 sightings, cap 1
  at 36 s, where adaptive leads by 2, below the margin.
- **The deadline barely matters.** Moving it to 36 s changes counts only for
  fixed remote Mask R-CNN (−4, in 12 of 72 parameter combinations) and the
  urgency-blind routers (−1, in 8).

**Interpretation.** Over this grid, no authored threshold setting makes the fast
model the best strategy on this workload. The one region where it comes close is
a precision-first sighting mission, and even there its advantage is recording-
dependent rather than mission-level. A defensible "fast model wins" result would
need a different workload, not different thresholds:
- a denser capture cadence, so slow models lose more observations (YOLO11s
  processes 18/18 captures vs 13.6 for remote Mask R-CNN at the current 2 s
  cadence), or
- targets visible for fewer captures, or
- a sighting-level mission that the runtime can actually express.

## Experiment 2: mission feasibility and hindsight bounds

**Physical-opportunity bounds** (`bounds.py`): zero latency, every capture processed.
- *Sensor*: any visible GT pixel counts as an observation.
- *Detectable*: ≥4 px, the evaluator's minimum component size.
- *Model union*: every candidate model on every capture, matched by the unchanged
  evaluator, with caps relaxed.

Each bound dominates every legal schedule, so failing one is a sound
infeasibility certificate.

**Execution-constrained hindsight** (`hindsight.py`): a GT-aware local search over
per-capture action scripts (6 actions + abstain).
- Every candidate runs through the unchanged causal runtime with the contract as
  public input.
- Seeds: every released policy's schedule and every constant schedule.
- Budget: ≤600 evaluations, ≤3 sweeps.
- Labels: a success is a *demonstrated feasible schedule*; a failure is
  *unresolved*. It is neither optimal nor a policy.

| Class (conditions of 72) | Survey* | Confirm | Strict | Revisit | Time-critical |
|---|---:|---:|---:|---:|---:|
| A: physically infeasible | 0 | 0 | 0 | 0 | 0 |
| B: model pool certified infeasible | 0 | 12 | 12 | 36 | 12 |
| B: unresolved | 29 | 9 | 16 | 21 | 9 |
| C: hindsight schedule only | n/a | 8 | 7 | 2 | 8 |
| C: a released policy succeeds | 43 | 43 | 37 | 13 | 43 |

\* Survey: no hindsight search (not a runtime input); its C class counts rescored
released histories.

- **Model-pool infeasibility.** On `sector_revisit/sector_3`, persons 1 and 2
  are visible on 16 and 15 captures (77 to 371 px), but across all three models
  each is matched exactly once (YOLO11s at 14 s). No schedule can confirm two
  people. These 12 conditions are excluded from the Experiment 1 denominators.
  No released policy succeeds in any certified-infeasible condition, which is the
  required soundness check.
- **Where the headroom is.** All 17 confirm conditions without a released-policy
  success but not certified infeasible (8 witness-only, 9 unresolved) use the
  degraded link under a schedule that becomes urgent. The 8 witness schedules
  are heterogeneous mixes that no released policy produces. One example: 8 local
  CABiNet calls, 2 local and 1 remote Mask R-CNN calls, and 1 remote YOLO11s
  call. Five of the 8 also abstain on 1 to 3 captures (`hindsight.jsonl`).
- **Policy gap (confirm, `exp2_policy_gap.csv`).** Of 51 demonstrated-feasible
  conditions, successes are:

  | Policy | Successes / 51 |
  |---|---:|
  | Quality+risk, urgency-aware | 43 |
  | Quality, urgency-blind | 42 |
  | Quality+risk, urgency-blind | 42 |
  | Quality, urgency-aware | 41 |
  | Fixed remote Mask R-CNN | 38 |
  | Fixed local Mask R-CNN | 12 |
  | Fast fixed models | 0 |

  These failures are policy-attributable: a valid schedule exists under the same
  costs and deadlines.

Resolves the warehouse concern: certified infeasibility is reported and excluded,
not blamed on the router.

## Experiment 3: causal execution ablation

**Protocols** (`runtime.py`):
- **Causal** (frozen): blocking computation makes intervening captures unavailable.
- **No-skip** (ablation, physically unrealizable): every capture is processed.
  Latency still sets completion time, deadline admission and energy; requests and
  model residency stay sequential. The serial downstream tracker takes completed
  predictions FIFO in capture order, its stage deadlines measured from dequeue.
  Concurrent memory is not charged.

**Latency multipliers** 0.5/1/2/4× scale every action's configured
inference/service latency, and local compute energy under the declared 15 W
model. Predictions, load costs, network transfer and downstream budgets stay
fixed. In total: 648 cells × 2 protocols × 4 multipliers = 5,184 runs.

| Multiplier | Both succeed | Only no-skip succeeds | Only causal succeeds |
|---:|---:|---:|---:|
| 0.5× | 272 | 12 | 0 |
| 1× | 218 | 37 | 0 |
| 2× | 185 | 70 | 1 |
| 4× | 53 | 28 | 0 |

- **Ranking flip at 1×.** Fixed remote Mask R-CNN vs urgency-aware quality
  routing is 38 vs 41 (causal) but 48 vs 45 (no-skip). Under no-skip, remote Mask
  R-CNN also ties urgency-blind quality, urgency-blind risk and urgency-aware risk
  routing (48 each). Under causal replay they lead it by 4 to 5 conditions.
  Per recording, urgency-aware risk routing beats remote Mask R-CNN on 2 of 5
  feasible recordings and ties on 3 (causal). Under no-skip it ties on all 5. The
  adaptive advantage over the best fixed baseline is visible only when blocked
  observations are modelled.
- **Nonlinearity.** From 1× to 2×, 33 causal runs go from success to failure.
  - In every one, a person confirmed at 1× lost support because at least one of
    its confirming captures became unavailable.
  - The median number of processed observations falls from 9 to 5.
  - Under no-skip, not one run changes outcome between 1× and 2×.
- **0.5× to 1×.** 61 causal runs flip to failure, 51 of them with a confirming
  observation lost to blocking. Under no-skip, 38 flip, all under the urgent
  schedules:
  - 24 add request deadline misses (fixed local Mask R-CNN and the urgency-blind
    routers).
  - 14 are urgency-aware routers with zero misses. In the urgent window they run
    local Mask R-CNN at 0.5× (84 calls), but switch to CABiNet/YOLO11s at 1×
    (48/36 calls), once Mask R-CNN no longer fits the 2 s budget.
- **Figures.** `fig3_causal_timeline.png` shows two predeclared cases plus one
  case selected after evaluation by a stated rule, and is labelled so.
  `fig4_latency_sweep.png` plots processed observations, confirmed people, timely
  reports and successes together.
- **Spread.** The 37 runs that succeed only without blocking span 4 of 6
  recordings and six policies.

## Experiment 4: mission-failure attribution

From the 648 reference receipts (`exp4_*.csv`). Requirements map to:
- **confirmation**: repeated evidence
- **reporting**: report recall, plus false/duplicate report caps
- **resources**: battery, memory, communication
- **execution**

Two more diagnostics are reported separately: *coverage* (fewer than two people
ever matched) and *request timing* (deadline misses). *Freshness* does not apply
to the single-window reference contract; it is exercised by the `revisit`
contract in Experiment 1.

**Urgency-aware vs urgency-blind, transfer split (48 conditions):**

| Pair | Successes | Deadline misses | Mean task energy | Remaining unmet requirement patterns |
|---|---|---|---|---|
| Quality+risk: aware / blind | 25 / 24 | 0 / 90 | 168 / 198 J | aware: 18 confirmation+report recall, 5 report recall only; blind: 24 confirmation+report recall |
| Quality: aware / blind | 24 / 24 | 0 / 90 | 171 / 198 J | aware: 20 confirmation+report recall, 4 report recall only; blind: 24 confirmation+report recall |

No remaining failure of either pair involves battery, memory, communication,
execution or report caps. The aware risk router's single extra success is the
release's `sector_revisit/sector_7` urgency-recovery case.

**Per-person evidence funnel (216 person-runs per policy, all 72 conditions):**

| Policy | Visible on a processed capture | Matched ≥1 | Confirmed | Reported on time |
|---|---:|---:|---:|---:|
| Quality+risk, urgency-aware | 216 | 172 | 139 | 129 |
| Quality+risk, urgency-blind | 216 | 153 | 121 | 120 |
| Fixed remote Mask R-CNN | 216 | 146 | 114 | 112 |
| Fixed local YOLO11s | 216 | 120 | 24 | 24 |

Every person is visible on some processed capture. The largest loss for every
policy is visible-but-never-matched: model recall, not timing.

**Counterfactual rescoring.** These are evaluation-only relaxations of fixed
histories, not new executions (`exp4_counterfactual_relaxations.csv`). Relaxing
one requirement at a time rescues few runs:
- Lowering report recall to 1/3 adds 4 successes for each urgency-aware router on
  the transfer split.
- Removing the battery reserve adds 6 to 12 successes, only for fixed local Mask
  R-CNN.
- Removing the raw false-component cap adds 1 (urgency-aware quality, development
  split).
- Relaxing repeated evidence, evidence recall, the report deadline or report caps
  adds none.

Failures therefore aren't near-misses on a single threshold. They are evidence
shortfalls on several requirements at once.

## Deviations and corrections

1. **`survey` is rescoring-only.** See the Experiment 1 amendment.
2. **No-skip tracker bug, found and fixed before analysis.** The first no-skip
   run left the tracker's own serial-order check
   (`experiments/lake_tracking.py:190-196`) active. The pipeline caught that error
   and silently discarded the evidence of overlapping captures, which biased the
   ablation *against* no-skip. A per-event trace exposed it. The fix:
   - The check is now relaxed in the same FIFO sense, scoped to no-skip runs only.
   - Any downstream execution failure now aborts a no-skip run.
   - Every stage was regenerated.

   The superseded output was deleted and never used.
3. **Leakage observer gap, fixed before the committed audit.** Its first version
   did not scan list elements for target identities; a unit test caught this. The
   committed audit uses the corrected observer.
4. **Sensitivity grid declared after Experiment 1.** The 144-setting grid was
   declared after the five Experiment 1 contracts were scored and after
   per-condition diagnostics (people sighted, raw false components) had been
   inspected. It was committed before any grid setting was scored. That is what
   motivated including the sightings-only and zero-false-alarm values.
5. **Timeline selection.** `fig3_causal_timeline.png` adds one case chosen
   after evaluation as an illustration, labelled in the figure.

## Limitations

- Six recordings on one known map. Results describe this workload and model pool
  and do not generalise to other maps, missions or hardware.
- Network, battery and energy remain declared models, not measurements.
- The hindsight search is heuristic. "Unresolved" never means infeasible, and
  witnesses never mean optimal.
- No-skip is an availability ablation, not a parallel-hardware model; concurrent
  memory is not charged.
- No Depot recordings are in the release bundle, so the Depot track is not covered.
- The extensibility experiment (comparing association/verification
  implementations) was not run.

## Reproduce

Start with the [artifact guide](ARTIFACT.md). From an installation with the
`dev,replay` extras, use an unused output directory:

```bash
python tools/artifact.py verify
python tools/artifact.py full --workers 4 --output results/artifact-full
python -m pytest -q tests/test_validation_experiments.py
```

The wrapper recomputes the 648 reference missions, runs all validation stages
and compares generated CSV/JSON data with the committed references. It preserves
the validation configuration and algorithms while isolating outputs from the
reference files. Add `--figures` after installing `.[report]` to regenerate the
plots as well.

Individual stages remain available: `parity`, `bounds`, `reruns`, `causal`,
`hindsight`, `verify`, `sweep`, `analyze`. Their historical defaults write tables
to `examples/validation`; the artifact wrapper keeps all new outputs in the
selected run directory instead. See [reproduction details](REPRODUCING.md) and
[the reference artifact index](../examples/validation/README.md).
