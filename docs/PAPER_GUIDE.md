# Paper-to-artifact guide

This repository accompanies **EmbodiedIntentBench: Benchmarking Intent-Driven
Runtime Adaptation for Mobile Robotic Missions**. The contribution is an
evaluation framework connecting explicit mission requirements to physical
observation opportunities, causal execution, resource accounting, and mission
fulfillment. Perception configuration selection is the evaluated application;
the benchmark does not propose one routing algorithm as its central contribution.

## Design and implementation

| Paper component | Implementation or supplied input |
| --- | --- |
| Structured mission contracts | [Multi-phase contract definitions](../experiments/lake_intent_contract.py), [mission objectives](../experiments/lake_missions/objectives.py), [dynamic Lake contract](../experiments/intent_mission_v2/protocol.py), [validation variants](../experiments/validation/contracts.py) |
| Physical workload and candidate predictions | [Portable Lake inputs](../data/replay/), [input manifest](../data/replay/manifest.json), [optional fresh acquisition](SIMULATION.md) |
| Runtime decisions using public state | [Dynamic Lake policies](../experiments/intent_mission_v2/policy.py), [multi-phase policies](../experiments/mission_coverage_routing/policy.py) |
| Causal execution and resource accounting | [Execution loop](../experiments/dynamic_intent/runtime.py), [portable replay binding](../tools/replay.py), [validation execution](../experiments/validation/runtime.py) |
| Evidence and mission fulfillment | [Dynamic outcome binding](../experiments/intent_mission_v2/runtime.py), [deadline-aware evidence/report pipeline](../experiments/mission_task_deadlines/pipeline.py), [validation analyses](../experiments/validation/) |

The benchmark holds the physical workload fixed within a matched comparison.
Policies can process different subsets because computation blocks observations.
Only completed, selected predictions contribute to the evidence pipeline;
private evaluation labels are not policy inputs. Replay recomputes decisions and
mission outcomes, while charging declared execution costs for stored predictions.
It is not a fresh neural-inference, field-flight, or live-server experiment.

## Two case studies, three cohorts

| Setting | Multi-phase Lake | Multi-phase Depot | Dynamic Lake |
| --- | --- | --- | --- |
| Recordings | 6 × 120 s | 4 × 120 s | 6 × 42 s |
| Evaluated captures per recording | 60 | 60 | 18 during a 36 s sensing pass |
| Designated targets | 6 across 3 sectors | 6 across 3 sectors | 3 focal people per recording |
| Mission requirements | Rapid survey or verified inspection; 4 phases, including a fresh revisit | Same two multi-phase contracts | Confirm and report at least 2 people, with repeated evidence |
| Model pool | 9 configurations from 4 checkpoints | 9 configurations from 4 checkpoints | 3 checkpoints × 2 execution locations |
| Execution locations | Local | Local | Local and modeled remote |
| Varied conditions | 2 contracts × 2 initial battery levels | 2 contracts × 2 initial battery levels | 3 request-urgency schedules × 2 links × 2 battery levels |
| Conditions per strategy | 24 | 16 | 72 |
| Strategies / total runs | 21 / 504 | 22 / 352 | 9 / 648 |
| Battery capacity model | 30 Wh | 30 Wh | 120 Wh |

The multi-phase rapid-survey contract requires at least half of the designated
targets in each phase, supported by two observations spanning at least 2 s.
Verified inspection requires all designated targets, three observations spanning
at least 4 s, and a higher mission report-recall threshold. Both have four 30 s
phases and reports due by 120 s, distinct from the outer 130 s execution limit.

Dynamic Lake uses one cumulative evidence window. Each confirmed person needs
two observations spanning at least 2 s, and reports are due by 44 s. Urgency
changes the allowance of newly released perception requests between 8 s and 2 s;
it does not redefine the sensing objective or erase accumulated evidence.
[EXPERIMENT.md](EXPERIMENT.md) specifies the full contract and cost profiles.

The dynamic study's four-recording comparative subset has 48 conditions per
strategy, or 24 at each link/battery level. The other two recordings supply 24
development conditions. Output files call the comparison subset `transfer`, but
all six recordings were known before the study. Validation reuses all 72
conditions; it introduces no new physical flights.

## Paper figures and reproduction scope

| Figure | Question and denominator | Supported reproduction |
| --- | --- | --- |
| 4 | Does perception recall predict mission fulfillment? Nine Depot configurations; recall over 680 visible target observations from 240 captures; success over 16 conditions | Recompute from original numeric case/frame extracts. Full Depot recordings, prediction banks, and all 352 execution records are not bundled |
| 5 | How do changing urgency, connectivity, and battery affect execution? Dynamic Lake: 48 comparison conditions per strategy; 24 per factor level | Rerun the bundled dynamic study, then reduce its outputs |
| 6 | Does blocking subsequent observations change outcomes and rankings? All 72 dynamic Lake conditions × 9 strategies per latency/protocol setting | Rerun causal/no-skip analyses; no-skip is an analytical ablation, not a parallel executor |
| §5.4 | Which conditions are infeasible or lose evidence? 72 dynamic Lake conditions; 216 person-condition instances for one strategy | Rerun feasibility, hindsight, and evidence diagnostics; hindsight is evaluator-only |

The [figure guide](../examples/paper_figures/README.md) supplies plotting commands,
numeric sources, and value verification for alternative plots. The plots in this
artifact are not identical to the final manuscript artwork. The optional
`fig7` plot visualizes the diagnostics discussed in §5.4; it is not a
manuscript figure. `tools/artifact.py full` reproduces the
**bundled dynamic Lake study and its validation suite**, not the complete
multi-phase case study. Primary multi-phase implementation remains available,
but the original raw input bundles are required to rerun those missions.

The Fig. 6 latency sweep scales configured inference/service latency and the
associated local compute energy under the declared power model; predictions,
network transfers, and downstream costs remain fixed. It is not a latency-only
perturbation with energy held constant.

Validation includes five contract variants. Its single-observation `survey`
relaxation and 144-setting threshold map rescore existing histories; they do not
show policy adaptation to those settings. Other supported contracts have
separate reruns. See [VALIDATION.md](VALIDATION.md) for the distinction between
certified infeasibility, observed successes, hindsight witnesses, and unresolved
conditions.

The [Jetson sensitivity study](../experiments/jetson/) substitutes measured
onboard costs in the dynamic-study replay; network and remote execution remain
modeled. The [intent-driven execution study](../experiments/intent_adaptation/)
compares a simple report-based stopping rule with the same base policies across
four separate mission contracts. Its device measurements execute precomputed
schedules, with mission outcomes assessed using stored predictions. Each study's
guide describes the included results, reproduction commands, and scope.

## Scope of the claims

The implemented extension interfaces support new contracts, recordings, models,
policies, execution profiles, and evidence/evaluation components. This does not
establish performance across arbitrary robotic tasks. The reported comparisons
focus on perception, fixed trajectories, static people, and authored simulator
scenes. Replay uses modeled network and energy costs, including a sensitivity
analysis grounded in measured onboard profiles; split inference, deployed remote-server
execution, open-ended intent compilation, and independent field generalization
are not evaluated by these experiments.
