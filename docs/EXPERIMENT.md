# Dynamic Lake experiment definition

This guide defines the bundled dynamic-urgency case study and its validation
analyses. It does not describe the separate 120-second multi-phase Lake and
Depot studies. See [the paper-to-artifact guide](PAPER_GUIDE.md) for those cohorts
and the scope of the supplied reproduction workflows.

## Fixed inputs and routed work

This case study evaluates perception-model routing on six Gazebo recordings from
one Lake terrain. Three route families each contribute two focal-sector
recordings. Each recording lasts 42 seconds, including a 36-second motion pass;
18 predeclared captures from that pass are evaluated.

The flight path, camera frames, target placement, model predictions, tracking
and report verification are shared across policies. A policy selects a
perception model and local or remote execution. It cannot change the path or
inspect evaluator identities, labels, future intent updates or future
connectivity. Downstream tracking and reporting are controlled components,
not additional routed model families.

Replay uses stored neural predictions and modeled timing, transmission and
energy. Remote execution is a replay action with modeled network costs, not an
RPC to a live server. Optional fresh inference and simulation are documented
separately in [MODELS.md](MODELS.md) and [SIMULATION.md](SIMULATION.md).

## Reference mission and deadlines

Confirm and report **at least two of three focal people**. Each confirmed person
requires at least two observations separated by at least two seconds. Reports
must arrive by 44 seconds. At most two raw false components, two false reports
and two duplicate reports are allowed; battery reserve and other execution
constraints must also hold. Correct nonfocal detections neither satisfy focal
recall nor count as false reports.

These are two different timing constraints:

- **Per-request allowance:** 8 or 2 seconds for each newly admitted perception
  request, including its assigned overheads. Work already running retains the
  deadline assigned when it started. Late request outputs are discarded while
  incurred costs remain charged.
- **Mission evidence/report deadline:** the reference contract requires the
  necessary evidence and reports by 44 seconds. Changing request urgency does
  not automatically change this deadline or discard accumulated evidence.

A mission succeeds only if all required checks pass. Report success counts
alongside deadline misses and modeled task energy; model recall alone is
insufficient.

## Models and policies

| Model family | Role |
| --- | --- |
| Mask R-CNN | General-purpose instance segmentation |
| YOLO11s | General-purpose instance segmentation |
| CABiNet | Aerial semantic segmentation trained on UAVid |

Each model can execute locally or remotely, giving six actions. Their person
masks enter the same downstream mission logic. Checkpoint identities, licenses
and inference settings are in [the model guide](MODELS.md).

| Policy IDs in results | Selection rule |
| --- | --- |
| `quality_current`, `quality_initial` | Highest estimated recall among actions passing time, energy, memory and communication guards |
| `risk_current`, `risk_initial` | Quality-first selection with an additional predicted false-component allowance check |
| `fastest_current` | Lowest expected completion time among feasible actions |
| `fixed_local_mask`, `fixed_local_yolo`, `fixed_local_cabinet`, `fixed_remote_mask` | Always use the named model and execution location |

`current` policies use the currently announced per-request allowance. `initial`
policies continue selecting with the initial allowance; evaluation still enforces
the actual deadline. Fixed policies retain their action even when resource
forecasts predict failure. The false-component guard is a heuristic, not a
guarantee. Quality priors come from an earlier pooled 162-frame calibration,
not current-request evaluator feedback.

## Reference experiment grid

| Factor | Values |
| --- | --- |
| Request allowance | Constant 8 s; 8 to 2 s at 12 s; 8 to 2 to 8 s at 12/24 s |
| Network | Good: 50 Mbps upload, 100 Mbps download, 30 ms RTT; degraded: 2/10 Mbps, 150 ms RTT |
| Battery | Ample: 95% initial charge; constrained: reserve plus complete-flight cost plus 250 J task energy |
| Energy model | 120 Wh capacity, 120 W flight power, 20% reserve |
| Recordings | Six fixed inspection passes across three route families on one map |

This gives **72 conditions per policy and 648 runs across nine policies**.
Two shoreline recordings form the development subset; four other recordings
form the comparative evaluation subset (named `transfer` in output files),
giving 48 conditions per policy. All were already
known. The transfer subset is not an unseen-map or prospectively acquired test
set. Conditions on the same recording are correlated observations.

## Additional mission contracts

The validation suite distinguishes the three urgency schedules above from
**five mission contracts**:

These validation variants are separate from the primary multi-phase study's
rapid-survey and verified-inspection contracts. In particular, the `survey`
relaxation below requires only one observation; the multi-phase rapid-survey
contract requires two observations separated by at least two seconds.

| Contract | Difference from reference | Evaluation mode |
| --- | --- | --- |
| `confirm` | Two people, two observations each, at least 2 s apart; reports by 44 s | Reference execution and rescoring |
| `survey` | One observation per person suffices | Rescoring only |
| `confirm_strict` | Three observations spanning at least 4 s per person | Rescoring and contract-conditioned rerun |
| `revisit` | Fresh confirmation in both halves of the 36 s pass | Rescoring and contract-conditioned rerun |
| `time_critical` | Required evidence and reports by 36 s | Rescoring and contract-conditioned rerun |

The reporting pipeline enforces a two-observation floor, so `survey` cannot be
supplied as a runtime contract. The 144-setting sensitivity analysis also
rescores fixed execution histories; policies do not adapt to those settings.
Keep these analyses separate from new policy executions.

The remaining validation experiments compute feasibility bounds, search for
successful hindsight schedules, compare causal and no-skip execution, and
trace evidence/report failures. Exact definitions, denominators, amendments
and numerical results are in [VALIDATION.md](VALIDATION.md).
