# EmbodiedIntentBench

EmbodiedIntentBench evaluates runtime perception decisions against structured
mission requirements on recorded UAV observation streams. It connects mission
contracts, simulated physical workloads, causal execution, and mission-level
scoring: computation advances mission time, can block later captures, and
consumes modeled resources.

The evaluated trajectories are fixed Gazebo recordings. Replay uses precomputed
neural predictions and modeled communication and execution costs, including a
sensitivity analysis with costs measured on a Jetson AGX Orin. Selected
precomputed schedules were also executed on the device for timing and energy
measurements. These experiments do not establish performance of a deployed drone
or autonomous navigation.

## Getting started

From the repository root on Linux with Python 3.11 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,replay]'

# Check the included inputs and experiment definition.
python tools/artifact.py verify

# Replay 27 missions and compare scientific results to the references.
python tools/artifact.py smoke --workers 2 --output results/artifact-smoke

# Replay the dynamic Lake study and its validation suite.
python tools/artifact.py full --workers 4 --output results/artifact-full
```

Choose a new output directory for each run. Verification and comparison failures
return a nonzero exit code. After installation, these commands do not require a
GPU, model download, or simulator. The full run is substantially longer; a prior
Linux execution took about 27 minutes with four workers. See the
[artifact guide](docs/ARTIFACT.md), [reproduction guide](docs/REPRODUCING.md),
and [verification receipts](docs/VERIFICATION.md) for details and expected
outputs. An independent check of this snapshot on a Linux server passed input
verification, the 27-mission smoke comparison, and the full suite: 648 baseline
rows, 1,944 contract reruns, 5,184 causal/no-skip runs, 25 successful hindsight
witnesses, and all 23 CSV and two JSON reference tables matched. This check used
an existing Python environment; it did not verify a fresh installation or the
later public upload. The bundled receipts describe earlier executions.

**Portability:** Linux is the documented environment. A subsequent 27-mission
run on Windows reproduced mission-success fields but failed exact parity in
communication and energy fields for 18 remotely routed missions. The cause has
not been established. Use the recorded Linux environment for exact comparisons
and consult the verification receipts; do not interpret this Windows check as a
successful smoke test.

## What is included

| Paper study | Included evidence and reproduction scope |
| --- | --- |
| Multi-phase Lake | Implementation retained; complete original recordings, prediction bank, and run outputs are not included. |
| Multi-phase Depot | Numeric case/frame results for Figure 4 are included; complete original replay inputs are not. |
| Dynamic Lake | Six recorded observation streams, prediction banks, profiles, reference outcomes, and replay code for 648 policy–condition executions. |
| Validation | Contract sensitivity, feasibility, causal/no-skip comparisons, and failure attribution code with reference CSV/JSON results. |
| Jetson sensitivity | Measured CPU/GPU profiles, replay code and reference outcomes, and fixed-local device scheduling measurements. |
| Intent-driven execution | Matched replays with and without a mission-requirement stopping rule, plus device measurements of selected precomputed schedules. |

The dynamic study's four-recording comparison subset has 48 conditions per
strategy; its validation uses all 72 conditions per strategy. These conditions
reuse known recordings and are not independent field flights. See the
[paper-to-artifact guide](docs/PAPER_GUIDE.md) for figure mappings, denominators,
and implementation references.

Replay recomputes runtime choices and mission outcomes from stored predictions;
it does not rerun neural inference or acquire new images. The 144-setting
contract sensitivity map is fixed-history rescoring, while selected contracts
also have separate policy reruns. The no-skip protocol is an analytical
counterfactual, not a deployed parallel executor. Hindsight success witnesses
show feasibility under the modeled assumptions, not global optimality.

## Navigate the code

- `tools/artifact.py`: input checks, smoke replay, and full replay entry points.
- `data/replay/`: recordings, prediction banks, profiles, and reference outcomes.
- `missionbench/` and `experiments/`: contracts, policies, execution, and scoring.
- `experiments/validation/`: validation analyses and result-generation code.
- `experiments/jetson/`: measured profiles, hardware sensitivity replay, and device checks.
- `experiments/intent_adaptation/`: mission-requirement experiment, reference results, and checks.
- `examples/validation/`: compact reference tables and diagnostic outputs.
- `examples/paper_figures/`: figure data, plotting scripts, and source checks.
- `tests/` and `docs/`: tests, experiment definitions, and recorded receipts.

Optional workflows for neural inference and Gazebo require additional
dependencies and assets; see [models](docs/MODELS.md) and
[simulation](docs/SIMULATION.md).

The remaining multi-phase workloads and prediction banks will be released upon acceptance.

## Attribution and release

First-party code is [MIT licensed](LICENSE). Third-party models and scenery
retain their own licenses and attribution; see [data attribution](data/replay/ATTRIBUTION.md)
and the model documentation.
