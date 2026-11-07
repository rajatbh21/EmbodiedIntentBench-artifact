# Intent-driven execution across mission requirements

This experiment uses the dynamic Lake workloads and compares a base policy with
the same policy augmented by a simple stopping rule. At each observation, the
rule reads the current public mission phase and its required coverage. If the
runtime has already issued enough reports with distinct runtime component
labels, based entirely on evidence from that phase, it skips further perception
in that phase. Otherwise the base policy selects an action. The rule reads only
completed reports and the public contract; it does not use evaluator labels,
future observations, or unselected predictions. Runtime component labels are
proxies and do not guarantee a count of distinct ground-truth people.

The study compares four separate contracts (confirmation, time-critical,
stricter confirmation, and revisit), six policies, 72 conditions, and three
cost profiles, with and without the stopping rule: 5,184 matched pairs. The
profiles are the original declared costs and corrected 15 W Jetson AGX Orin
CPU/GPU profiles. These are stored-prediction replays, not device executions
of all 10,368 runs. This comparison varies requirements across runs; it does
not evaluate changing the mission contract during a run.

## Inspect results

```bash
python experiments/intent_adaptation/check_results.py
```

The script checks pair completeness and summarizes outcomes and modeled task
energy from `reference_rows.jsonl`. It also summarizes integrated device
measurements from `device_measurements.json`. Under the Orin GPU profile,
fixed-local Mask R-CNN averages 197.81 J without the rule and 142.38 J with
it for the confirmation contract; both variants succeed in 60/72 conditions.
The stricter confirmation contract averages 158.68 J with the rule and also
has 60/72 successes. Across all pairs, no previously successful replay becomes
a failure and three failures become successes. These results are specific to
the supplied workloads, policies, requirements, and cost profiles.

## Replay

Install the CPU replay dependencies described in the main artifact guide. A
prepared replay directory with `plan.json` is needed; for example, create one
using the bundled artifact workflow:

```bash
python tools/replay.py --all --workers 4 --output results/replay-full
python experiments/intent_adaptation/gated_replay.py \
  --replay-root results/replay-full \
  --orin-dir experiments/jetson/profiles \
  --workers 4 --out results/intent-adaptation
python experiments/intent_adaptation/check_results.py \
  --compare results/intent-adaptation/rows.jsonl
```

The output directory must be new. The full replay runs 10,368 executions; for
a shorter test add `--limit 1` to select the first eligible condition for each
combination. The comparison checks the generated rows against the matching
reference rows, including energy with numerical tolerance. Passing a subset
does not validate the omitted runs. See the Jetson documentation for measured
profile provenance and dependency details.

## Device scheduling check

`device_schedules.json` records precomputed gate-on and gate-off schedules for
the two confirmation contracts and six recordings. `device_replay.py` runs real
frame-paced local Mask R-CNN inference for the supplied schedules on an Orin;
the other processing stages are emulated. It does not make the stopping
decision online. `device_measurements.json` contains the original 24 integrated
per-run energy measurements and counts; it does not contain raw power traces.
The device results measure inference and whole-pass module energy separately.
Mission outcomes were scored offline using stored predictions, rather than
newly retained device masks. Device power and timing vary across runs.

With the hardware environment and model assets installed as described in the
Jetson documentation, a fresh output file can be obtained with:

```bash
python experiments/intent_adaptation/device_replay.py \
  --release . --schedules experiments/intent_adaptation/device_schedules.json \
  --out results/device-measurements.json
```

This hardware command requires an appropriate Jetson and is separate from the
CPU-only replay and its reference comparison.
