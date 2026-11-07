"""Portable, verified replay of the 648-cell perception-routing experiment.

The data bundle contains recorded RGB/ground truth and native prediction masks.
Every routing decision, request cost, report and mission outcome is recomputed.
This entry point does not perform fresh neural inference or a new Gazebo flight.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from types import FunctionType

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
DEFAULT_DATA = REPO / "data/replay"


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_path(root, relative):
    path = (Path(root) / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Bundle path must stay inside its root")
    return path


def source_hashes():
    paths = [p for folder in ("missionbench", "experiments") for p in (REPO / folder).rglob("*.py")]
    paths.append(Path(__file__).resolve())
    return {str(p.relative_to(REPO)): sha(p) for p in sorted(paths)}


def verify_data(data):
    data = Path(data)
    manifest = read(data / "manifest.json")
    if manifest["schema_version"] != "portable_perception_replay_v1":
        raise ValueError("Unknown replay bundle schema")
    for relative, expected in manifest["files"].items():
        path = checked_path(data, relative)
        if path.stat().st_size != expected["bytes"] or sha(path) != expected["sha256"]:
            raise ValueError("Replay input changed: " + relative)
    plan = read(data / "study.json")
    from experiments.intent_mission_v2.study import grid
    from experiments.intent_mission_v2.protocol import SCHEDULES, WINDOWS
    if plan["grid"] != grid(plan["cases"]) or len(plan["grid"]) != 648:
        raise ValueError("Incomplete or changed factorial experiment")
    if plan["intent_schedules"] != {k: [list(v) for v in values] for k, values in SCHEDULES.items()}:
        raise ValueError("Intent schedules changed")
    if plan["report_windows"] != [list(v) for v in WINDOWS]:
        raise ValueError("Report windows changed")
    references = read(data / "expected_rows.json")
    if {r["run_id"] for r in references} != {r["run_id"] for r in plan["grid"]} or len(references) != 648:
        raise ValueError("Reference comparison is incomplete")
    return plan


def prepare(data, output):
    data, output = Path(data).resolve(), Path(output).resolve()
    plan = verify_data(data)
    if output.exists():
        raise ValueError("Use a new output directory; prior executions are preserved")
    shutil.copytree(data, output)
    plan.update(source_hashes=source_hashes(), release_bundle_sha256=sha(data / "manifest.json"),
                release_provenance="Fresh release source hashes and portable input hashes; original historical freeze hashes are not asserted to match.")
    save(output / "plan.json", plan)
    return plan


def load_case(root, plan, item):
    """Same scenario values as the study, with recordings resolved inside the bundle."""
    from missionbench.schemas.common import ComparisonOperator
    from missionbench.schemas.contract import Contract, PrivacyLevel
    from experiments.lake_causal_router import MissionCosts
    from experiments.mission_network_policies import ActionDescriptor, PolicyCosts
    from experiments.v4_benchmark.resources import ResourceConfig
    from experiments.v4_gazebo_pilot.timing import CaptureSchedule
    from experiments.wide_area_routing.adapter import WideReplayScenario
    from experiments.strong_routing_20261030.study import initial_battery, link_trace
    from experiments.intent_mission_v2.protocol import MissionIntent
    case = plan["cases"][item["flight"]]
    recording = read(checked_path(root, plan["recordings"][item["flight"]]))
    expected = initial_battery(case, item["energy_condition"])
    if item["battery"] != expected:
        raise ValueError("Initial charge differs from the declared energy condition")
    actions = tuple(ActionDescriptor(**a) for a in plan["actions"])
    outer = Contract("LOCAL_INSPECTION", "HUMAN_SEARCH_SEGMENTATION", "instance_mask_set", "target_recall",
                     ComparisonOperator(">="), 0., case["duration_s"] + 10, 10000., .2, PrivacyLevel.REMOTE_ALLOWED)
    resources = ResourceConfig.parse(dict(memory_capacity_mb=4096, models={a.action_id: dict(
        memory_mb=a.memory_mb, load_latency_s=a.load_latency_s, load_energy_j=a.load_energy_j) for a in actions}),
        [a.action_id for a in actions])
    scenario = WideReplayScenario(case["duration_s"], 2., 120., 120., expected, .05, .75, .3,
        recording["world"], {o["label"]: o["object_id"] for o in recording["world"]["objects"]}, outer,
        recording["recording_identity"], CaptureSchedule(**recording["capture_schedule"], step_ns=10_000_000), resources)
    return dict(scenario=scenario, contract=MissionIntent(case, expected), actions=actions,
        costs=PolicyCosts(MissionCosts.from_scenario(scenario), **plan["policy_costs"]),
        trace=link_trace(item["link"], scenario.duration_s),
        profiles={name: read(checked_path(root, path)) for name, path in plan["profiles"].items()},
        learned_profile=read(checked_path(root, plan["calibration_path"])))


def bound_run_case():
    # Preserve the original execution loop; replace only portable loading/provenance.
    from experiments.dynamic_intent.runtime import run_case
    from experiments.intent_mission_v2.protocol import WINDOWS
    from experiments.intent_mission_v2.policy import make_policy
    from experiments.intent_mission_v2.runtime import outcome
    namespace = dict(run_case.__globals__)
    namespace.update(WINDOWS=WINDOWS, load_case=load_case, make_policy=make_policy,
                     dynamic_outcome=outcome, source_hashes=source_hashes)
    return FunctionType(run_case.__code__, namespace, "run_case")


def initialize(output, plan, binding):
    global _STATE
    from experiments.wide_area_diagnosis.paid_clip import load_bank
    from experiments.lake_tracking_runtime import load_public_terrain
    binding = dict(binding, recording_path=str(checked_path(output, binding["recording_path"])))
    terrain, _ = load_public_terrain(checked_path(output, plan["assets"]))
    bank = load_bank(binding)
    manifest = read(Path(output) / "predictions" / binding["case_id"] / "manifest.json")
    _STATE = (output, plan, terrain, bank, manifest, bound_run_case())


def execute_one(item):
    output, plan, terrain, bank, manifest, loop = _STATE
    row = loop(output, plan, item, terrain, bank, manifest)
    save(Path(output) / row["path"] / "row.json", row)
    return row


def scientific_row(row):
    """Only provenance hashes and artifact location are excluded, never metrics."""
    return {key: value for key, value in row.items() if key not in ("artifacts_sha256", "path")}


def compare_rows(rows, references):
    expected = {r["run_id"]: scientific_row(r) for r in references}
    differences = []
    for row in rows:
        actual = scientific_row(row)
        reference = expected.get(row["run_id"], {})
        keys = sorted(k for k in actual.keys() | reference.keys() if actual.get(k) != reference.get(k))
        if keys:
            differences.append(dict(run_id=row["run_id"], fields=keys))
    return differences


def run(data, output, *, all_runs=False, workers=2):
    started = time.perf_counter()
    plan = prepare(data, output)
    output = Path(output).resolve()
    selected = plan["grid"]
    if not all_runs:
        # All nine policies and three intent schedules, one transfer clip/link/battery.
        first = sorted(k for k in plan["cases"] if not k.startswith("shoreline_loop/"))[0]
        selected = [i for i in selected if i["flight"] == first and i["link"] == "good" and i["energy_condition"] == "ample"]
    rows = []
    for binding in read(output / "native_bindings.json")["bindings"]:
        items = [i for i in selected if i["flight"] == binding["case_id"]]
        if not items:
            continue
        with ProcessPoolExecutor(max_workers=workers, initializer=initialize, initargs=(output, plan, binding)) as pool:
            rows.extend(pool.map(execute_one, items))
        print(f"Recomputed {len(rows)}/{len(selected)} missions", flush=True)
    rows.sort(key=lambda r: r["run_id"])
    differences = compare_rows(rows, read(output / "expected_rows.json"))
    if plan["source_hashes"] != source_hashes():
        raise ValueError("Release implementation changed during replay")
    verify_data(output)
    save(output / "rows.json", rows)
    receipt = dict(complete=len(rows) == len(selected), runs=len(rows), exact_scientific_parity=not differences,
        differing_rows=differences, comparison_exclusions=["artifacts_sha256", "path"],
        elapsed_host_s=time.perf_counter() - started, fresh_neural_inference=False, new_simulator_flight=False,
        plan_sha256=sha(output / "plan.json"), rows_sha256=sha(output / "rows.json"),
        bundle_sha256=sha(output / "manifest.json"), source_hashes=plan["source_hashes"])
    save(output / "validation.json", receipt)
    print(json.dumps({k: receipt[k] for k in ("runs", "exact_scientific_parity", "elapsed_host_s")}, indent=2))
    if differences:
        raise ValueError("Scientific parity failed; inspect validation.json")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=REPO / "results/replay")
    parser.add_argument("--all", action="store_true", help="Run all 648 cells; default is a 27-cell smoke replay")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.verify_only:
        verify_data(args.data)
        print("Replay bundle verified: 648 declared cells")
    else:
        run(args.data, args.output, all_runs=args.all, workers=args.workers)


if __name__ == "__main__":
    main()
