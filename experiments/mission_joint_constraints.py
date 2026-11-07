"""Frozen factorial development replay: intent, battery, link and onboard compute.

All runtime, transport, neural predictions and existing policy implementations
remain unchanged. A new scenario adapter assigns each explicit factorial cell.
"""

from __future__ import annotations

import argparse
import itertools
import math
import shutil
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from experiments.mission_joint_constraints_scenario import (
    actions_for,
    derivative_documents,
    policy_costs,
    policy_for,
    scenario_for,
    trace_for,
    tracking_cost,
)
from experiments.mission_measured_cost_study_v2 import calibrated_arm, require
from experiments.mission_network_policies import FIXED_LOCAL_POLICIES
from experiments.mission_network_prepare_v2 import runtime_inventory
from experiments.mission_network_study_v1.study import authored_traces
from experiments.mission_network_study_v1.study import specification as network_specification
from experiments.mission_runtime_profile import DEVELOPMENT, REPO, imports, read, save, sha

OUTPUT = REPO / "results/mission_joint_constraints_v1_20261027"
PARENT = REPO / "results/mission_network_study_v2_20261027"
POLICIES = (
    "fixed_local_CFG_MASKRCNN_1280",
    "fixed_local_CFG_MASKRCNN_1280_ROT0_90",
    "fixed_local_CFG_YOLO11S_960_ROT0",
    "fixed_remote_union",
    "snapshot_quality_2s",
    "mission_evidence_network",
    "mission_evidence_network_blind",
    "mission_evidence_local_only",
)
FACTOR_KEYS = ("flight", "mission", "initial_battery_frac", "link", "onboard_latency_scale")


@dataclass(frozen=True)
class FactorialSpec:
    """Explicit Cartesian design; subsets are valid, duplicate/unknown levels are not."""

    flights: tuple = ("east_shore_day", "west_rocks_day")
    missions: tuple = ("rapid_survey", "verified_inspection")
    batteries: tuple = (0.34, 0.65)
    links: tuple = ("good", "constrained", "intermittent")
    onboard_latency_scales: tuple = (1.0, 2.0)
    policies: tuple = POLICIES

    def __post_init__(self):
        choices = {
            "flights": {"east_shore_day", "west_rocks_day"},
            "missions": {"rapid_survey", "verified_inspection"},
            "links": set(authored_traces()),
            "policies": set(POLICIES),
        }
        for name, values in vars(self).items():
            require(isinstance(values, tuple) and values, "Nonempty tuple levels required")
            require(len(values) == len(set(values)), "Duplicate factor level: " + name)
            if name in choices:
                require(set(values) <= choices[name], "Unknown factor level: " + name)
            else:
                require(
                    all(type(v) in (int, float) and math.isfinite(v) for v in values),
                    "Finite numeric factor levels required",
                )
        require(all(0 < x <= 1 for x in self.batteries), "Battery outside (0,1]")
        require(all(x > 0 for x in self.onboard_latency_scales), "Positive compute scales required")
        require(
            all(isinstance(v, float) for v in self.onboard_latency_scales),
            "Use explicit float compute scales for stable action-table keys",
        )

    @classmethod
    def from_dict(cls, data):
        require(set(data) == set(cls.__dataclass_fields__), "Missing or unknown design fields")
        return cls(**{key: tuple(value) for key, value in data.items()})


def cell_name(flight, mission, battery, link, scale):
    return f"{flight}__{mission}__b{battery:.8g}__{link}__c{scale:.8g}"


def shard_name(flight, mission):
    return flight + "__" + mission


def specification(config=None):
    c = config or FactorialSpec()
    rows = []
    for flight, mission, battery, link, scale, policy in itertools.product(
        c.flights, c.missions, c.batteries, c.links, c.onboard_latency_scales, c.policies
    ):
        rows.append(
            dict(
                cell=cell_name(flight, mission, battery, link, scale),
                flight=flight,
                mission=mission,
                initial_battery_frac=battery,
                link=link,
                onboard_latency_scale=scale,
                policy=policy,
                shard=shard_name(flight, mission),
            )
        )
    require(len({(r["cell"], r["policy"]) for r in rows}) == len(rows), "Ambiguous cell labels")
    old = network_specification()
    return dict(
        schema_version="joint_constraints_development_spec_v1",
        factors={k: list(v) for k, v in asdict(c).items()},
        flights=list(c.flights),
        missions=list(c.missions),
        policies=list(c.policies),
        links=list(c.links),
        grid=rows,
        expected_replays=len(rows),
        shards=[shard_name(f, m) for f in c.flights for m in c.missions],
        traces={link: authored_traces()[link] for link in c.links},
        remote_service_factor=0.25,
        local_power_w=15.0,
        communication_budget_mb=100.0,
        privacy_level="remote_allowed",
        policy_costs=old["policy_costs"],
        endpoint=old["endpoint"],
        future_trace_visible_to_policy=False,
        new_neural_inference=False,
        fresh_evaluation=False,
        known_sensor_infeasible_intents=["verified_inspection"],
        feasibility_scope="The source development protocol certifies strict inspection "
        "sensor-infeasible; strict cells are retained stress/integration controls, not "
        "feasible routing discrimination. "
        "Rapid-survey paid feasibility is not assumed for any resource combination.",
        intervention="Only local inference latency is scaled from measured host warm p95; "
        "onboard inference energy is15W times scaled latency. Remote service always uses "
        "0.25 times unscaled baseline p95. Loading, codec, routing and tracking remain fixed.",
        selection_rationale="The known network study's strongest local Mask and union, "
        "fast local full YOLO11s, remote union and all4 authored adaptive policies; "
        "selected before any joint-grid outcomes. All18 actions remain available.",
        comparisons="Within-recording matched one-factor pairs and two-factor differences "
        "in differences over the complete grid; action/schedule sensitivity is separate "
        "from full-success, phase, cancellation and reserve changes. Failures retained.",
        matched_policy_controls=[
            "mission_evidence_network_blind",
            "mission_evidence_local_only",
            "snapshot_quality_2s",
        ],
        scope="Development factorial sensitivity on2 reused day recordings and fixed routes; "
        "authored virtual links and hypothetical onboard compute scaling/energy. No "
        "empirical wireless/device slowdown, held-out generalization or population inference.",
    )


def freeze_specification(root, config=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    save(root / "specification.json", specification(config))
    save(
        root / "specification_chronology.json",
        dict(
            written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            specification_sha256=sha(root / "specification.json"),
            joint_outcomes_exist=False,
            prior_network_and_cost_results_known=True,
        ),
    )


def scaled_actions(parent_actions, scale):
    """Remote action descriptors are byte-for-byte unchanged by local slowdown."""
    result = []
    for item in parent_actions:
        value = dict(item)
        if item["placement"] == "local":
            value["latency_s"] = item["latency_s"] * scale
            value["energy_j"] = value["latency_s"] * 15.0
        result.append(value)
    return result


def expected_cases(spec, source_protocol):
    cases = {}
    for row in spec["grid"]:
        source_case = source_protocol["cases"][row["flight"]]
        require(source_case["role"] == "development", "Evaluation recording forbidden")
        cases[row["cell"]] = dict(
            source_flight=row["flight"],
            mission=row["mission"],
            link=row["link"],
            initial_battery_frac=row["initial_battery_frac"],
            onboard_latency_scale=row["onboard_latency_scale"],
            role="development",
            known_sensor_infeasible=row["mission"] == "verified_inspection",
            environment=source_case["environment"],
            lineage=source_case["lineage"],
            layout_group=source_case["layout_group"],
        )
    return cases


def prepare(root):
    from experiments.mission_network_study_v1.study import verify as verify_parent

    root = Path(root)
    spec = read(root / "specification.json")
    require(spec == specification(FactorialSpec.from_dict(spec["factors"])), "Changed design")
    require(not (root / "plan.json").exists() and not (root / "runs").exists(), "Preserve attempt")
    parent = verify_parent(PARENT)
    require(read(PARENT / "completion.json")["complete"] is True, "Incomplete parent evidence")
    protocol = imports()["development_verify"](DEVELOPMENT)
    frozen, copied = dict(parent["input_sha256"]), {}

    def copy_file(path, relative):
        path = Path(path)
        frozen[str(path.resolve())] = sha(path)
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        require(sha(path) == sha(destination), "Changed copy")
        copied[str(relative)] = sha(destination)

    for name in ("plan.json", "completion.json", "summary.json", "validation.json"):
        frozen[str(PARENT / name)] = sha(PARENT / name)
    for name in ("specification.json", "specification_chronology.json"):
        frozen[str((root / name).resolve())] = sha(root / name)
    for key, relative in parent["profile_paths"].items():
        copy_file(PARENT / relative, Path("profiles") / (key + ".json"))
    cases = expected_cases(spec, protocol)
    for flight in spec["flights"]:
        for mission in spec["missions"]:
            for name in ("scenario", "intent", "contract"):
                copy_file(
                    DEVELOPMENT / flight / mission / (name + ".json"),
                    Path("originals") / flight / mission / (name + ".json"),
                )
    for cell, case in cases.items():
        original = root / "originals" / case["source_flight"] / case["mission"]
        values = derivative_documents(
            *(read(original / (n + ".json")) for n in ("scenario", "intent", "contract")),
            parent["local_latency_s"],
            case["initial_battery_frac"],
            case["onboard_latency_scale"],
        )
        for name, value in zip(("scenario", "intent", "contract"), values, strict=True):
            dest = root / "missions" / cell / case["mission"] / (name + ".json")
            save(dest, value)
            copied[str(dest.relative_to(root))] = sha(dest)
    implementations = dict(parent["implementation_sha256"])
    for path in [
        *REPO.glob("experiments/mission_joint_constraints*.py"),
        *REPO.glob("tests/test_mission_joint_constraints*.py"),
    ]:
        implementations[str(path.resolve())] = sha(path)
    snapshots = {}
    for path, expected in implementations.items():
        require(sha(path) == expected, "Changed inherited implementation")
        relative = Path("source_snapshot") / Path(path).relative_to(REPO)
        copy_file(path, relative)
        snapshots[path] = str(relative)
    plan = dict(spec)
    plan.update(
        schema_version="mission_joint_constraints_v1",
        cohort=parent["cohort"],
        assets=parent["assets"],
        cases=cases,
        tracking_cost=parent["tracking_cost"],
        local_latency_s=parent["local_latency_s"],
        model_signatures=parent["model_signatures"],
        actions_by_scale={
            str(s): scaled_actions(parent["actions"], s)
            for s in spec["factors"]["onboard_latency_scales"]
        },
        calibration=parent["calibration"],
        calibration_sha256=parent["calibration_sha256"],
        parent_plan=str(PARENT / "plan.json"),
        parent_plan_sha256=sha(PARENT / "plan.json"),
        profile_paths={k: "profiles/" + k + ".json" for k in parent["profile_paths"]},
        input_sha256=frozen,
        copied_input_sha256=copied,
        implementation_sha256=implementations,
        source_snapshot_paths=snapshots,
        runtime_versions={
            d: runtime_inventory(d, m)
            for d, m in (("numpy", "numpy"), ("Pillow", "PIL"), ("scipy", "scipy"))
        },
        prepared_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    for cell in cases:
        scenario = scenario_for(root, plan, cell)
        for name in spec["policies"]:
            policy_for(root, plan, name, scenario)
    save(root / "plan.json", plan)
    return verify(root)


def verify(root):
    root = Path(root)
    plan = read(root / "plan.json")
    require(plan["schema_version"] == "mission_joint_constraints_v1", "Unknown joint study")
    spec = specification(FactorialSpec.from_dict(plan["factors"]))
    for key, expected in spec.items():
        if key != "schema_version":
            require(plan[key] == expected, "Changed factorial specification: " + key)
    for field in ("input_sha256", "implementation_sha256"):
        for path, expected in plan[field].items():
            require(sha(path) == expected, "Changed frozen input/source: " + path)
    for relative, expected in plan["copied_input_sha256"].items():
        path = (root / relative).resolve()
        require(
            path.is_relative_to(root.resolve()) and sha(path) == expected,
            "Changed or escaped copied input: " + relative,
        )
    require(sha(plan["parent_plan"]) == plan["parent_plan_sha256"], "Changed parent")
    parent = read(plan["parent_plan"])
    for key in (
        "cohort",
        "assets",
        "tracking_cost",
        "local_latency_s",
        "model_signatures",
        "calibration",
        "calibration_sha256",
    ):
        require(plan[key] == parent[key], "Changed inherited calibration/physical source: " + key)
    arm = calibrated_arm(read(Path(plan["calibration"]) / "profile.json"), plan["model_signatures"])
    require(
        arm["latency_s"] == plan["local_latency_s"]
        and arm["tracking_cost"] == plan["tracking_cost"],
        "Calibration assignment differs",
    )
    require(
        plan["actions_by_scale"]
        == {
            str(s): scaled_actions(parent["actions"], s)
            for s in plan["factors"]["onboard_latency_scales"]
        },
        "Changed local scale or remote costs",
    )
    require(
        plan["cases"] == expected_cases(spec, read(DEVELOPMENT / "protocol.json")),
        "Changed factorial lineage",
    )
    for cell, case in plan["cases"].items():
        original = root / "originals" / case["source_flight"] / case["mission"]
        values = derivative_documents(
            *(read(original / (n + ".json")) for n in ("scenario", "intent", "contract")),
            plan["local_latency_s"],
            case["initial_battery_frac"],
            case["onboard_latency_scale"],
        )
        for name, expected in zip(("scenario", "intent", "contract"), values, strict=True):
            require(
                read(root / "missions" / cell / case["mission"] / (name + ".json")) == expected,
                "Undeclared scenario, intent or resource intervention",
            )
    return plan


def expected_local_groups(plan):
    f = plan["factors"]
    return (
        len(f["flights"])
        * len(f["missions"])
        * len(f["batteries"])
        * len(f["onboard_latency_scales"])
        * sum(p in FIXED_LOCAL_POLICIES for p in plan["policies"])
    )


def execute(root, shard=None):
    from experiments.mission_network_study_v1.executors import PlacementExecutor
    from experiments.mission_network_study_v1.runtime import run_tracking_mission

    root = Path(root)
    plan = verify(root)
    require(shard is None or shard in plan["shards"], "Unknown factor shard")
    tag = "" if shard is None else "_" + shard
    grid = [r for r in plan["grid"] if shard is None or r["shard"] == shard]
    require(not (root / ("run_index" + tag + ".json")).exists(), "Preserve existing index")
    require(
        all(not (root / "runs" / r["cell"] / r["policy"]).exists() for r in grid),
        "Preserve every existing execution attempt",
    )
    api = imports()
    source, rows = Path(plan["cohort"]), []
    terrain, metadata = api["load_public_terrain"](Path(plan["assets"]))
    try:
        for number, item in enumerate(grid):
            verify(root)
            cell, source_flight, policy = item["cell"], item["flight"], item["policy"]
            scenario = scenario_for(root, plan, cell)
            folder = source / source_flight / "predictions"
            manifest = api["checked_manifest"](folder, source / "protocol.json")
            require(manifest["recording_identity"] == scenario.recording_identity, "Changed flight")
            trace, costs = trace_for(plan, cell, scenario), policy_costs(plan, scenario)
            executors = {
                a.action_id: PlacementExecutor(
                    a,
                    api["VerifiedMasks"](folder, manifest["models"][a.source_config_id]),
                    sha(folder / "manifest.json"),
                    trace,
                    costs,
                )
                for a in actions_for(plan, scenario)
            }
            destination = root / "runs" / cell / policy
            contract = api["IntentContract"](
                read(root / "missions" / cell / item["mission"] / "intent.json")
            )
            backend = api["ReplayBackend"](scenario, source / source_flight / "recording")
            try:
                outcome = run_tracking_mission(
                    scenario,
                    backend,
                    policy_for(root, plan, policy, scenario),
                    executors,
                    contract,
                    destination,
                    config=api["TrackingConfig"](remember_reports=True),
                    terrain=terrain,
                    map_metadata=metadata,
                    router_name=policy,
                    cost=tracking_cost(plan),
                    trace=trace,
                    costs=costs,
                    study_plan_sha256=sha(root / "plan.json"),
                )
            finally:
                backend.close()
            result, events = (
                read(destination / ("execution/" + n + ".json")) for n in ("results", "events")
            )
            row = dict(
                **item,
                path=str(destination.relative_to(root)),
                full_success=outcome["mission_success"],
                calls=len(events),
                phases_succeeded=sum(p["success"] for p in outcome["phases"]),
                report_recall=outcome["reporting"]["recall_by_deadline"],
                failed_constraints=outcome["failed_constraints"],
                expected_transport_timeouts=result["expected_transport_timeouts"],
                communication_mb=result["cumulative_communication_mb"],
                final_battery=result["final_battery_frac"],
                selections=dict(Counter(e["execution"]["config_id"] for e in events)),
            )
            for name, path in dict(
                outcome="outcome.json",
                events="execution/events.json",
                results="execution/results.json",
                protocol="protocol.json",
            ).items():
                row[name + "_sha256"] = sha(destination / path)
            rows.append(row)
            save(
                root / ("progress" + tag + ".json"),
                dict(completed=len(rows), expected=len(grid), cases=rows),
            )
            require(
                result["status"] == "completed"
                and result["router_errors"] == 0
                and result["invalid_actions"] == 0
                and result["inference_failures"] == result["expected_transport_timeouts"],
                "Unexpected internal failure; retain complete attempt, no rerun",
            )
            print(f"{number + 1}/{len(grid)} {cell} {policy}: {len(events)} decisions", flush=True)
    except BaseException as exc:
        save(
            root / ("failure" + tag + ".json"),
            dict(
                error=type(exc).__name__,
                message=str(exc),
                completed=len(rows),
                cases=rows,
                plan_sha256=sha(root / "plan.json"),
            ),
        )
        raise
    save(
        root / ("run_index" + tag + ".json"),
        dict(
            complete_execution_grid=shard is None,
            complete_shard=shard is not None,
            shard=shard,
            cases=rows,
            expected_replays=len(grid),
            plan_sha256=sha(root / "plan.json"),
        ),
    )
    verify(root)
    return rows


def combine_shards(plan, shards, plan_sha256):
    require(len(shards) == len(plan["shards"]), "Every declared factor shard required")
    require({s["shard"] for s in shards} == set(plan["shards"]), "Missing or duplicate shard")
    rows = []
    for shard in shards:
        expected = [r for r in plan["grid"] if r["shard"] == shard["shard"]]
        require(
            shard["complete_shard"] is True
            and shard["complete_execution_grid"] is False
            and shard["plan_sha256"] == plan_sha256
            and shard["expected_replays"] == len(shard["cases"]) == len(expected),
            "Incomplete or foreign factor shard",
        )
        require(all(r["shard"] == shard["shard"] for r in shard["cases"]), "Mixed factor shard")
        rows.extend(shard["cases"])
    lookup = {(r["cell"], r["policy"]): r for r in rows}
    expected = [(r["cell"], r["policy"]) for r in plan["grid"]]
    require(
        len(rows) == len(lookup) == len(expected) and set(lookup) == set(expected),
        "Missing, duplicated or foreign factorial execution",
    )
    for item in plan["grid"]:
        row = lookup[(item["cell"], item["policy"])]
        require({k: row[k] for k in item} == item, "Changed factor metadata")
    return [lookup[k] for k in expected]


def collect(root):
    root = Path(root)
    plan = verify(root)
    require(not (root / "run_index.json").exists(), "Preserve complete index")
    paths = [root / ("run_index_" + s + ".json") for s in plan["shards"]]
    rows = combine_shards(plan, [read(p) for p in paths], sha(root / "plan.json"))
    for row in rows:
        for name, filename in dict(
            outcome="outcome.json",
            events="execution/events.json",
            results="execution/results.json",
            protocol="protocol.json",
        ).items():
            require(
                sha(root / row["path"] / filename) == row[name + "_sha256"],
                "Changed shard execution",
            )
    save(
        root / "run_index.json",
        dict(
            complete_execution_grid=True,
            expected_replays=plan["expected_replays"],
            cases=rows,
            plan_sha256=sha(root / "plan.json"),
            shard_sha256={p.name: sha(p) for p in paths},
        ),
    )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("specify", "prepare", "verify", "execute", "collect", "finalize")
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--shard")
    parser.add_argument("--factors", type=Path, help="Explicit FactorialSpec JSON for a new study")
    args = parser.parse_args()
    if args.command == "specify":
        require(args.shard is None, "Shard applies only to execution")
        freeze_specification(
            args.output, FactorialSpec.from_dict(read(args.factors)) if args.factors else None
        )
    elif args.command == "execute":
        require(args.factors is None, "Frozen factors cannot change during execution")
        execute(args.output, args.shard)
    else:
        require(args.shard is None and args.factors is None, "Unexpected specification/shard")
        if args.command == "finalize":
            from experiments.mission_joint_constraints_analysis import finalize

            finalize(args.output)
        else:
            {"prepare": prepare, "verify": verify, "collect": collect}[args.command](args.output)


if __name__ == "__main__":
    main()
