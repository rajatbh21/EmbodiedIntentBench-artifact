"""Evaluate complete fast-model caches and legal policies with frozen lake semantics.

GT-aware stage bounds remain separate from fixed/periodic policy observations.
CPU latency is measured; energy, memory and switching remain declared assumptions.
"""

from __future__ import annotations

import argparse
import copy
import math
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from missionbench.v2.evaluation import MissionEvaluator
from missionbench.v2.scenario import ExecutorConfigSpec
from experiments.lake_fast_perception import FAMILIES, KIND
from experiments.lake_feasibility_inference import read
from experiments.lake_feasibility_metrics import optimistic_bound, validate_table
from experiments.lake_missions.assets import save, sha
from experiments.lake_missions.objectives import score
from experiments.lake_missions.run import freeze
from experiments.lake_missions.scenario import load
from experiments.lake_perception_recovery_report import summarize
from experiments.lake_rotation_evaluation import load_pipeline_scenario, scenario_document
from experiments.lake_rotation_latency_bound import latency_bound
from experiments.lake_rotation_study import CONTROL, ENSEMBLE, ROTATED, candidate_specs
from experiments.routing_mission.experiment import router_for
from experiments.v4_benchmark.backends import ReplayBackend
from experiments.v4_benchmark.probe_cache import CachedExecutor, build_cached, rgb_key
from experiments.v4_benchmark.runner import evaluation_frame, run_mission


class PeriodicViews:
    """Fixed alternation; cannot read evaluator outputs or future observations."""

    def __init__(self, configs):
        if len(configs) != 2 or configs[0] == configs[1]:
            raise ValueError("Two distinct configurations required")
        self.configs, self.index = tuple(configs), 0

    def select(self, observation):
        config = self.configs[self.index % 2]
        self.index += 1
        return config


def policies(configs):
    return ["static:" + c for c in configs] + ["alternate:" + f for f, _, _ in FAMILIES]


def prepare(output):
    p = read(output / "protocol.json")
    path = output / "evaluation_protocol.json"
    if path.exists():
        raise FileExistsError(path)
    configs = [CONTROL, ROTATED, ENSEMBLE] + [s["config_id"] for s in p["configurations"]]
    save(
        path,
        dict(
            schema_version="1.0",
            implementation_sha256=sha(Path(__file__)),
            source_protocol_sha256=sha(output / "protocol.json"),
            policies=policies(configs),
            timing="Frozen prior host profiles plus new max-environment p95, rounded up to 10 ms",
            energy="Assumed 30 J per pass; 60 J ensemble; no speed-based energy discount",
            costs_scope="Laptop host timing sensitivity; no deployment energy claim",
            expected_mission_runs=p["expected_mission_runs"],
            bounds="Exact stage relaxation for all nine configurations; not a feasible policy",
            missions_sha256=sha(Path(p["source"]) / "missions.json"),
        ),
    )


def host_costs(measurements, specs, old):
    """Cost independent of ground truth or mission outcomes; never use cache timing."""
    costs = dict(old)
    for s in specs:
        if s.parameters["fixed_views_ccw"] == "0,90":
            continue
        costs[s.config_id] = max(
            float(
                np.percentile(
                    [r["measured_wall_clock_s"] for r in rows if r["config_id"] == s.config_id], 95
                )
            )
            for rows in measurements.values()
        )
    for family, _, _ in FAMILIES:
        union = max(
            float(
                np.percentile(
                    [r["union_s"] for r in rows if r["config_id"] == f"CFG_{family}_ROT0_90"], 95
                )
            )
            for rows in measurements.values()
        )
        costs[f"CFG_{family}_ROT0_90"] = (
            costs[f"CFG_{family}_ROT0"] + costs[f"CFG_{family}_ROT90"] + union
        )
    return {c: math.ceil(v * 100) / 100 for c, v in costs.items()}


def load_fast_scenario(path, environment, assets, control):
    """Validate shared schema using a known backend, then restore explicit custom specs."""
    original = read(path)
    validation = copy.deepcopy(original)
    for s in validation["executor_configs"]:
        if s["kind"] == KIND:
            if s["parameters"]["fixed_views_ccw"] not in ("0", "90", "0,90"):
                raise ValueError("Unsupported fixed view")
            s["kind"], s["parameters"] = control.kind, dict(control.parameters)
    validation["contract_path"] = str((path.parent / original["contract_path"]).resolve())
    with tempfile.TemporaryDirectory(dir=path.parent) as folder:
        temporary = Path(folder) / "validation.json"
        save(temporary, validation)
        scenario = load_pipeline_scenario(temporary, environment, assets)
    return replace(
        scenario,
        source=original,
        executors=tuple(ExecutorConfigSpec(**s) for s in original["executor_configs"]),
    )


def measurements_for(output, p):
    result = {}
    specs = [s["config_id"] for s in p["configurations"]]
    for env, meta in p["environments"].items():
        folder = output / env / "predictions"
        m, c = read(folder / "manifest.json"), read(folder / "completion.json")
        assert c["complete"] and c["manifest_sha256"] == sha(folder / "manifest.json")
        assert m["protocol_sha256"] == sha(output / "protocol.json")
        assert m["recording_sha256"] == meta["recording_sha256"]
        assert m["inference_sha256"] == sha(folder / "inference.json")
        assert m["capture_count"] == meta["captures"]
        rows = read(folder / "inference.json")
        keys = [(r["capture_id"], r["config_id"]) for r in rows]
        expected = {(i, config) for i in range(meta["captures"]) for config in specs}
        if len(keys) != len(expected) or set(keys) != expected:
            raise ValueError("Incomplete or duplicate dense inference log")
        assert c["actual_neural_calls"] == meta["captures"] * 4
        assert c["derived_ensembles"] == meta["captures"] * 2
        result[env] = rows
    return result


def evaluate(output):
    p, ep = read(output / "protocol.json"), read(output / "evaluation_protocol.json")
    assert ep["source_protocol_sha256"] == sha(output / "protocol.json")
    assert ep["implementation_sha256"] == sha(Path(__file__))
    assert p["implementation_sha256"] == sha(Path(__file__).with_name("lake_fast_perception.py"))
    source, prior = Path(p["source"]), Path(p["prior"])
    freeze(source)
    assert p["source_lock_sha256"] == sha(source / "lock.json")
    assert p["prior_summary_sha256"] == sha(prior / "summary.json")
    assert p["prior_costs_sha256"] == sha(prior / "costs.json")
    assert ep["missions_sha256"] == sha(source / "missions.json")
    suite = read(source / "suite.json")
    missions = read(source / "missions.json")["missions"]
    specs = [ExecutorConfigSpec(**s) for s in p["configurations"]]
    measurements = measurements_for(output, p)
    costs = host_costs(measurements, specs, read(prior / "costs.json")["host_latency_sensitivity"])
    save(
        output / "costs.json",
        dict(
            host_latency_sensitivity=costs,
            energy_measured=False,
            estimation="Maximum environment p95 per view; sum plus union for ensembles; ceil 10 ms",
            caveat="Sum of marginal p95 is not joint p95; energy and resources remain assumed",
        ),
    )
    prior_protocol = read(prior / "protocol.json")
    original = Path(prior_protocol["prior"])
    comparisons, bounds, diagnostics = [], {}, {}
    for env in p["environments"]:
        base = load(source / "scenario.json", suite["environments"][env], Path(suite["assets"]))
        control = next(s for s in base.executors if s.config_id == CONTROL)
        all_specs = [control, *candidate_specs(control), *specs]
        old_folder = original / env / "predictions"
        assert (
            sha(old_folder / "manifest.json")
            == prior_protocol["environments"][env]["prior_manifest_sha256"]
        )
        old = build_cached(base.executors, old_folder, base.recording_identity)
        cached = {CONTROL: old[CONTROL]}
        del old
        cached.update(
            build_cached(
                candidate_specs(control), prior / env / "predictions", base.recording_identity
            )
        )
        cached.update(build_cached(specs, output / env / "predictions", base.recording_identity))
        scorers = {
            c: MissionEvaluator(
                base.matching_iou_threshold, len(base.targets), base.observation_interval_s
            )
            for c in cached
        }
        lookup = {(r["capture_id"], r["config_id"]): r for r in measurements[env]}
        backend = ReplayBackend(base, source / env / "recording")
        assert backend.provenance["recording_sha256"] == p["environments"][env]["recording_sha256"]
        rows, times = [], []
        try:
            for i, timestamp in enumerate(backend.frames):
                backend.advance_to(timestamp)
                frame = backend.capture()
                t, key = (timestamp - base.schedule.epoch_ns) / 1e9, rgb_key(frame.rgb)
                times.append(t)
                truth = evaluation_frame(frame, i, t, base)
                for family, _, _ in FAMILIES:
                    assert np.array_equal(
                        cached[f"CFG_{family}_ROT0_90"].predictions[key],
                        cached[f"CFG_{family}_ROT0"].predictions[key]
                        | cached[f"CFG_{family}_ROT90"].predictions[key],
                    )
                for config, executor in cached.items():
                    timing = costs[config]
                    if (i, config) in lookup:
                        measured = lookup[i, config]
                        assert measured["rgb_sha256"] == key
                        assert measured["timestamp_ns"] == timestamp
                        assert measured["capture_time_s"] == t
                        timing = measured["measured_wall_clock_s"]
                    rows.append(
                        dict(
                            capture_id=i,
                            config_id=config,
                            measured_wall_clock_s=timing,
                            **scorers[config].update(truth, executor.predictions[key]).to_dict(),
                        )
                    )
        finally:
            backend.close()
        validate_table(rows, times, list(cached))
        save(output / env / "outcomes.json", rows)
        diagnostics[env] = {
            c: summarize([r for r in rows if r["config_id"] == c], base.targets.values())
            for c in cached
        }
        # Old rows use their frozen latency profiles, not fresh measurements.
        for c in (CONTROL, ROTATED, ENSEMBLE):
            diagnostics[env][c]["timing_scope"] = "Prior frozen host p95, no fresh timing samples"
        bounds[env] = []
        for mission in missions:
            phases = [
                latency_bound(phase, rows, base.schedule, costs, base.router_latency_s)
                for phase in mission["phases"]
            ]
            bounds[env].append(
                dict(
                    mission_id=mission["mission_id"],
                    phases=phases,
                    conclusion="certified_latency_infeasible"
                    if any(ph["cannot_meet_recall"] for ph in phases)
                    else "feasibility_unresolved",
                    unlimited=optimistic_bound(mission, rows, times, list(cached)),
                )
            )
        save(output / env / "bounds.json", bounds[env])
        for mission in missions:
            case = output / env / "missions" / mission["mission_id"]
            case.mkdir(parents=True, exist_ok=False)
            save(
                case / "scenario.json",
                scenario_document(read(source / "scenario.json"), all_specs, mission, costs),
            )
            contract = read(source / "contract.json")
            contract["min_final_battery_frac"] = mission["min_final_battery_frac"]
            save(case / "contract.json", contract)
            scenario = load_fast_scenario(
                case / "scenario.json", suite["environments"][env], Path(suite["assets"]), control
            )
            assert scenario.recording_identity == base.recording_identity
            assert policies(list(cached)) == ep["policies"]
            for policy in ep["policies"]:
                destination = case / policy.replace(":", "_")
                if policy.startswith("alternate:"):
                    family = policy.split(":", 1)[1]
                    router = PeriodicViews([f"CFG_{family}_ROT0", f"CFG_{family}_ROT90"])
                else:
                    router = router_for(policy, scenario, suite["seed"])
                executors = {
                    s.config_id: CachedExecutor(
                        s, cached[s.config_id].predictions, cached[s.config_id].manifest_sha
                    )
                    for s in scenario.executors
                }
                backend = ReplayBackend(scenario, source / env / "recording")
                canonical = {t: f["body_pose"] for t, f in backend.frames.items()}
                try:
                    result = run_mission(
                        scenario, backend, router, executors, router_name=policy, output=destination
                    )
                finally:
                    backend.close()
                events = read(destination / "events.json")
                assert all(
                    e["capture_body_pose"] == canonical[e["capture_timestamp_ns"]] for e in events
                )
                objective = score(mission, events, result)
                reproduced = None
                if policy in ["static:" + c for c in (CONTROL, ROTATED, ENSEMBLE)]:
                    previous = next(
                        r
                        for r in read(prior / "summary.json")["comparisons"]
                        if r["environment"] == env
                        and r["policy"] == policy
                        and r["mission_id"] == mission["mission_id"]
                        and r["cost_mode"] == "host_timing_sensitivity"
                    )
                    old_path = Path(previous["output"]) / "objectives.json"
                    assert sha(old_path) == previous["objectives_sha256"]
                    assert objective == read(old_path)
                    reproduced = True
                save(destination / "objectives.json", objective)
                comparisons.append(
                    dict(
                        environment=env,
                        mission_id=mission["mission_id"],
                        policy=policy,
                        mission_success=objective["mission_success"],
                        phase_success_rate=objective["phase_success_rate"],
                        resource_success=objective["resource_success"],
                        execution_success=objective["execution_success"],
                        final_battery_frac=result["final_battery_frac"],
                        failed_phases=[
                            ph["phase_id"] for ph in objective["phases"] if not ph["success"]
                        ],
                        output=str(destination.resolve()),
                        exact_capture_poses=True,
                        previous_control_reproduced=reproduced,
                        **{
                            f"{n}_sha256": sha(destination / f"{n}.json")
                            for n in ("results", "events", "objectives")
                        },
                    )
                )
                save(output / "comparisons.json", comparisons)
                print(
                    f"{env}/{mission['mission_id']}/{policy}: "
                    f"success={objective['mission_success']} "
                    f"phases={objective['phase_success_rate']}",
                    flush=True,
                )
                assert objective["execution_success"]
        del cached
    assert len(comparisons) == ep["expected_mission_runs"]
    save(
        output / "summary.json",
        dict(
            schema_version="1.0",
            scope=p["scope"],
            diagnostics=diagnostics,
            bounds=bounds,
            comparisons=comparisons,
            full_mission_successes=sum(r["mission_success"] for r in comparisons),
            fresh_neural_calls=1424,
            derived_ensemble_masks=712,
            evaluated_pairs=3204,
            protocol_sha256=sha(output / "protocol.json"),
            evaluation_protocol_sha256=sha(output / "evaluation_protocol.json"),
            implementation_sha256=sha(Path(__file__)),
            bound_implementation_sha256=sha(
                Path(__file__).with_name("lake_rotation_latency_bound.py")
            ),
            energy_measured=False,
            optimal_policy_computed=False,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    prepare(args.output) if args.prepare else evaluate(args.output)
