"""Dense feasibility and legal fixed/alternating rotation-policy execution.

GT-aware bounds and mission scoring never enter deployable policy inputs.
Measured host timing is a separate sensitivity cohort; energy stays assumed.
"""

from __future__ import annotations

import argparse
import copy
import math
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from missionbench.v2.evaluation import MissionEvaluator
from missionbench.v2.scenario import ExecutorConfigSpec
from experiments.lake_feasibility_inference import read
from experiments.lake_feasibility_metrics import optimistic_bound, validate_table
from experiments.lake_missions.assets import save, sha
from experiments.lake_missions.objectives import score
from experiments.lake_missions.run import freeze
from experiments.lake_missions.scenario import load
from experiments.lake_rotation_study import CONTROL, ENSEMBLE, KIND, ROTATED, candidate_specs
from experiments.routing_mission.experiment import router_for
from experiments.v4_benchmark.backends import ReplayBackend
from experiments.v4_benchmark.probe_cache import CachedExecutor, build_cached, rgb_key
from experiments.v4_benchmark.runner import evaluation_frame, run_mission


class AlternatingViews:
    """One fixed alternation per completed decision; no GT or lookahead."""

    def __init__(self):
        self.index = 0

    def select(self, observation):
        config = (CONTROL, ROTATED)[self.index % 2]
        self.index += 1
        return config


def prepare(output):
    protocol = read(output / "protocol.json")
    source = Path(protocol["source"])
    path = output / "evaluation_protocol.json"
    if path.exists():
        raise FileExistsError("Evaluation protocol already exists; use a fresh study")
    save(
        path,
        dict(
            schema_version="1.0",
            policies=["static:" + c for c in (CONTROL, ROTATED, ENSEMBLE)] + ["alternating_views"],
            cost_cohorts=["configured", "host_timing_sensitivity"],
            missions_sha256=sha(source / "missions.json"),
            policy_rule="Fixed selection or decision-count alternation; no GT or learned ranks",
            expected_mission_runs=4
            * 2
            * len(protocol["environments"])
            * len(read(source / "missions.json")["missions"]),
            analysis_implementation_sha256=sha(Path(__file__)),
            source_protocol_sha256=sha(output / "protocol.json"),
            success_claim="Legal successes only; no optimality or held-out superiority claim",
        ),
    )


def scenario_document(source_data, specs, mission, latencies):
    data = copy.deepcopy(source_data)
    configs = [asdict(replace(s, mission_latency_s=latencies[s.config_id])) for s in specs]
    data["executor_configs"] = configs
    data["public_profiles"] = [
        dict(
            config_id=s["config_id"],
            expected_latency_ms=s["mission_latency_s"] * 1000,
            expected_upload_mb=0,
            quality_tier=s["quality_tier"],
        )
        for s in configs
    ]
    base_resource = data["resources"]["models"][CONTROL]
    data["resources"]["models"] = {s.config_id: copy.deepcopy(base_resource) for s in specs}
    # The ensemble assumes sequential passes with one resident weight set.
    data["mission"]["fallback_config_id"] = CONTROL
    data["mission"]["initial_battery_frac"] = mission["initial_battery_frac"]
    return data


def load_pipeline_scenario(path, environment, assets):
    """Validate common fields with the frozen schema, retaining custom pipeline kinds.

    The saved scenario fails in the generic neural factory by design. Only the
    explicit verified-cache path executes these custom preprocessing pipelines.
    """
    data = read(path)
    validation = copy.deepcopy(data)
    for spec in validation["executor_configs"]:
        if spec["kind"] == KIND:
            views = spec["parameters"].get("fixed_views_ccw")
            if views not in ("90", "0,90"):
                raise ValueError("Unsupported fixed rotation pipeline")
            spec["kind"] = "torch_instance_segmentation"
    validation["contract_path"] = str((path.parent / data["contract_path"]).resolve())
    with tempfile.TemporaryDirectory(dir=path.parent) as directory:
        temporary = Path(directory) / "schema_only.json"
        save(temporary, validation)
        scenario = load(temporary, environment, assets)
    return replace(
        scenario,
        source=data,
        executors=tuple(ExecutorConfigSpec(**s) for s in data["executor_configs"]),
    )


def complete_cache(output, environment, scenario):
    protocol = read(output / "protocol.json")
    folder = output / environment / "predictions"
    complete = read(folder / "completion.json")
    manifest = read(folder / "manifest.json")
    assert complete["complete"] and complete["manifest_sha256"] == sha(folder / "manifest.json")
    assert manifest["protocol_sha256"] == sha(output / "protocol.json")
    assert manifest["inference_sha256"] == sha(folder / "inference.json")
    recording = protocol["environments"][environment]
    assert manifest["recording_sha256"] == recording["recording_sha256"]
    assert manifest["capture_count"] == complete["fresh_rotation_calls"] == recording["captures"]
    rows = read(folder / "inference.json")
    if len(rows) != recording["captures"]:
        raise ValueError("Dense inference log has missing or extra capture rows")
    control = next(s for s in scenario.executors if s.config_id == CONTROL)
    cached = build_cached(candidate_specs(control), folder, scenario.recording_identity)
    return cached, rows


def host_latencies(protocol, measurements):
    rotation = max(
        float(np.percentile([r["measured_rotation_call_s"] for r in rows], 95))
        for rows in measurements.values()
    )
    union = max(
        float(np.percentile([r["measured_union_s"] for r in rows], 95))
        for rows in measurements.values()
    )
    control = protocol["original_control_host_p95_s"]
    return {
        k: math.ceil(v * 100) / 100
        for k, v in {
            CONTROL: control,
            ROTATED: rotation,
            ENSEMBLE: control + rotation + union,
        }.items()
    }


def analyze(output):
    protocol = read(output / "protocol.json")
    evaluation_protocol = read(output / "evaluation_protocol.json")
    assert evaluation_protocol["source_protocol_sha256"] == sha(output / "protocol.json")
    assert evaluation_protocol["analysis_implementation_sha256"] == sha(Path(__file__))
    assert protocol["implementation_sha256"] == sha(
        Path(__file__).with_name("lake_rotation_study.py")
    )
    source, prior = Path(protocol["source"]), Path(protocol["prior"])
    freeze(source)
    assert protocol["source_lock_sha256"] == sha(source / "lock.json")
    suite = read(source / "suite.json")
    assets = Path(suite["assets"])
    missions = read(source / "missions.json")["missions"]
    measurements, bounds, caches, scenarios = {}, {}, {}, {}
    for environment in protocol["environments"]:
        scenario = load(source / "scenario.json", suite["environments"][environment], assets)
        scenarios[environment] = scenario
        new, measurements[environment] = complete_cache(output, environment, scenario)
        old_folder = prior / environment / "predictions"
        assert (
            sha(old_folder / "manifest.json")
            == protocol["environments"][environment]["prior_manifest_sha256"]
        )
        old = build_cached(scenario.executors, old_folder, scenario.recording_identity)
        cached = {**old, **new}
        caches[environment] = cached
        scorers = {
            c: MissionEvaluator(
                scenario.matching_iou_threshold,
                len(scenario.targets),
                scenario.observation_interval_s,
            )
            for c in cached
        }
        backend = ReplayBackend(scenario, source / environment / "recording")
        rows, times = [], []
        try:
            for index, timestamp in enumerate(backend.frames):
                backend.advance_to(timestamp)
                frame = backend.capture()
                key = rgb_key(frame.rgb)
                t = (timestamp - scenario.schedule.epoch_ns) / 1e9
                times.append(t)
                inference = measurements[environment][index]
                assert inference["capture_id"] == index and inference["timestamp_ns"] == timestamp
                assert inference["rgb_sha256"] == key and inference["capture_time_s"] == t
                truth = evaluation_frame(frame, index, t, scenario)
                assert np.array_equal(
                    cached[ENSEMBLE].predictions[key],
                    cached[CONTROL].predictions[key] | cached[ROTATED].predictions[key],
                )
                for config, executor in cached.items():
                    rows.append(
                        dict(
                            capture_id=index,
                            config_id=config,
                            **scorers[config].update(truth, executor.predictions[key]).to_dict(),
                        )
                    )
        finally:
            backend.close()
        validate_table(rows, times, list(cached))
        save(output / environment / "outcomes.json", rows)
        bounds[environment] = [optimistic_bound(m, rows, times, list(cached)) for m in missions]
        save(output / environment / "optimistic_bounds.json", bounds[environment])
        for b in bounds[environment]:
            print(f"{environment}/{b['mission_id']}: {b['conclusion']}", flush=True)
    latencies = host_latencies(protocol, measurements)
    save(
        output / "costs.json",
        dict(
            host_latency_sensitivity=latencies,
            host_estimation="Marginal p95 components summed for ensemble, rounded up to 10 ms",
            caveat="Sum of marginal p95 values is not a measured joint p95 or a guarantee",
            energy="Assumed 30 J per constituent; ensemble 60 J; not electrically measured",
            memory="Configured 2400 MB per configuration; sequential ensemble weight reuse assumed",
            model_loading="Configured 0.3 s and 2 J on first use/switch; no warmup profiling claim",
        ),
    )
    comparisons = []
    for environment, base in scenarios.items():
        control = next(s for s in base.executors if s.config_id == CONTROL)
        specs = [control, *candidate_specs(control)]
        for mode, costs in (
            ("configured", {s.config_id: s.mission_latency_s for s in specs}),
            ("host_timing_sensitivity", latencies),
        ):
            for mission in missions:
                case = output / environment / "missions" / mode / mission["mission_id"]
                case.mkdir(parents=True, exist_ok=False)
                data = scenario_document(read(source / "scenario.json"), specs, mission, costs)
                contract = read(source / "contract.json")
                contract["min_final_battery_frac"] = mission["min_final_battery_frac"]
                save(case / "scenario.json", data)
                save(case / "contract.json", contract)
                scenario = load_pipeline_scenario(
                    case / "scenario.json", suite["environments"][environment], assets
                )
                assert scenario.recording_identity == base.recording_identity
                for policy in [*("static:" + s.config_id for s in specs), "alternating_views"]:
                    destination = case / policy.replace(":", "_")
                    executors = {
                        s.config_id: CachedExecutor(
                            s,
                            caches[environment][s.config_id].predictions,
                            caches[environment][s.config_id].manifest_sha,
                        )
                        for s in scenario.executors
                    }
                    router = (
                        AlternatingViews()
                        if policy == "alternating_views"
                        else router_for(policy, scenario, suite["seed"])
                    )
                    backend = ReplayBackend(scenario, source / environment / "recording")
                    canonical = {t: f["body_pose"] for t, f in backend.frames.items()}
                    try:
                        result = run_mission(
                            scenario,
                            backend,
                            router,
                            executors,
                            router_name=policy,
                            output=destination,
                        )
                    finally:
                        backend.close()
                    events = read(destination / "events.json")
                    assert all(
                        e["capture_body_pose"] == canonical[e["capture_timestamp_ns"]]
                        for e in events
                    )
                    objective = score(mission, events, result)
                    if mode == "configured" and policy == "static:" + CONTROL:
                        previous = next(
                            r
                            for r in read(prior / environment / "baselines.json")["rows"]
                            if r["mission_id"] == mission["mission_id"] and r["policy"] == policy
                        )
                        previous_path = Path(previous["output"])
                        for filename in ("events", "results"):
                            assert (
                                sha(previous_path / f"{filename}.json")
                                == previous[f"{filename}_sha256"]
                            )
                        previous_objective = score(
                            mission,
                            read(previous_path / "events.json"),
                            read(previous_path / "results.json"),
                        )
                        if objective != previous_objective:
                            raise ValueError("Configured control differs from the frozen baseline")
                    save(destination / "objectives.json", objective)
                    comparisons.append(
                        dict(
                            environment=environment,
                            cost_mode=mode,
                            mission_id=mission["mission_id"],
                            policy=policy,
                            output=str(destination.resolve()),
                            mission_success=objective["mission_success"],
                            phase_success_rate=objective["phase_success_rate"],
                            weighted_evidence_recall=objective["weighted_evidence_recall"],
                            resource_success=objective["resource_success"],
                            execution_success=objective["execution_success"],
                            final_battery_frac=result["final_battery_frac"],
                            failed_phases=[
                                p["phase_id"] for p in objective["phases"] if not p["success"]
                            ],
                            results_sha256=sha(destination / "results.json"),
                            events_sha256=sha(destination / "events.json"),
                            objectives_sha256=sha(destination / "objectives.json"),
                            exact_capture_poses=True,
                            previous_control_reproduced=(
                                True
                                if mode == "configured" and policy == "static:" + CONTROL
                                else None
                            ),
                        )
                    )
                    save(output / "comparisons.json", comparisons)
                    print(
                        f"{environment}/{mode}/{mission['mission_id']}/{policy}: "
                        f"success={objective['mission_success']}",
                        flush=True,
                    )
                    if not objective["execution_success"]:
                        raise RuntimeError("Mission execution error; retained partial results")
    summary = dict(
        schema_version="1.0",
        scope=protocol["scope"],
        protocol_sha256=sha(output / "protocol.json"),
        analysis_sha256=sha(Path(__file__)),
        new_rotation_calls=sum(len(v) for v in measurements.values()),
        evaluated_configurations=7,
        full_captures_per_environment={e: len(v) for e, v in measurements.items()},
        bounds=bounds,
        comparisons=comparisons,
        full_mission_successes=sum(r["mission_success"] for r in comparisons),
        note="Legal successes witness feasibility, not optimality or held-out superiority",
        energy_measured=False,
        optimal_policy_computed=False,
    )
    save(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    prepare(args.output) if args.prepare else analyze(args.output)
