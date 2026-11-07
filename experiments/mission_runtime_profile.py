"""Development-only policy overhead, with paired uninstrumented causal controls.

This diagnostic never substitutes measured host time into the frozen simulator.
It creates only results/mission_runtime_profile_v1_20261027 and refuses overwrite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "results/mission_runtime_profile_v1_20261027"
DEVELOPMENT = REPO / "results/mission_policy_selection_v1_20261018"
FLIGHTS = ("east_shore_day", "west_rocks_day")
MISSIONS = ("rapid_survey", "verified_inspection")
POLICIES = (
    "coverage_tail",
    "phase_survival",
    "mission_survival",
    "temporal_mission",
    "round_robin",
    "greedy_quality",
    "approxdet_eq5_adapter",
    "static:CFG_YOLO11S_960_ROT0",
)
THREAD_ENV = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class TimedPolicy:
    """Observe one real select call; never copy/replay inputs or change its return."""

    def __init__(self, delegate, wall=time.perf_counter_ns, cpu=time.process_time_ns):
        self.delegate, self.wall, self.cpu = delegate, wall, cpu
        self.samples = []

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def select(self, observation):
        # Metadata work is outside the timed region. Only public causal fields.
        row = dict(
            decision_index=len(self.samples),
            instance_cold=not self.samples,
            capture_id=observation.state.frame_id,
            capture_time_s=observation.state.current_time_s,
            previous_completion_s=observation.evidence_history["last_completion_s"],
        )
        wall0, cpu0 = self.wall(), self.cpu()
        try:
            selected = self.delegate.select(observation)
        except BaseException as exc:
            cpu1, wall1 = self.cpu(), self.wall()
            row.update(error=type(exc).__name__, selected_config_id=None)
            row.update(wall_s=(wall1 - wall0) / 1e9, process_s=(cpu1 - cpu0) / 1e9)
            self.samples.append(row)
            raise
        cpu1, wall1 = self.cpu(), self.wall()
        row.update(error=None, selected_config_id=selected)
        row.update(wall_s=(wall1 - wall0) / 1e9, process_s=(cpu1 - cpu0) / 1e9)
        self.samples.append(row)
        return selected


def quantile(values, q):
    """Linear interpolation between ordered samples, matching NumPy's default."""
    if not values or not 0 <= q <= 1:
        raise ValueError("Nonempty samples and a probability required")
    values = sorted(values)
    at = (len(values) - 1) * q
    lo, hi = math.floor(at), math.ceil(at)
    return values[lo] + (values[hi] - values[lo]) * (at - lo)


def summarize(values, declared_s=0.05):
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Finite nonnegative timing samples required")
    return dict(
        n=len(values),
        min_s=min(values),
        mean_s=statistics.fmean(values),
        p50_s=quantile(values, 0.5),
        p95_s=quantile(values, 0.95),
        p99_s=quantile(values, 0.99),
        max_s=max(values),
        above_declared_count=sum(v > declared_s for v in values),
        above_declared_fraction=sum(v > declared_s for v in values) / len(values),
        declared_s=declared_s,
    )


def semantic_artifacts(folder):
    """Strip exactly nondeterministic timings and hashes derived from those times."""
    folder = Path(folder)
    events = read(folder / "execution/events.json")
    for event in events:
        event.pop("measured_router_wall_clock_s")
        event["execution"].pop("measured_wall_clock_s")
    result = read(folder / "execution/results.json")
    outcome = read(folder / "outcome.json")
    outcome.pop("events_sha256")
    outcome.pop("execution_sha256")
    return dict(events=events, results=result, outcome=outcome)


def compare_pair(control, profiled, samples):
    left, right = semantic_artifacts(control), semantic_artifacts(profiled)
    if left != right:
        changed = [key for key in left if left[key] != right[key]]
        raise ValueError("Instrumentation changed causal execution: " + ", ".join(changed))
    events = right["events"]
    expected = [(e["observation_id"], e["execution"]["config_id"]) for e in events]
    actual = [(s["capture_id"], s["selected_config_id"]) for s in samples]
    if actual != expected or any(s["error"] is not None for s in samples):
        raise ValueError("Timing samples do not match completed selected actions")
    if [s["instance_cold"] for s in samples] != [True] + [False] * (len(samples) - 1):
        raise ValueError("Each fresh policy has exactly one instance-cold decision")
    result = right["results"]
    if result["status"] != "completed" or any(
        result[k] for k in ("router_errors", "inference_failures", "invalid_actions")
    ):
        raise ValueError("Cannot accept an execution error as timing parity")
    return dict(
        passed=True,
        semantic_sha256=digest(left),
        decisions=len(events),
        exact_actions_schedules_states_traces_scores=True,
        exact_semantic_outcome=True,
        mission_success=right["outcome"]["mission_success"],
        excluded_event_fields=["measured_router_wall_clock_s", "execution.measured_wall_clock_s"],
        excluded_outcome_fields=["events_sha256", "execution_sha256"],
    )


def require_development(protocol, coverage):
    if (
        protocol.get("schema_version") != "mission_policy_selection_v1"
        or protocol.get("scope") != "development_selection_known_map"
        or protocol.get("fresh_test") is not False
    ):
        raise ValueError("Only the declared DEVELOPMENT selection protocol is permitted")
    training_lineages = coverage["training_lineage"]
    for name in FLIGHTS:
        case = protocol["cases"][name]
        if case.get("role") != "development" or case["lineage"] not in training_lineages:
            raise ValueError("Profiling recording must belong to DEVELOPMENT training")


def implementation_inventory():
    paths = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.suffix == ".py" and any(
            path.is_relative_to(REPO / tree) for tree in ("experiments", "missionbench")
        ):
            paths.add(path)
    return {str(p): sha(p) for p in sorted(paths)}


def host_state():
    status = Path("/proc/self/status").read_text().splitlines()
    return dict(
        load_average=list(os.getloadavg()),
        process_threads=int(next(x.split()[1] for x in status if x.startswith("Threads:"))),
        affinity=sorted(os.sched_getaffinity(0)),
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


def imports():
    """Delay numerical imports until thread variables and affinity are configured."""
    from experiments.lake_hindsight_witness import VerifiedMasks
    from experiments.lake_intent_contract import IntentContract
    from experiments.lake_tracking import TrackingConfig
    from experiments.lake_tracking_runner import run_mission
    from experiments.lake_tracking_runtime import load_public_terrain
    from experiments.mission_coverage_routing.train import verify as coverage_verify
    from experiments.mission_policy_selection.protocol import verify as development_verify
    from experiments.mission_policy_selection.study import scenario_for
    from experiments.mission_routing_v2.runtime import run_tracking_mission
    from experiments.mission_routing_v2.train import checked_manifest
    from experiments.mission_routing_v4.train import verify as risk_verify
    from experiments.mission_temporal_routing.train import verify as temporal_verify
    from experiments.mission_transfer_comparison_v4.study import make_policy
    from experiments.v4_benchmark.backends import ReplayBackend
    from experiments.v4_benchmark.calibrated_router import model_signature
    from experiments.v4_benchmark.probe_cache import CachedExecutor

    return dict(
        VerifiedMasks=VerifiedMasks,
        IntentContract=IntentContract,
        TrackingConfig=TrackingConfig,
        load_public_terrain=load_public_terrain,
        coverage_verify=coverage_verify,
        development_verify=development_verify,
        scenario_for=scenario_for,
        run_tracking_mission=run_tracking_mission,
        checked_manifest=checked_manifest,
        risk_verify=risk_verify,
        temporal_verify=temporal_verify,
        make_policy=make_policy,
        ReplayBackend=ReplayBackend,
        model_signature=model_signature,
        CachedExecutor=CachedExecutor,
        frozen_runner=run_mission,
    )


def prepare(output, api, runtime):
    if output.resolve() != OUTPUT.resolve():
        raise ValueError("Only the separate versioned profiling output is permitted")
    if output.exists():
        raise FileExistsError("Preserve the prior profiling run")
    p = api["development_verify"](DEVELOPMENT)
    base = REPO / "results"
    paths = dict(
        recall=base / "mission_policy_joint_training_v1_20261018/all_development/recall.json",
        risk=base / "mission_routing_v4_risk_training_20261017/risk_profile.json",
        temporal=base / "mission_temporal_training_v2_20261025/profile.json",
        coverage=base / "mission_coverage_training_v1_20261025/profile.json",
    )
    profiles = {key: read(path) for key, path in paths.items()}
    api["coverage_verify"](paths["coverage"].parent)
    api["risk_verify"](paths["risk"].parent)
    api["temporal_verify"](paths["temporal"].parent)
    require_development(p, profiles["coverage"])
    output.mkdir(parents=True)
    frozen = {str(DEVELOPMENT / "protocol.json"): sha(DEVELOPMENT / "protocol.json")}
    frozen.update({str(path): sha(path) for path in paths.values()})
    source = Path(p["cohort"])
    copied_inputs = {}
    for name in FLIGHTS:
        for filename in (
            "recording/recording.json",
            "predictions/manifest.json",
            "predictions/completion.json",
            "predictions/inference.json",
        ):
            path = source / name / filename
            frozen[str(path)] = sha(path)
        for mission in MISSIONS:
            dest = output / "missions" / name / mission
            dest.mkdir(parents=True)
            for filename in ("scenario.json", "contract.json", "intent.json"):
                path = DEVELOPMENT / name / mission / filename
                frozen[str(path)] = sha(path)
                copied = dest / filename
                shutil.copyfile(path, copied)
                if sha(copied) != frozen[str(path)]:
                    raise ValueError("Copied mission input differs from source")
                copied_inputs[str(copied.relative_to(output))] = sha(copied)
    frozen[str(source / "protocol.json")] = sha(source / "protocol.json")
    terrain, metadata = api["load_public_terrain"](Path(p["assets"]))
    for path in (Path(metadata["raster_path"]), Path(p["assets"]) / "harmonic_terrain/model.sdf"):
        frozen[str(path)] = sha(path)
    grid = [
        dict(flight=f, mission=m, policy=policy)
        for f in FLIGHTS
        for m in MISSIONS
        for policy in POLICIES
    ]
    plan = dict(
        schema_version="development_routing_runtime_profile_v1",
        purpose="development_only_host_overhead_diagnostic",
        evaluation_recordings_used=False,
        source_development_protocol=str(DEVELOPMENT / "protocol.json"),
        cohort=str(source),
        profiles={key: str(path) for key, path in paths.items()},
        grid=grid,
        expected_pairs=len(grid),
        expected_replays=2 * len(grid),
        grid_selection="Both development day layouts; both existing survey/inspection intents; "
        "seven distinct routing objectives and a fixed YOLO11S control",
        role="Known development recordings and fitted profiles; not predictive evaluation",
        declared_router_latency_s=0.05,
        initial_battery_frac=0.65,
        configured_power_w=15,
        simulated_costs_changed=False,
        measured_times_affect_policy=False,
        timing_boundary="Delegate policy.select: RGB features, guard and planning; "
        "excludes feedback adapter construction, prediction cache I/O and scoring",
        constructor_timing="Policy factory wall/process time; excludes profile disk loading",
        cold_definition="First select of a fresh instance; includes lazy state/budget setup. "
        "Not a cold process, CPU cache, filesystem cache or boot measurement",
        warm_definition="Subsequent select calls; no repeated-input microbenchmark",
        ordering="Fixed grid; even pairs control then profiled, odd pairs profiled then control",
        independent_units="Two development layouts; decisions and four contexts are dependent",
        confidence_intervals=False,
        performance_comparison=False,
        host_exclusivity=False,
        power_measured=False,
        runtime=runtime,
        input_sha256=frozen,
        copied_input_sha256=copied_inputs,
        implementation_sha256=implementation_inventory(),
        dependency_validation="Frozen development/training validators; ReplayBackend verifies "
        "each source frame and VerifiedMasks verifies each selected prediction",
    )
    save(output / "plan.json", plan)
    return p, profiles, terrain, metadata, plan


def verify_pins(plan, output=None):
    for field in ("input_sha256", "implementation_sha256"):
        for path, expected in plan[field].items():
            if sha(path) != expected:
                raise ValueError("Changed profiling source/input: " + path)
    for relative, expected in plan.get("copied_input_sha256", {}).items():
        if output is None:
            raise ValueError("Copied inputs require their output root")
        path = (output / relative).resolve()
        if not path.is_relative_to(output.resolve()) or sha(path) != expected:
            raise ValueError("Changed copied mission input: " + relative)


def execute(output, api, runtime):
    p, profiles, terrain, metadata, plan = prepare(output, api, runtime)
    source, pairs = Path(p["cohort"]), []
    for index, cell in enumerate(plan["grid"]):
        verify_pins(plan, output)
        flight, mission, policy = (cell[k] for k in ("flight", "mission", "policy"))
        destination = output / "runs" / flight / mission / policy.replace(":", "_")
        scenario = api["scenario_for"](output / "missions", p, flight, mission, 15)
        if scenario.initial_battery_frac != plan["initial_battery_frac"]:
            raise ValueError("Unexpected development battery condition")
        if scenario.router_latency_s != plan["declared_router_latency_s"]:
            raise ValueError("Unexpected declared router cost")
        folder = source / flight / "predictions"
        manifest = api["checked_manifest"](folder, source / "protocol.json")
        if manifest["recording_identity"] != scenario.recording_identity or manifest[
            "recording_sha256"
        ] != sha(source / flight / "recording/recording.json"):
            raise ValueError("Prediction bank does not belong to development recording")
        contract = api["IntentContract"](
            read(output / "missions" / flight / mission / "intent.json")
        )
        measurements = {}
        order = ("control", "profiled") if index % 2 == 0 else ("profiled", "control")
        for arm in order:
            executors = {}
            for spec in scenario.executors:
                model = manifest["models"][spec.config_id]
                if model["signature"] != api["model_signature"](spec):
                    raise ValueError("Model signature mismatch")
                executors[spec.config_id] = api["CachedExecutor"](
                    spec, api["VerifiedMasks"](folder, model), sha(folder / "manifest.json")
                )
            wall0, cpu0 = time.perf_counter_ns(), time.process_time_ns()
            router = api["make_policy"](policy, profiles, scenario)
            cpu1, wall1 = time.process_time_ns(), time.perf_counter_ns()
            construction = dict(wall_s=(wall1 - wall0) / 1e9, process_s=(cpu1 - cpu0) / 1e9)
            timed = TimedPolicy(router) if arm == "profiled" else None
            backend = api["ReplayBackend"](scenario, source / flight / "recording")
            before = host_state()
            if before["process_threads"] != 1:
                raise ValueError("Profiling requires one actual process thread")
            try:
                api["run_tracking_mission"](
                    scenario,
                    backend,
                    timed if timed is not None else router,
                    executors,
                    contract,
                    destination / arm,
                    config=api["TrackingConfig"](remember_reports=True),
                    terrain=terrain,
                    map_metadata=metadata,
                    router_name=policy,
                    gt_aware=False,
                )
            finally:
                backend.close()
            after = host_state()
            if after["process_threads"] != 1:
                raise ValueError("A replay created additional process threads")
            arm_events = read(destination / arm / "execution/events.json")
            measurements[arm] = dict(
                constructor=construction,
                host_before=before,
                host_after=after,
                samples=timed.samples if timed is not None else None,
                outer_router_wall_s=[e["measured_router_wall_clock_s"] for e in arm_events],
            )
        parity = compare_pair(
            destination / "control", destination / "profiled", measurements["profiled"]["samples"]
        )
        save(destination / "timing.json", measurements)
        save(destination / "parity.json", parity)
        row = dict(
            **cell,
            **parity,
            path=str(destination.relative_to(output)),
            timing_sha256=sha(destination / "timing.json"),
            parity_sha256=sha(destination / "parity.json"),
            execution_artifacts_sha256={
                f"{arm}/{name}": sha(destination / arm / name)
                for arm in ("control", "profiled")
                for name in (
                    "protocol.json",
                    "outcome.json",
                    "execution/events.json",
                    "execution/results.json",
                )
            },
        )
        pairs.append(row)
        print(
            f"{index + 1}/{len(plan['grid'])} {flight} {mission} {policy}: "
            f"{parity['decisions']} decisions; exact semantic parity",
            flush=True,
        )
    verify_pins(plan, output)
    summary = report(output, plan, pairs)
    save(output / "summary.json", summary)
    save(
        output / "completion.json",
        dict(
            complete=True,
            plan_sha256=sha(output / "plan.json"),
            summary_sha256=sha(output / "summary.json"),
            report_sha256=sha(output / "REPORT.md"),
            pairs=len(pairs),
            causal_replays=2 * len(pairs),
            profiled_decisions=sum(p["decisions"] for p in pairs),
            all_semantic_parity=True,
            host_after=host_state(),
        ),
    )
    return summary


def report(output, plan, pairs):
    by_policy = {}
    for policy in POLICIES:
        cells = [p for p in pairs if p["policy"] == policy]
        samples, constructor, outer_control = [], [], []
        for cell in cells:
            measurements = read(output / cell["path"] / "timing.json")
            samples.extend(measurements["profiled"]["samples"])
            constructor.append(measurements["profiled"]["constructor"])
            outer_control.extend(measurements["control"]["outer_router_wall_s"])
        groups = dict(
            all=samples,
            instance_cold=[s for s in samples if s["instance_cold"]],
            subsequent=[s for s in samples if not s["instance_cold"]],
        )
        by_policy[policy] = dict(
            pairs=len(cells),
            select={
                group: {
                    clock: summarize([s[clock] for s in rows]) for clock in ("wall_s", "process_s")
                }
                for group, rows in groups.items()
            },
            construction={
                clock: summarize([r[clock] for r in constructor])
                for clock in ("wall_s", "process_s")
            },
            uninstrumented_outer_router_wall=summarize(outer_control),
        )
    summary = dict(
        complete=True,
        schema_version="development_routing_runtime_summary_v1",
        plan_sha256=sha(output / "plan.json"),
        pairs=pairs,
        policies=by_policy,
        observations="Descriptive dependent samples on one shared CPU host; percentiles are "
        "empirical, not independent-trial confidence bounds or WCET guarantees",
        scope="Timing only; no mission-benefit, generalization, energy, or deployment claim",
        configured_budget_comparison="50 ms is unchanged accounting. Exceedances diagnose a "
        "cost mismatch; no counterfactual deadline failures inferred",
    )
    lines = [
        "# Development-only routing overhead",
        "",
        "The table measures real policy selection on two development recordings and two "
        "existing intents. Each of 32 profiled replays has an uninstrumented causal control. "
        "Actions, schedules, states, decision traces, scores and semantic endpoints must match.",
        "",
        "| Policy | Calls | First-call median (ms; n=4) | Subsequent wall p50/p95/p99 (ms) "
        "| Subsequent process p95 (ms) | All calls >50 ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for policy, row in by_policy.items():
        all_s = row["select"]["all"]["wall_s"]
        cold = row["select"]["instance_cold"]["wall_s"]
        warm = row["select"]["subsequent"]["wall_s"]
        cpu = row["select"]["subsequent"]["process_s"]
        lines.append(
            f"| {policy} | {all_s['n']} | {1000 * cold['p50_s']:.3f} | "
            f"{1000 * warm['p50_s']:.3f} / {1000 * warm['p95_s']:.3f} / "
            f"{1000 * warm['p99_s']:.3f} | {1000 * cpu['p95_s']:.3f} | "
            f"{all_s['above_declared_count']}/{all_s['n']} |"
        )
    lines += [
        "",
        "First-call means a fresh policy instance, not cold process/OS caches. "
        "Profile construction is measured separately in summary.json. The measured boundary "
        "includes RGB feature extraction, shared guard and planning inside select; it excludes "
        "the public-feedback adapter, cached inference, tracking and scoring. The unchanged "
        "runner also records the entire adapter+select wall time for both arms.",
        "",
        "One logical CPU and one numerical-library thread are configured; the shared laptop "
        "is not an exclusive timing host. Host load, thread controls and observed "
        "process thread counts are retained. "
        "All timings are host measurements. They do not replace the configured 50 ms routing "
        "latency, do not change paid energy, and do not establish an onboard deadline guarantee. "
        "No Lake/Depot evaluation recording or outcome was used; no model or threshold was fitted.",
        "",
        "Every cell is retained, including unsuccessful authored development missions. "
        "Their endpoints are parity checks, not new benchmark efficacy results.",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines))
    return summary


def verify(output):
    """Re-read every paired execution and its source locks, rather than trust pass flags."""
    plan, complete, summary = (
        read(output / n) for n in ("plan.json", "completion.json", "summary.json")
    )
    if not complete["complete"] or complete["plan_sha256"] != sha(output / "plan.json"):
        raise ValueError("Changed or incomplete profiling plan")
    for name, key in (("summary.json", "summary_sha256"), ("REPORT.md", "report_sha256")):
        if complete[key] != sha(output / name):
            raise ValueError("Changed profiling summary/report")
    verify_pins(plan, output)
    actual = [
        {key: pair[key] for key in ("flight", "mission", "policy")} for pair in summary["pairs"]
    ]
    if actual != plan["grid"] or len(actual) != complete["pairs"]:
        raise ValueError("Incomplete or changed profiling grid")
    decisions = 0
    for pair in summary["pairs"]:
        folder = output / pair["path"]
        for name, expected in pair["execution_artifacts_sha256"].items():
            if sha(folder / name) != expected:
                raise ValueError("Changed paired execution artifact")
        for name in ("timing", "parity"):
            if sha(folder / (name + ".json")) != pair[name + "_sha256"]:
                raise ValueError("Changed timing/parity artifact")
        timing = read(folder / "timing.json")
        audit = compare_pair(folder / "control", folder / "profiled", timing["profiled"]["samples"])
        if audit != read(folder / "parity.json"):
            raise ValueError("Parity certificate disagrees with execution")
        decisions += audit["decisions"]
    if decisions != complete["profiled_decisions"]:
        raise ValueError("Changed decision denominator")
    return dict(
        complete=True,
        pairs=len(actual),
        profiled_decisions=decisions,
        exact_semantic_parity=True,
        source_and_artifact_hashes_verified=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify(args.output), sort_keys=True))
        return
    for name in THREAD_ENV:
        os.environ[name] = "1"
    prior_affinity = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {prior_affinity[0]})
    api = imports()
    before = host_state()
    if before["process_threads"] != 1:
        raise ValueError("Numerical imports must leave exactly one process thread")
    cpuinfo = Path("/proc/cpuinfo").read_text()
    model = next(
        line.split(":", 1)[1].strip()
        for line in cpuinfo.splitlines()
        if line.startswith("model name")
    )
    runtime = dict(
        python=sys.version,
        executable=sys.executable,
        platform=platform.platform(),
        cpu_model=model,
        prior_affinity=prior_affinity,
        affinity=sorted(os.sched_getaffinity(0)),
        thread_environment={name: os.environ[name] for name in THREAD_ENV},
        thread_control="Environment set before numerical imports; one logical CPU; "
        "actual process thread count checked at each replay boundary",
        threadpool_api_queried=False,
        host_before=before,
        clocks={name: vars(time.get_clock_info(name)) for name in ("perf_counter", "process_time")},
    )
    execute(args.output, api, runtime)
    print(json.dumps(verify(args.output), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
