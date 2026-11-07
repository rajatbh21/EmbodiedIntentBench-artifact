"""Versioned binding of prediction history into the unchanged paid tracking runner."""

from dataclasses import asdict
from pathlib import Path

from experiments.lake_feasibility_inference import read
from experiments.lake_intent_contract import evaluate
from experiments.lake_intent_runner import bind_scenario
from experiments.lake_missions.assets import save, sha
from experiments.lake_tracking import SurfaceProjector, TrackingPipeline
from experiments.lake_tracking_evaluation import RunnerReportEvaluator
from experiments.lake_tracking_runtime import DEFAULT_TRACKING_COST
from experiments.mission_frustum_routing.feedback import FeedbackAdapter
from experiments.v4_benchmark.calibrated_router import bind_router
from experiments.v4_benchmark.scenario import digest


def run_tracking_mission(
    scenario,
    backend,
    router,
    executors,
    contract,
    output,
    *,
    config,
    terrain,
    map_metadata,
    router_name,
    gt_aware,
    cost=DEFAULT_TRACKING_COST,
):
    from experiments.lake_tracking_runner import FROZEN_RUNNER_SHA256, run_mission

    if type(gt_aware) is not bool:
        raise ValueError("Explicit GT provenance required")
    original = Path(__file__).parents[1] / "v4_benchmark/runner.py"
    if sha(original) != FROZEN_RUNNER_SHA256:
        raise ValueError("Frozen runner changed; review the versioned tracking fork")
    bound = bind_scenario(scenario, contract)
    bind_router(router, bound)
    projector = SurfaceProjector(terrain if config.association == "terrain_world" else None)
    if config.association == "terrain_world" and terrain is None:
        raise ValueError("Terrain association requires a declared map")
    tracker = TrackingPipeline(config, projector)
    tracker.provenance.update(
        map=map_metadata if config.association == "terrain_world" else None,
        anchor_height_m=projector.anchor_height_m,
        hfov_rad=projector.hfov,
        static_target_assumption=True,
    )
    public_tracking = dict(configuration=tracker.provenance, cost=asdict(cost))
    adapter = FeedbackAdapter(
        router, contract, public_tracking, tracker, bound.world["camera_pitch_rad"]
    )
    scorer = RunnerReportEvaluator(
        bound.matching_iou_threshold, bound.targets.values(), config.min_hits
    )
    output.mkdir(parents=True, exist_ok=False)
    save(output / "intent.json", contract.document())
    save(output / "public_intent.json", contract.public_document())
    protocol = dict(
        schema_version="frustum_feedback_execution_v1",
        feedback_implementation_sha256={p.name: sha(p) for p in Path(__file__).parent.glob("*.py")},
        router=router_name,
        gt_aware=gt_aware,
        deployable_policy=not gt_aware,
        primary_outcome="outcome.json",
        contract_sha256=contract.sha256,
        public_contract_sha256=contract.public_sha256,
        scenario_sha256=digest(bound.source),
        recording_identity=bound.recording_identity,
        tracking=tracker.provenance,
        public_tracking=public_tracking,
        tracking_cost=asdict(cost),
        report_scoring="consistent_report_support_v1",
        frozen_runner_sha256=FROZEN_RUNNER_SHA256,
        implementation_sha256={
            name: sha(Path(__file__).parents[1] / name)
            for name in (
                "lake_tracking.py",
                "lake_tracking_evaluation.py",
                "lake_tracking_runner.py",
                "lake_tracking_runtime.py",
                "lake_intent_contract.py",
                "lake_intent_runner.py",
            )
        },
        caveat="Configured tracking costs; perfect visual DEM prior; no measured deployment claim",
    )
    save(output / "protocol.json", protocol)
    result = run_mission(
        bound,
        backend,
        adapter,
        executors,
        router_name=router_name,
        output=output / "execution",
        tracker=tracker,
        report_scoring=scorer,
        tracking_cost=cost,
    )
    if (
        result["scenario_sha256"] != protocol["scenario_sha256"]
        or result["backend"]["recording_identity"] != bound.recording_identity
        or result["tracking"] != protocol["tracking"]
        or result["tracking_cost"] != protocol["tracking_cost"]
    ):
        raise ValueError("Execution binding mismatch")
    events = read(output / "execution/events.json")
    outcome = evaluate(contract, events, result)
    outcome.update(
        schema_version="tracked_intent_result_v1",
        intent_schema_version=outcome["schema_version"],
        report_scoring="consistent_report_support_v1",
        gt_aware=gt_aware,
        deployable_policy=not gt_aware,
        public_contract_deliveries=adapter.calls,
        protocol_sha256=sha(output / "protocol.json"),
        events_sha256=sha(output / "execution/events.json"),
        execution_sha256=sha(output / "execution/results.json"),
    )
    save(output / "outcome.json", outcome)
    return outcome
