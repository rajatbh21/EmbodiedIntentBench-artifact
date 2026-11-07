"""Public-only network feedback and explicitly versioned endpoint binding."""

from dataclasses import asdict, fields
from pathlib import Path

from experiments.lake_feasibility_inference import read
from experiments.lake_intent_contract import immutable
from experiments.lake_intent_runner import IntentRouter, bind_scenario
from experiments.lake_missions.assets import save, sha
from experiments.lake_tracking import SurfaceProjector, TrackingPipeline
from experiments.lake_tracking_evaluation import RunnerReportEvaluator
from experiments.mission_network_policies import NetworkFeedbackInput
from experiments.mission_network_study_v1.endpoint import evaluate_network
from experiments.mission_network_study_v1.runner import run_mission
from experiments.mission_routing_v2.feedback import snapshot
from experiments.v4_benchmark.routers import RouterInput
from experiments.v4_benchmark.scenario import digest


class NetworkAdapter(IntentRouter):
    def __init__(self, delegate, contract, tracking, tracker):
        super().__init__(delegate, contract)
        self.tracking, self.tracker = immutable(tracking), tracker
        self.context = None

    def set_public_context(self, link, prepared, last_attempt):
        self.context = (link, len(prepared.payload), prepared.response_bytes, last_attempt)

    def select(self, observation):
        if self.context is None:
            raise ValueError("Current causal network context required")
        link, payload, response, last = self.context
        self.calls += 1
        return self.delegate.select(
            NetworkFeedbackInput(
                **{f.name: getattr(observation, f.name) for f in fields(RouterInput)},
                intent_contract=self.public,
                public_contract_sha256=self.public_sha256,
                tracking=self.tracking,
                evidence_history=immutable(
                    snapshot(self.tracker, observation.state.current_time_s)
                ),
                public_link=link,
                current_payload_bytes=payload,
                response_bytes=response,
                last_attempt=last,
            )
        )


def run_tracking_mission(
    scenario,
    backend,
    policy,
    executors,
    contract,
    output,
    *,
    config,
    terrain,
    map_metadata,
    router_name,
    cost,
    trace,
    costs,
    study_plan_sha256,
):
    bound = bind_scenario(scenario, contract)
    projector = SurfaceProjector(terrain)
    tracker = TrackingPipeline(config, projector)
    tracker.provenance.update(
        map=map_metadata,
        anchor_height_m=projector.anchor_height_m,
        hfov_rad=projector.hfov,
        static_target_assumption=True,
    )
    public_tracking = dict(configuration=tracker.provenance, cost=asdict(cost))
    adapter = NetworkAdapter(policy, contract, public_tracking, tracker)
    scorer = RunnerReportEvaluator(
        bound.matching_iou_threshold, bound.targets.values(), config.min_hits
    )
    output.mkdir(parents=True, exist_ok=False)
    save(output / "intent.json", contract.document())
    save(output / "public_intent.json", contract.public_document())
    protocol = dict(
        schema_version="network_tracking_execution_v1",
        router=router_name,
        gt_aware=False,
        deployable_policy=True,
        primary_outcome="outcome.json",
        study_plan_sha256=study_plan_sha256,
        contract_sha256=contract.sha256,
        public_contract_sha256=contract.public_sha256,
        scenario_sha256=digest(bound.source),
        recording_identity=bound.recording_identity,
        tracking=tracker.provenance,
        public_tracking=public_tracking,
        tracking_cost=asdict(cost),
        network_costs=asdict(costs),
        network_trace=[asdict(s) for s in trace.segments],
        network_clock="mission-origin nanoseconds; physical timestamp minus recording epoch",
        report_scoring="consistent_report_support_v1",
        implementation_sha256={
            str(p.resolve()): sha(p) for p in Path(__file__).parent.glob("*.py")
        },
        caveat="Virtual network/server; frozen RGB; assumed energy; reused development",
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
        trace=trace,
        costs=costs,
    )
    if (
        result["scenario_sha256"] != protocol["scenario_sha256"]
        or result["backend"]["recording_identity"] != bound.recording_identity
    ):
        raise ValueError("Physical or execution binding mismatch")
    events = read(output / "execution/events.json")
    outcome = evaluate_network(contract, events, result)
    outcome.update(
        gt_aware=False,
        deployable_policy=True,
        public_contract_deliveries=adapter.calls,
        protocol_sha256=sha(output / "protocol.json"),
        events_sha256=sha(output / "execution/events.json"),
        execution_sha256=sha(output / "execution/results.json"),
    )
    save(output / "outcome.json", outcome)
    return outcome
