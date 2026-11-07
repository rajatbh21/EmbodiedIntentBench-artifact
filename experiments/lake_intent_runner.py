"""Bind one declared intent to policy input, physical replay and primary scoring.

The frozen runner remains an execution engine. Its historical success flag lives
under execution/; outcome.json is authoritative for the declared intent version.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

from experiments.lake_intent_contract import IntentContract, evaluate, immutable, validate_runtime
from experiments.lake_missions.assets import save, sha
from experiments.v4_benchmark.calibrated_router import bind_router
from experiments.v4_benchmark.mission_requirements import MissionRequirements
from experiments.v4_benchmark.routers import RouterInput
from experiments.v4_benchmark.runner import run_mission
from experiments.v4_benchmark.scenario import digest


@dataclass(frozen=True)
class IntentRouterInput(RouterInput):
    intent_contract: Mapping[str, Any] | None = None
    public_contract_sha256: str = ""


class IntentRouter:
    """Forward only the immutable public intent; never retain the private contract."""

    def __init__(self, delegate: Any, contract: IntentContract):
        self.delegate = delegate
        self.public = immutable(contract.public_document())
        self.public_sha256 = contract.public_sha256
        self.calls = 0

    @property
    def last_decision(self) -> Any:
        return getattr(self.delegate, "last_decision", None)

    def select(self, observation: RouterInput) -> str:
        self.calls += 1
        return self.delegate.select(
            IntentRouterInput(
                **{field.name: getattr(observation, field.name) for field in fields(RouterInput)},
                intent_contract=self.public,
                public_contract_sha256=self.public_sha256,
            )
        )


def bind_scenario(scenario: Any, contract: IntentContract) -> Any:
    data = contract.document()
    validate_runtime(contract, asdict(scenario.contract))
    if (
        data["duration_s"] != scenario.duration_s
        or data["mission"]["initial_battery_frac"] != scenario.initial_battery_frac
        or set(data["evaluator_target_ids"]) != set(scenario.targets.values())
    ):
        raise ValueError("Intent and scenario duration, battery or target scope differ")
    reports = data["reporting"]
    requirements = MissionRequirements()
    if reports["mode"] == "report_required":
        requirements = MissionRequirements(
            require_detection=True,
            min_report_recall=reports["min_report_recall"],
            max_false_reports=reports["max_false_reports"],
            max_duplicate_reports=reports["max_duplicate_reports"],
            confirmed_report_deadline_s=reports["deadline_s"],
        )
    source = copy.deepcopy(scenario.source)
    source["mission_requirements"] = requirements.to_dict()
    bound = replace(scenario, source=source, requirements=requirements)
    if bound.recording_identity != scenario.recording_identity:
        raise ValueError("Intent binding changed the fixed physical flight")
    return bound


def run_intent_mission(
    scenario: Any,
    backend: Any,
    router: Any,
    executors: dict,
    contract: IntentContract,
    output: Path,
    *,
    router_name: str,
    gt_aware: bool,
) -> dict:
    """Write the declared contract before any action; caller owns backend cleanup."""
    import json

    if type(gt_aware) is not bool:
        raise ValueError("Explicit GT provenance required")
    bound = bind_scenario(scenario, contract)
    # Preserve train/test isolation validation before wrapping calibrated routers.
    bind_router(router, bound)
    adapter = IntentRouter(router, contract)
    output.mkdir(parents=True, exist_ok=False)
    save(output / "intent.json", contract.document())
    save(output / "public_intent.json", contract.public_document())
    protocol = dict(
        schema_version="lake_intent_execution_v1",
        contract_sha256=contract.sha256,
        public_contract_sha256=contract.public_sha256,
        scenario_sha256=digest(bound.source),
        recording_identity=bound.recording_identity,
        gt_aware=gt_aware,
        deployable_policy=not gt_aware,
        router=router_name,
        primary_outcome="outcome.json",
        historical_engine_diagnostics="execution/results.json",
        reporting=bound.reporting.to_dict(),
        resources=bound.resources.to_dict(),
        implementation_sha256={
            name: sha(Path(__file__).with_name(name))
            for name in ("lake_intent_contract.py", "lake_intent_runner.py")
        },
    )
    save(output / "protocol.json", protocol)
    result = run_mission(
        bound,
        backend,
        adapter,
        executors,
        router_name=router_name,
        output=output / "execution",
    )
    if (
        result["scenario_sha256"] != protocol["scenario_sha256"]
        or result["backend"]["recording_identity"] != protocol["recording_identity"]
        or result["reporting"] != protocol["reporting"]
        or result["resources"] != protocol["resources"]
        or result["mission_requirements"] != bound.requirements.to_dict()
    ):
        raise ValueError("Execution does not match declared protocol")
    events = json.loads((output / "execution/events.json").read_text())
    outcome = evaluate(contract, events, result)
    outcome.update(
        gt_aware=gt_aware,
        deployable_policy=not gt_aware,
        router=router_name,
        public_contract_deliveries=adapter.calls,
        protocol_sha256=sha(output / "protocol.json"),
        events_sha256=sha(output / "execution/events.json"),
        execution_sha256=sha(output / "execution/results.json"),
    )
    save(output / "outcome.json", outcome)
    return outcome
