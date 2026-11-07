"""Serial downstream execution with separately enforced invocation deadlines.

The association budget starts when the selected perception result is available.
Waiting, the shared downstream routing charge, association loading and service,
and physics-tick availability all consume that budget. Verification is released
only by an admitted association result and pays its own loading/service/rounding.

Missed work finishes and is paid in full, but its output is discarded. Association
and verification commit separately: late verification cannot undo timely tracks,
and late association cannot dispatch verification. Loading/reset persists on a
dispatched miss; an undispatched verifier neither loads nor pays. All costs are
the existing catalog's authored assumptions, not new hardware measurements.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict
from types import MappingProxyType

from experiments.mission_pipeline_routing.catalog import Capability, Task, default_catalog
from experiments.mission_pipeline_routing.runtime import (
    AssociationExecutor,
    CompletedPrediction,
    PublicContext,
    Selection,
    VerificationExecutor,
)
from experiments.mission_pipeline_routing.verification import (
    VerificationFactory,
    default_factories,
)
from experiments.mission_task_deadlines.contract import (
    DeadlineProfile,
    TaskDeadline,
    TaskKind,
    assess_completion,
)


class _CountedRule:
    """Count actual selected forwards, including failures, without weakening guards."""

    def __init__(self, rule):
        self.rule, self.calls = rule, 0

    def decide(self, features):
        self.calls += 1
        return self.rule.decide(features)


class DeadlinePipeline:
    """Advance one paid invocation to its full finish; never expose future history.

    ``execute`` is a synchronous discrete-event operation. The returned
    ``completed_ns`` is the next legal history time, even for discarded work.
    ``executed_calls`` counts actual executor invocations; ``stage_dispatches``
    also includes paid attempts discarded before executor evaluation. Learned
    forwards are counted separately, after the public intent/deadline guards.
    When capacity is supplied, the next stage cannot dispatch after paid work
    plus baseline flight has exhausted the public battery. Baseline flight is
    used for this guard but is still charged only by the caller's mission clock.
    """

    def __init__(
        self,
        *,
        profile: DeadlineProfile,
        terrain=None,
        catalog=None,
        verification_factories=None,
        step_ns=10_000_000,
        routing_latency_ns=50_000_000,
        routing_energy_j=0.75,
        battery_capacity_j=None,
        flight_power_w=0.0,
    ):
        if not isinstance(profile, DeadlineProfile):
            raise ValueError("A typed deadline profile is required")
        if type(step_ns) is not int or step_ns <= 0:
            raise ValueError("Physics step must be positive integer nanoseconds")
        if type(routing_latency_ns) is not int or routing_latency_ns < 0:
            raise ValueError("Routing latency must be nonnegative integer nanoseconds")
        if (
            type(routing_energy_j) not in (int, float)
            or not math.isfinite(routing_energy_j)
            or routing_energy_j < 0
        ):
            raise ValueError("Routing energy must be finite and nonnegative")
        if battery_capacity_j is not None and (
            type(battery_capacity_j) not in (int, float)
            or not math.isfinite(battery_capacity_j)
            or battery_capacity_j <= 0
        ):
            raise ValueError("Optional battery capacity must be finite and positive")
        if (
            type(flight_power_w) not in (int, float)
            or not math.isfinite(flight_power_w)
            or flight_power_w < 0
        ):
            raise ValueError("Flight power must be finite and nonnegative")
        self.profile, self.terrain, self.catalog = profile, terrain, catalog or default_catalog()
        self.step_ns = step_ns
        self.routing_latency_ns, self.routing_energy_j = routing_latency_ns, routing_energy_j
        self.battery_capacity_j, self.flight_power_w = battery_capacity_j, flight_power_w
        self.verification_factories = default_factories(self.catalog)
        for key, factory in (verification_factories or {}).items():
            if (
                not isinstance(factory, VerificationFactory)
                or key != factory.descriptor.algorithm_id
            ):
                raise ValueError("Factory registry key/typed identity mismatch")
            if key in self.verification_factories:
                raise ValueError("Default verifier identities cannot be silently replaced")
            self.verification_factories[key] = factory
        self._association = self._verification = None
        self._association_id = self._verification_id = None
        self._generation = self._next_job = self._report_count = 0
        self._last_capture_ns = self._last_completion_ns = self._last_busy_end_ns = -1
        self._clock_ns = 0
        self.cumulative_energy_j = 0.0
        self.completed_jobs = self.canceled_jobs = self.failed_jobs = self.late_jobs = 0
        self.battery_depleted_jobs = 0
        self.stage_dispatches = {}
        self.executed_calls = {}
        self.selected_rule_calls = {}

    @property
    def final_time_ns(self):
        return self._clock_ns

    @staticmethod
    def _increment(counter, key, amount=1):
        counter[key] = counter.get(key, 0) + amount

    def _validate(self, selection, prediction, context):
        if (
            not isinstance(selection, Selection)
            or not isinstance(prediction, CompletedPrediction)
            or not isinstance(context, PublicContext)
        ):
            raise ValueError("Typed selection, completed prediction and public context required")
        if context.battery_frac <= 0:
            raise ValueError("No association task may dispatch with a depleted battery")
        if (
            context.now_ns < self._clock_ns
            or context.now_ns < prediction.available_ns
            or prediction.capture_ns <= self._last_capture_ns
        ):
            raise ValueError("Future prediction, reused capture or reversed time")
        if prediction.capture_ns < self._last_busy_end_ns:
            raise ValueError("Capture occurred while prior serial pipeline was busy")
        if not context.phase_start_ns <= prediction.capture_ns < context.phase_end_ns:
            raise ValueError("Prediction belongs to another public phase")
        capabilities = {Capability.COMPLETED_MASK}
        if prediction.pose is not None:
            capabilities.add(Capability.PUBLIC_POSE)
        if self.terrain is not None:
            capabilities.add(Capability.PUBLIC_TERRAIN)
        association = self.catalog.resolve(selection.association_id, Task.ASSOCIATION, capabilities)
        verification = self.catalog.resolve(
            selection.verification_id, Task.REPORT_VERIFICATION, association.provides
        )
        previous_verifier_mb = (
            self.catalog.resolve(
                self._verification_id, Task.REPORT_VERIFICATION, {Capability.TRACK_EVIDENCE}
            ).costs.memory_mb
            if self._verification_id is not None
            else 0.0
        )
        # The previous verifier remains resident during association. Reserve
        # both sequential configurations' peak, even when the next verifier is
        # smaller; a missed association cannot silently unload the old slot.
        memory_mb = (
            context.upstream_resident_mb
            + association.costs.memory_mb
            + max(previous_verifier_mb, verification.costs.memory_mb)
        )
        if memory_mb > context.memory_capacity_mb:
            raise ValueError("Selected pipeline exceeds resident memory budget")
        changed_a = selection.association_id != self._association_id
        changed_v = selection.verification_id != self._verification_id
        # Constructors/checkpoint identities are checked before any mutation or
        # charge. Constructing a local candidate does not install/reset a slot.
        next_a = (
            AssociationExecutor(selection.association_id, self._generation + 1, self.terrain)
            if changed_a
            else self._association
        )
        if changed_v:
            factory = self.verification_factories.get(selection.verification_id)
            if factory is None:
                raise ValueError("Unimplemented report verifier")
            next_v = VerificationExecutor(
                selection.verification_id, factory.instantiate(verification)
            )
        else:
            next_v = self._verification
        return association, verification, memory_mb, changed_a, changed_v, next_a, next_v

    def _stage(self, task, descriptor, release_ns, start_ns, changed, memory_mb):
        routing_ns = self.routing_latency_ns if task is TaskKind.ASSOCIATION else 0
        routing_j = self.routing_energy_j if task is TaskKind.ASSOCIATION else 0.0
        load_ns = descriptor.costs.load_latency_ns if changed else 0
        load_j = descriptor.costs.load_energy_j if changed else 0.0
        raw_finish = start_ns + routing_ns + load_ns + descriptor.costs.latency_ns
        result = assess_completion(
            TaskDeadline(task, release_ns, self.profile.budget_for(task)), raw_finish, self.step_ns
        ).to_dict()
        result.update(
            algorithm_id=descriptor.algorithm_id,
            start_ns=start_ns,
            natural_finish_ns=raw_finish,
            dispatched=True,
            committed=False,
            reset_slot=changed,
            deadline_status=result["status"],
            costs=dict(
                routing_latency_ns=routing_ns,
                routing_energy_j=routing_j,
                load_latency_ns=load_ns,
                load_energy_j=load_j,
                service_latency_ns=descriptor.costs.latency_ns,
                service_energy_j=descriptor.costs.energy_j,
                charged_energy_j=routing_j + load_j + descriptor.costs.energy_j,
                queue_wait_ns=start_ns - release_ns,
                availability_rounding_ns=result["usable_finish_ns"] - raw_finish,
                resident_memory_mb=memory_mb,
                provenance=descriptor.costs.provenance,
            ),
        )
        return result

    def _pay(self, stage):
        self.cumulative_energy_j += stage["costs"]["charged_energy_j"]
        self._increment(self.stage_dispatches, stage["algorithm_id"])

    def execute(self, selection, prediction, context):
        a, v, memory, changed_a, changed_v, next_a, next_v = self._validate(
            selection, prediction, context
        )
        association = self._stage(
            TaskKind.ASSOCIATION, a, prediction.available_ns, context.now_ns, changed_a, memory
        )
        if changed_a:
            self._association, self._association_id = next_a, selection.association_id
            self._generation += 1
        self._next_job += 1
        self._last_capture_ns = prediction.capture_ns
        self._pay(association)
        remaining_after_association_j = (
            context.battery_frac * self.battery_capacity_j
            - association["costs"]["charged_energy_j"]
            - (association["usable_finish_ns"] - context.now_ns) / 1e9 * self.flight_power_w
            if self.battery_capacity_j is not None
            else None
        )
        battery_depleted = (
            remaining_after_association_j is not None and remaining_after_association_j <= 0
        )
        candidates, decisions, reports = (), [], []
        stages, actual_rule_calls = [association], 0
        if association["on_time"]:
            try:
                working_a = copy.copy(self._association)
                working_a.state = copy.deepcopy(self._association.state)
                working_a.sources = dict(self._association.sources)
                self._increment(self.executed_calls, selection.association_id)
                candidates = working_a.update(prediction, association["usable_finish_ns"])
            except Exception as exc:
                association.update(status="execution_failed", error_type=type(exc).__name__)
            else:
                self._association = working_a
                association["committed"] = True
        if association["committed"] and not battery_depleted:
            release = association["usable_finish_ns"]
            verification = self._stage(
                TaskKind.VERIFICATION, v, release, release, changed_v, memory
            )
            stages.append(verification)
            if changed_v:
                self._verification, self._verification_id = next_v, selection.verification_id
            self._pay(verification)
            if verification["on_time"]:
                counted_rule = None
                try:
                    working_v = copy.deepcopy(self._verification)
                    counted_rule = _CountedRule(working_v.rule)
                    working_v.rule = counted_rule
                    self._increment(self.executed_calls, selection.verification_id)
                    accepted, decisions = working_v.evaluate(
                        candidates, context, verification["usable_finish_ns"]
                    )
                except Exception as exc:
                    verification.update(status="execution_failed", error_type=type(exc).__name__)
                else:
                    working_v.rule = counted_rule.rule
                    self._verification = working_v
                    verification["committed"] = True
                    for candidate, support in accepted:
                        self._report_count += 1
                        reports.append(
                            dict(
                                report_id=self._report_count,
                                component_id=candidate.component_id,
                                first_prediction_s=support[0]["capture_ns"] / 1e9,
                                completion_time_s=verification["usable_finish_ns"] / 1e9,
                                confirmation_latency_s=(
                                    verification["usable_finish_ns"] - support[0]["capture_ns"]
                                )
                                / 1e9,
                                evidence=[
                                    dict(
                                        capture_s=e["capture_ns"] / 1e9,
                                        component_id=e["component_id"],
                                    )
                                    for e in support
                                ],
                            )
                        )
                actual_rule_calls = counted_rule.calls if counted_rule is not None else 0
                self._increment(
                    self.selected_rule_calls, selection.verification_id, actual_rule_calls
                )
        else:
            stages.append(
                dict(
                    task=TaskKind.VERIFICATION.value,
                    algorithm_id=selection.verification_id,
                    status=(
                        "not_dispatched_battery_depleted"
                        if battery_depleted
                        else "not_dispatched_predecessor_missing"
                    ),
                    release_ns=None,
                    natural_finish_ns=None,
                    usable_finish_ns=None,
                    deadline_ns=None,
                    dispatched=False,
                    committed=False,
                    reset_slot=False,
                    costs={"charged_energy_j": 0.0},
                )
            )
        paid_stages = [s for s in stages if s["dispatched"]]
        completed_ns = paid_stages[-1]["usable_finish_ns"]
        self._clock_ns = self._last_busy_end_ns = completed_ns
        if any(s["committed"] for s in paid_stages):
            self._last_completion_ns = max(
                s["usable_finish_ns"] for s in paid_stages if s["committed"]
            )
        if battery_depleted:
            self.battery_depleted_jobs += 1
            status = "battery_depleted"
        elif any(s["status"] == "execution_failed" for s in paid_stages):
            self.failed_jobs += 1
            status = "execution_failed"
        elif any(not s["on_time"] for s in paid_stages):
            self.late_jobs += 1
            status = "task_deadline_miss"
        else:
            self.completed_jobs += 1
            status = "completed"
        return dict(
            job_id=self._next_job,
            status=status,
            completed_ns=completed_ns,
            selection=asdict(selection),
            deadline_profile=self.profile.name,
            candidates=[candidate.features(context) for candidate in candidates],
            verification_decisions=decisions,
            emitted_reports=reports,
            cumulative_reports=self._report_count,
            stages=stages,
            selected_rulecalls=actual_rule_calls,
            selected_verifier_rule_calls=sum(self.selected_rule_calls.values()),
            battery_depleted_before_verification=battery_depleted,
            remaining_after_association_j=remaining_after_association_j,
            charged_energy_j=sum(s["costs"]["charged_energy_j"] for s in paid_stages),
            cumulative_energy_j=self.cumulative_energy_j,
        )

    def snapshot(self, now_ns):
        if type(now_ns) is not int or now_ns < self._clock_ns:
            raise ValueError("History requested before its availability")
        tracks = []
        if self._association is not None:
            for tid, state in sorted(self._association.state.tracks.items()):
                tracks.append(
                    MappingProxyType(
                        dict(
                            track_id=f"{self._generation}:{tid}",
                            capture_times_ns=tuple(
                                round(e["capture_s"] * 1e9) for e in state["evidence"]
                            ),
                        )
                    )
                )
        return MappingProxyType(
            dict(
                available_at_ns=now_ns,
                last_completion_ns=self._last_completion_ns,
                association_id=self._association_id,
                verification_id=self._verification_id,
                completed_jobs=self.completed_jobs,
                canceled_jobs=self.canceled_jobs,
                failed_jobs=self.failed_jobs,
                late_jobs=self.late_jobs,
                battery_depleted_jobs=self.battery_depleted_jobs,
                emitted_reports=self._report_count,
                tracks=tuple(tracks),
                ground_truth_access=False,
            )
        )
