"""Causal association and report-verification slots; navigation is never exposed.

One implementation is resident in each slot. Switching immediately discards that
slot's temporal state and pays its declared load cost. A pending job exposes no
candidate/report; cancellation keeps the paid residency/reset but admits no mask
into history. The selected algorithms execute only at paid completion. All
service/load/routing charges are authored assumptions, paid in full on dispatch.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from types import MappingProxyType

import numpy as np

from experiments.lake_tracking import (
    StaticReportTracker,
    SurfaceProjector,
    TrackingConfig,
    detections,
)
from experiments.mission_pipeline_routing.catalog import Capability, Task, default_catalog
from experiments.mission_pipeline_routing.verification import (
    RepeatedEvidenceRule,
    RuleVerdict,
    VerificationFactory,
    default_factories,
)


@dataclass(frozen=True)
class PublicPose:
    position_m: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    camera_pitch_rad: float

    def __post_init__(self):
        if not isinstance(self.position_m, tuple) or len(self.position_m) != 3:
            raise ValueError("Immutable 3D public position required")
        if not isinstance(self.quaternion_xyzw, tuple) or len(self.quaternion_xyzw) != 4:
            raise ValueError("Immutable public orientation required")
        if not all(
            math.isfinite(v)
            for v in (*self.position_m, *self.quaternion_xyzw, self.camera_pitch_rad)
        ):
            raise ValueError("Finite public pose required")
        if abs(sum(v * v for v in self.quaternion_xyzw) - 1) > 1e-5:
            raise ValueError("Unit quaternion required")

    def document(self):
        return dict(position_m=list(self.position_m), quaternion_xyzw=list(self.quaternion_xyzw))


@dataclass(frozen=True)
class CompletedPrediction:
    source_config_id: str
    capture_ns: int
    available_ns: int
    mask: np.ndarray
    pose: PublicPose | None

    def __post_init__(self):
        if (
            not self.source_config_id
            or type(self.capture_ns) is not int
            or type(self.available_ns) is not int
        ):
            raise ValueError("Source identity and integer timestamps required")
        if self.capture_ns < 0 or self.available_ns < self.capture_ns:
            raise ValueError("Prediction cannot precede capture")
        if (
            not isinstance(self.mask, np.ndarray)
            or self.mask.dtype != np.bool_
            or self.mask.ndim != 2
        ):
            raise ValueError("Completed selected boolean mask required")
        if min(self.mask.shape) <= 0:
            raise ValueError("Nonempty image dimensions required")
        # Bytes backing prevents a caller from restoring writeability or changing history.
        object.__setattr__(
            self,
            "mask",
            np.frombuffer(self.mask.tobytes(), dtype=np.bool_).reshape(self.mask.shape),
        )
        if self.pose is not None and not isinstance(self.pose, PublicPose):
            raise ValueError("Typed public pose required")


@dataclass(frozen=True)
class PublicContext:
    now_ns: int
    battery_frac: float
    deadline_ns: int
    phase_id: str
    phase_start_ns: int
    phase_end_ns: int
    phase_deadline_ns: int
    required_hits: int
    required_span_ns: int
    memory_capacity_mb: float
    upstream_resident_mb: float

    def __post_init__(self):
        ns = (
            self.now_ns,
            self.deadline_ns,
            self.phase_start_ns,
            self.phase_end_ns,
            self.phase_deadline_ns,
            self.required_span_ns,
        )
        if any(type(v) is not int or v < 0 for v in ns):
            raise ValueError("Nonnegative integer public times required")
        if self.phase_start_ns >= self.phase_end_ns or self.phase_deadline_ns < self.phase_start_ns:
            raise ValueError("Valid public phase interval required")
        if type(self.required_hits) is not int or self.required_hits < 2:
            raise ValueError("Repeated evidence floor of at least two required")
        if not self.phase_id or not 0 <= self.battery_frac <= 1:
            raise ValueError("Phase identity and valid battery fraction required")
        if any(
            type(v) not in (int, float) or not math.isfinite(v) or v < 0
            for v in (self.memory_capacity_mb, self.upstream_resident_mb)
        ):
            raise ValueError("Finite memory budget required")


@dataclass(frozen=True)
class Selection:
    association_id: str
    verification_id: str


@dataclass(frozen=True)
class EvidenceRef:
    capture_ns: int
    component_id: int
    source_config_id: str


@dataclass(frozen=True)
class ReportCandidate:
    track_id: str
    component_id: int
    current_capture_ns: int
    evidence: tuple[EvidenceRef, ...]
    position_m: tuple[float, float] | None

    def features(self, context):
        evidence = tuple(
            e
            for e in self.evidence
            if context.phase_start_ns <= e.capture_ns < context.phase_end_ns
        )
        times = [e.capture_ns for e in evidence]
        return dict(
            track_id=self.track_id,
            current_component_id=self.component_id,
            current_capture_ns=self.current_capture_ns,
            phase_id=context.phase_id,
            hits=len(times),
            span_ns=times[-1] - times[0] if times else 0,
            last_gap_ns=times[-1] - times[-2] if len(times) > 1 else None,
            source_config_ids=sorted({e.source_config_id for e in evidence}),
            predicted_position_m=list(self.position_m) if self.position_m is not None else None,
            required_hits=context.required_hits,
            required_span_ns=context.required_span_ns,
            phase_deadline_ns=context.phase_deadline_ns,
            evidence=[asdict(e) for e in evidence],
            ground_truth_access=False,
        )


class AssociationExecutor:
    def __init__(self, algorithm_id, generation, terrain):
        modes = {
            "ASSOC_IMAGE_IOU": "image_iou",
            "ASSOC_FLAT_WORLD": "flat_world",
            "ASSOC_TERRAIN_WORLD": "terrain_world",
        }
        if algorithm_id not in modes:
            raise ValueError("Unimplemented association algorithm")
        self.mode = modes[algorithm_id]
        if self.mode == "terrain_world" and terrain is None:
            raise ValueError("Terrain association requires an explicit public terrain prior")
        self.generation = generation
        # Association owns no reporting policy; its legacy emitter is disabled.
        self.state = StaticReportTracker(TrackingConfig(association=self.mode, min_hits=2**31))
        self.projectors = (
            {}
            if self.mode == "image_iou"
            else {self.mode: SurfaceProjector(terrain if self.mode == "terrain_world" else None)}
        )
        self.sources = {}
        self.calls = 0

    def update(self, prediction, completion_ns):
        if self.projectors and prediction.pose is None:
            raise ValueError("World association requires public pose")
        measurements = detections(
            prediction.mask,
            prediction.pose.document() if prediction.pose else None,
            prediction.pose.camera_pitch_rad if prediction.pose else None,
            self.projectors,
        )
        emitted = self.state.update(measurements, prediction.capture_ns / 1e9, completion_ns / 1e9)
        if emitted:
            raise ValueError("Association must not bypass the verification task")
        self.sources[prediction.capture_ns] = prediction.source_config_id
        self.calls += 1
        output = []
        for assignment in self.state.assignments:
            track = self.state.tracks[assignment["track_id"]]
            refs = tuple(
                EvidenceRef(
                    round(e["capture_s"] * 1e9),
                    e["component_id"],
                    self.sources[round(e["capture_s"] * 1e9)],
                )
                for e in track["evidence"]
            )
            output.append(
                ReportCandidate(
                    f"{self.generation}:{assignment['track_id']}",
                    assignment["component_id"],
                    prediction.capture_ns,
                    refs,
                    tuple(track["position"]) if track["position"] is not None else None,
                )
            )
        return tuple(output)


def _freeze_features(value):
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze_features(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_features(v) for v in value)
    return value


class VerificationExecutor:
    def __init__(self, algorithm_id, rule=None):
        rules = {"VERIFY_REPEAT2": (2, 2_000_000_000), "VERIFY_REPEAT3": (3, 4_000_000_000)}
        if rule is None:
            if algorithm_id not in rules:
                raise ValueError("Unimplemented report verifier")
            rule = RepeatedEvidenceRule(*rules[algorithm_id])
        if not callable(getattr(rule, "decide", None)):
            raise ValueError("Report verifier rule interface required")
        self.rule = rule
        self.rule_calls = 0
        self.reported_tracks = set()
        self.reported_positions = []
        self.calls = 0

    def evaluate(self, candidates, context, completion_ns):
        decisions = []
        emitted = []
        self.calls += 1
        for candidate in candidates:
            features = candidate.features(context)
            reason = "insufficient_selected_evidence"
            decision = "defer"
            support = features["evidence"]
            required_hits = context.required_hits
            required_span = context.required_span_ns
            verdict = None
            if candidate.track_id in self.reported_tracks or (
                candidate.position_m is not None
                and any(math.dist(candidate.position_m, p) <= 1.0 for p in self.reported_positions)
            ):
                decision, reason = "suppress", "already_reported_prediction_track_or_world_position"
            elif completion_ns > min(context.deadline_ns, context.phase_deadline_ns):
                reason = "completion_after_public_deadline"
            elif (
                features["hits"] >= required_hits
                and features["span_ns"] >= required_span
                and support
                and support[-1]["capture_ns"] == candidate.current_capture_ns
                and len({e["capture_ns"] for e in support}) == len(support)
            ):
                # A selected rule may only narrow this runtime-owned admissible set.
                verdict = self.rule.decide(_freeze_features(features))
                if not isinstance(verdict, RuleVerdict):
                    raise ValueError("Selected verifier returned an invalid typed verdict")
                self.rule_calls += 1
                if verdict.accept:
                    decision, reason = "emit", "selected_verifier_and_public_intent_floor_satisfied"
                    self.reported_tracks.add(candidate.track_id)
                    if candidate.position_m is not None:
                        self.reported_positions.append(candidate.position_m)
                    emitted.append((candidate, support))
                else:
                    reason = "selected_verifier_deferred"
            decisions.append(
                dict(
                    candidate=features,
                    decision=decision,
                    reason=reason,
                    effective_required_hits=required_hits,
                    effective_required_span_ns=required_span,
                    selected_rule=asdict(verdict) if verdict is not None else None,
                )
            )
        return emitted, decisions


@dataclass(frozen=True)
class Pending:
    job_id: int
    start_ns: int
    ready_ns: int
    charged_energy_j: float
    routing_latency_ns: int
    routing_energy_j: float
    load_latency_ns: int
    load_energy_j: float
    service_latency_ns: int
    service_energy_j: float
    resident_memory_mb: float
    reset_slots: tuple[str, ...]
    selection: Selection

    def to_dict(self):
        return asdict(self)


class Pipeline:
    def __init__(
        self,
        *,
        terrain=None,
        catalog=None,
        routing_latency_ns=50_000_000,
        routing_energy_j=0.75,
        verification_factories=None,
    ):
        if type(routing_latency_ns) is not int or routing_latency_ns < 0:
            raise ValueError("Routing latency must be integer nanoseconds")
        if (
            type(routing_energy_j) not in (int, float)
            or not math.isfinite(routing_energy_j)
            or routing_energy_j < 0
        ):
            raise ValueError("Routing energy must be finite nonnegative")
        self.catalog = catalog or default_catalog()
        self.verification_factories = default_factories(self.catalog)
        if verification_factories is not None:
            for key, factory in verification_factories.items():
                if (
                    not isinstance(factory, VerificationFactory)
                    or key != factory.descriptor.algorithm_id
                ):
                    raise ValueError("Factory registry key/typed identity mismatch")
                if key in self.verification_factories:
                    raise ValueError("Default verifier identities cannot be silently replaced")
                self.verification_factories[key] = factory
        self.terrain = terrain
        self.routing_latency_ns, self.routing_energy_j = routing_latency_ns, routing_energy_j
        self._association = self._verification = None
        self._association_id = self._verification_id = None
        self._generation = self._next_job = self._report_count = 0
        self._pending = None
        self._last_capture_ns = self._last_completion_ns = self._last_busy_end_ns = -1
        self._clock_ns = 0
        self.cumulative_energy_j = 0.0
        self.completed_jobs = self.canceled_jobs = self.failed_jobs = 0
        self.executed_calls = {}

    def dispatch(self, selection, prediction, context):
        if (
            not isinstance(selection, Selection)
            or not isinstance(prediction, CompletedPrediction)
            or not isinstance(context, PublicContext)
        ):
            raise ValueError("Typed selection, completed prediction and public context required")
        if self._pending is not None:
            raise ValueError("Serial pipeline already has unpaid completion")
        now = context.now_ns
        if (
            now < self._clock_ns
            or now < prediction.available_ns
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
        memory = (
            context.upstream_resident_mb
            + association.costs.memory_mb
            + verification.costs.memory_mb
        )
        if memory > context.memory_capacity_mb:
            raise ValueError("Selected pipeline exceeds resident memory budget")
        changed_a = selection.association_id != self._association_id
        changed_v = selection.verification_id != self._verification_id
        loads = [
            a.costs
            for a, changed in ((association, changed_a), (verification, changed_v))
            if changed
        ]
        load_ns = sum(c.load_latency_ns for c in loads)
        load_j = sum(c.load_energy_j for c in loads)
        service_ns = association.costs.latency_ns + verification.costs.latency_ns
        service_j = association.costs.energy_j + verification.costs.energy_j
        # Validate constructors without changing residency, counters, history or payment.
        next_association = (
            AssociationExecutor(selection.association_id, self._generation + 1, self.terrain)
            if changed_a
            else self._association
        )
        if changed_v:
            factory = self.verification_factories.get(selection.verification_id)
            if factory is None:
                raise ValueError("Unimplemented report verifier")
            next_verification = VerificationExecutor(
                selection.verification_id, factory.instantiate(verification)
            )
        else:
            next_verification = self._verification
        pending = Pending(
            self._next_job + 1,
            now,
            now + self.routing_latency_ns + load_ns + service_ns,
            self.routing_energy_j + load_j + service_j,
            self.routing_latency_ns,
            self.routing_energy_j,
            load_ns,
            load_j,
            service_ns,
            service_j,
            memory,
            tuple(
                name
                for name, changed in (
                    ("association", changed_a),
                    ("report_verification", changed_v),
                )
                if changed
            ),
            selection,
        )
        if changed_a:
            self._generation += 1
            self._association = next_association
            self._association_id = selection.association_id
        if changed_v:
            self._verification = next_verification
            self._verification_id = selection.verification_id
        self._next_job += 1
        self._pending = (pending, prediction, context)
        self._last_capture_ns = prediction.capture_ns
        self._clock_ns = now
        self.cumulative_energy_j += pending.charged_energy_j
        return pending

    def complete(self, job_id, now_ns):
        if self._pending is None or job_id != self._pending[0].job_id:
            raise ValueError("No such pending selected job")
        pending, prediction, context = self._pending
        if type(now_ns) is not int or now_ns < pending.ready_ns:
            raise ValueError("Outputs are unavailable before paid completion")
        # Both selected stages commit together; a failing injected rule cannot expose
        # a partially updated track/report ledger or receive a free same-job retry.
        association = copy.copy(self._association)
        association.state = copy.deepcopy(self._association.state)
        association.sources = dict(self._association.sources)
        # Read-only map/projector priors are shared; temporal state is isolated.
        verification = copy.deepcopy(self._verification)
        try:
            candidates = association.update(prediction, now_ns)
            accepted, decisions = verification.evaluate(candidates, context, now_ns)
        except Exception:
            self._pending = None
            self._clock_ns = self._last_busy_end_ns = now_ns
            self.failed_jobs += 1
            raise
        self._association, self._verification = association, verification
        reports = []
        for candidate, support in accepted:
            self._report_count += 1
            reports.append(
                dict(
                    report_id=self._report_count,
                    component_id=candidate.component_id,
                    first_prediction_s=support[0]["capture_ns"] / 1e9,
                    completion_time_s=now_ns / 1e9,
                    confirmation_latency_s=(now_ns - support[0]["capture_ns"]) / 1e9,
                    evidence=[
                        dict(capture_s=e["capture_ns"] / 1e9, component_id=e["component_id"])
                        for e in support
                    ],
                )
            )
        for key in (pending.selection.association_id, pending.selection.verification_id):
            self.executed_calls[key] = self.executed_calls.get(key, 0) + 1
        self._clock_ns = self._last_completion_ns = self._last_busy_end_ns = now_ns
        self._pending = None
        self.completed_jobs += 1
        return dict(
            completed_ns=now_ns,
            selection=asdict(pending.selection),
            candidates=[c.features(context) for c in candidates],
            verification_decisions=decisions,
            emitted_reports=reports,
            cumulative_reports=self._report_count,
            selected_verifier_rule_calls=self._verification.rule_calls,
            costs=pending.to_dict(),
        )

    def cancel(self, job_id, now_ns):
        if self._pending is None or self._pending[0].job_id != job_id:
            raise ValueError("No such pending selected job")
        pending, _, _ = self._pending
        if type(now_ns) is not int or now_ns < self._clock_ns or now_ns >= pending.ready_ns:
            raise ValueError("Cancellation must precede ready time without reversing time")
        self._pending = None
        self._clock_ns = self._last_busy_end_ns = now_ns
        self.canceled_jobs += 1
        return dict(
            status="canceled",
            prediction_committed=False,
            emitted_reports=[],
            charged_energy_j=pending.charged_energy_j,
            reset_slots=pending.reset_slots,
        )

    def snapshot(self, now_ns):
        if type(now_ns) is not int or now_ns < self._clock_ns or now_ns < self._last_completion_ns:
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
                emitted_reports=self._report_count,
                tracks=tuple(tracks),
                ground_truth_access=False,
            )
        )
