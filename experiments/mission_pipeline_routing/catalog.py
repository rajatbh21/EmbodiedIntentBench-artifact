"""Typed task compatibility and explicitly assumed downstream resource costs."""

import math
from dataclasses import asdict, dataclass
from enum import StrEnum


class Task(StrEnum):
    SEGMENTATION = "segmentation"
    ASSOCIATION = "association"
    REPORT_VERIFICATION = "report_verification"


class Capability(StrEnum):
    COMPLETED_MASK = "completed_selected_boolean_mask"
    PUBLIC_POSE = "public_camera_pose"
    PUBLIC_TERRAIN = "public_terrain_prior"
    TRACK_EVIDENCE = "completed_selected_track_evidence"


@dataclass(frozen=True)
class Costs:
    latency_ns: int
    energy_j: float
    memory_mb: float
    load_latency_ns: int
    load_energy_j: float
    provenance: str = "authored_assumption_not_host_or_electrical_measurement"

    def __post_init__(self):
        if any(type(v) is not int or v < 0 for v in (self.latency_ns, self.load_latency_ns)):
            raise ValueError("Nonnegative integer nanosecond costs required")
        if any(
            type(v) not in (int, float) or not math.isfinite(v) or v < 0
            for v in (self.energy_j, self.memory_mb, self.load_energy_j)
        ):
            raise ValueError("Finite nonnegative energy/memory costs required")
        if not self.provenance:
            raise ValueError("Cost provenance required")


@dataclass(frozen=True)
class Algorithm:
    algorithm_id: str
    task: Task
    implementation: str
    requires: frozenset[Capability]
    provides: frozenset[Capability]
    costs: Costs
    kind: str = "classical_algorithm"
    checkpoint_sha256: str | None = None

    def __post_init__(self):
        if not isinstance(self.task, Task) or not self.algorithm_id or not self.implementation:
            raise ValueError("Typed task and implementation identity required")
        if self.kind not in (
            "classical_algorithm",
            "learned_cached_prediction",
            "learned_evidence_model",
        ):
            raise ValueError("Declare classical versus learned implementation")
        if self.kind == "learned_evidence_model" and (
            not isinstance(self.checkpoint_sha256, str)
            or len(self.checkpoint_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.checkpoint_sha256)
        ):
            raise ValueError("Learned evidence model requires a checkpoint SHA256")
        if self.kind != "learned_evidence_model" and self.checkpoint_sha256 is not None:
            raise ValueError("Checkpoint field is reserved for learned evidence models")
        if not isinstance(self.requires, frozenset) or not isinstance(self.provides, frozenset):
            raise ValueError("Immutable capability sets required")
        if any(not isinstance(c, Capability) for c in self.requires | self.provides):
            raise ValueError("Unknown capability")

    def to_dict(self):
        return dict(
            algorithm_id=self.algorithm_id,
            task=self.task.value,
            implementation=self.implementation,
            kind=self.kind,
            checkpoint_sha256=self.checkpoint_sha256,
            requires=sorted(c.value for c in self.requires),
            provides=sorted(c.value for c in self.provides),
            costs=asdict(self.costs),
        )


class Catalog:
    def __init__(self, algorithms):
        rows = tuple(algorithms)
        if not rows or len({a.algorithm_id for a in rows}) != len(rows):
            raise ValueError("Nonempty unique algorithm catalog required")
        self._algorithms = {a.algorithm_id: a for a in rows}

    @property
    def algorithms(self):
        return tuple(self._algorithms.values())

    def resolve(self, algorithm_id, task, capabilities):
        if not isinstance(task, Task):
            raise ValueError("Typed requested task required")
        if algorithm_id not in self._algorithms:
            raise ValueError("Unknown algorithm")
        algorithm = self._algorithms[algorithm_id]
        if algorithm.task is not task:
            raise ValueError("Algorithm cannot execute requested task")
        if not algorithm.requires <= frozenset(capabilities):
            raise ValueError("Algorithm requires unavailable capabilities")
        return algorithm

    def document(self):
        return [a.to_dict() for a in self._algorithms.values()]


def default_catalog():
    mask = frozenset({Capability.COMPLETED_MASK})
    pose = mask | {Capability.PUBLIC_POSE}
    tracks = frozenset({Capability.TRACK_EVIDENCE})
    return Catalog(
        (
            Algorithm(
                "ASSOC_IMAGE_IOU",
                Task.ASSOCIATION,
                "greedy_image_box_iou",
                mask,
                tracks,
                Costs(5_000_000, 0.075, 8, 20_000_000, 0.3),
            ),
            Algorithm(
                "ASSOC_FLAT_WORLD",
                Task.ASSOCIATION,
                "greedy_flat_world_distance",
                pose,
                tracks,
                Costs(10_000_000, 0.15, 16, 30_000_000, 0.45),
            ),
            Algorithm(
                "ASSOC_TERRAIN_WORLD",
                Task.ASSOCIATION,
                "greedy_terrain_world_distance",
                pose | {Capability.PUBLIC_TERRAIN},
                tracks,
                Costs(30_000_000, 0.45, 64, 50_000_000, 0.75),
            ),
            Algorithm(
                "VERIFY_REPEAT2",
                Task.REPORT_VERIFICATION,
                "repeat2_span2_with_intent_floor",
                tracks,
                frozenset(),
                Costs(1_000_000, 0.015, 4, 5_000_000, 0.075),
            ),
            Algorithm(
                "VERIFY_REPEAT3",
                Task.REPORT_VERIFICATION,
                "repeat3_span4_with_intent_floor",
                tracks,
                frozenset(),
                Costs(2_000_000, 0.03, 4, 5_000_000, 0.075),
            ),
        )
    )
