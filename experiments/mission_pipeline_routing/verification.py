"""Checkpoint-bound selected rule factories behind a runtime-owned evidence guard.

A rule only accepts or defers a candidate. It cannot emit reports, replace the
canonical evaluator, change history, or waive public evidence/deadline requirements.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from experiments.mission_pipeline_routing.catalog import Algorithm, Capability, Task


@dataclass(frozen=True)
class RuleVerdict:
    accept: bool
    reason: str
    score: float | None = None

    def __post_init__(self):
        if type(self.accept) is not bool or not isinstance(self.reason, str) or not self.reason:
            raise ValueError("Boolean rule verdict and nonempty reason required")
        if self.score is not None and (
            type(self.score) not in (int, float)
            or not math.isfinite(self.score)
            or not 0 <= self.score <= 1
        ):
            raise ValueError("Optional finite probability score required")


class RepeatedEvidenceRule:
    def __init__(self, hits, span_ns):
        self.hits, self.span_ns = hits, span_ns

    def decide(self, features):
        accepted = features["hits"] >= self.hits and features["span_ns"] >= self.span_ns
        return RuleVerdict(accepted, "selected_repeated_evidence_rule")


@dataclass(frozen=True)
class VerificationFactory:
    descriptor: Algorithm
    create: Callable
    artifacts: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        d = self.descriptor
        if (
            d.task is not Task.REPORT_VERIFICATION
            or d.requires != frozenset({Capability.TRACK_EVIDENCE})
            or d.provides
            or not callable(self.create)
        ):
            raise ValueError(
                "Verification factory requires the exact typed evidence-only interface"
            )
        if type(self.artifacts) is not tuple:
            raise ValueError("Immutable checkpoint inventory required")
        for row in self.artifacts:
            if (
                not isinstance(row, tuple)
                or len(row) != 2
                or not isinstance(row[0], str)
                or not Path(row[0]).is_absolute()
                or not isinstance(row[1], str)
                or len(row[1]) != 64
                or any(c not in "0123456789abcdef" for c in row[1])
            ):
                raise ValueError("Absolute checkpoint paths and lowercase SHA256 required")
        if len({path for path, _ in self.artifacts}) != len(self.artifacts):
            raise ValueError("Duplicate checkpoint artifact")
        if d.kind == "learned_evidence_model" and (
            not self.artifacts or d.checkpoint_sha256 not in {h for _, h in self.artifacts}
        ):
            raise ValueError("Learned verifier must bind its catalog checkpoint digest")
        if d.kind == "learned_cached_prediction":
            raise ValueError("Cached perception is not an evidence verifier")

    def instantiate(self, selected_descriptor):
        if selected_descriptor != self.descriptor:
            raise ValueError(
                "Selected verifier identity, capabilities or costs differ from factory"
            )
        for path, expected in self.artifacts:
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise ValueError("Selected verifier checkpoint changed")
        rule = self.create()
        if not callable(getattr(rule, "decide", None)):
            raise ValueError("Selected verifier must implement decide(features)")
        return rule


def default_factories(catalog):
    output = {}
    for name, hits, span in (
        ("VERIFY_REPEAT2", 2, 2_000_000_000),
        ("VERIFY_REPEAT3", 3, 4_000_000_000),
    ):
        descriptor = catalog.resolve(name, Task.REPORT_VERIFICATION, {Capability.TRACK_EVIDENCE})
        output[name] = VerificationFactory(
            descriptor, lambda h=hits, s=span: RepeatedEvidenceRule(h, s)
        )
    return output
