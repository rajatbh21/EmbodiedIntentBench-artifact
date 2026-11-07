"""The executor registry, and the context its factories are built from.

Lets the composition root pick a backend by name -- ``--executor replay`` -- without the
runner importing any backend, and lets a new backend join by registering itself rather than
by editing a branch somewhere in the core.

Why a context object
--------------------
The registry previously held the executor *classes*, whose constructors take different
arguments: ``ProfileExecutor(profiles, ...)`` against ``make_replay_executor(path, ...)``.
There was no uniform ``create(name, ...)`` call, so nothing could dispatch through it and it
served only to print help text. Giving every factory the same signature --
``(ExecutorContext) -> Executor`` -- is what turns the registry into a real construction
path.

The context carries everything any backend might need. A factory that does not need a field
ignores it; a factory whose field is missing raises with an explanation rather than
substituting a different backend, because a run that silently changed executor mid-selection
would report a provenance it did not have.
"""

from __future__ import annotations

from dataclasses import dataclass

from missionbench.executor.base import Executor, PredictionSource
from missionbench.executor.profile_executor import DEFAULT_REMOTE_TIMEOUT_S, ProfileExecutor
from missionbench.executor.real_segmentation_executor import RealSegmentationExecutor
from missionbench.executor.replay_executor import ReplayExecutor, ReplayRecordSet
from missionbench.registry import Registry
from missionbench.schemas.loading import SchemaValidationError
from missionbench.schemas.profile import ProfileCatalog

__all__ = ["ExecutorContext", "build_executor", "executor_registry"]


@dataclass(frozen=True, slots=True)
class ExecutorContext:
    """Everything the composition root can offer a backend, assembled once per episode."""

    episode_id: str
    #: Per-configuration costs for this episode's platform. Used by ``profile``.
    profiles: ProfileCatalog | None = None
    #: Supplies prediction payloads for a simulating backend. Used by ``profile``.
    prediction_source: PredictionSource | None = None
    #: Precomputed outcomes for this episode. Used by ``replay``.
    replay_record_set: ReplayRecordSet | None = None
    #: Whether a missing replay entry raises instead of recording a failed inference.
    replay_strict: bool = False
    remote_timeout_s: float = DEFAULT_REMOTE_TIMEOUT_S


def _build_profile(context: ExecutorContext) -> Executor:
    if context.profiles is None:
        raise SchemaValidationError(
            f"episode {context.episode_id!r}: the 'profile' executor needs a configuration "
            "profile catalog for this episode's platform, and none was supplied"
        )
    return ProfileExecutor(
        context.profiles,
        predictions=context.prediction_source,
        remote_timeout_s=context.remote_timeout_s,
    )


def _build_replay(context: ExecutorContext) -> Executor:
    record_set = context.replay_record_set
    if record_set is None:
        raise SchemaValidationError(
            f"episode {context.episode_id!r}: the 'replay' executor needs a replay record "
            f"set declaring episode_id {context.episode_id!r}, and none was found under the "
            "data root's predictions/ directory. Record one with "
            "`python -m missionbench.tools.record_replay`, or run with --executor profile."
        )
    return ReplayExecutor(
        record_set.records,
        strict=context.replay_strict,
        record_set_id=record_set.record_set_id,
        source=record_set.source,
    )


def _build_real_segmentation(context: ExecutorContext) -> Executor:
    del context
    # Constructing it raises. Kept as a registry entry so selecting it reports "not
    # implemented" rather than "unknown executor", which would suggest a typo.
    return RealSegmentationExecutor()


executor_registry: Registry[Executor] = Registry("executor")

#: Costs synthesised from a per-platform profile. The V1 default: no GPU, no weights, and
#: exactly reproducible. Every number it reports is synthetic.
executor_registry.register("profile", _build_profile)

#: Precomputed outcomes replayed from a record set, so two policies that select the same
#: configuration on the same frame see identical predictions. The bridge to real
#: frame x config predictions.
executor_registry.register("replay", _build_replay)

#: Interface stub; construction raises.
executor_registry.register("real_segmentation", _build_real_segmentation)


def build_executor(name: str, context: ExecutorContext) -> Executor:
    """Construct the named backend from ``context``.

    The single construction path used by the composition root and the CLI, so that the
    selected name and the resulting object cannot disagree.
    """
    return executor_registry.create(name, context=context)
