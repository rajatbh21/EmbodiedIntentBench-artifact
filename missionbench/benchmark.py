"""The composition root: turn a data directory plus a policy name into results.

This is the one module allowed to know about every layer at once. It resolves fixtures by
ID, constructs concrete components from the registries, injects them into an
``EpisodeRunner``, and scores what comes back. Everything below it talks through protocols.

Keeping composition here is what makes the extensibility claims real. Swapping the executor,
the battery model, or the task is a change to *this* file or a registry entry -- never to the
runner, the metrics, or a policy.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

from missionbench import SCHEMA_VERSION, __version__
from missionbench.executor.base import Executor
from missionbench.executor.registry import ExecutorContext, build_executor
from missionbench.executor.replay_executor import ReplayRecordSet, load_replay_record_set
from missionbench.metrics.aggregate_metrics import AggregateMetrics, aggregate_metrics
from missionbench.metrics.episode_metrics import EpisodeMetrics, compute_episode_metrics
from missionbench.policies.base import Policy
from missionbench.policies.registry import policy_registry
from missionbench.schemas.configuration import ConfigCatalog, load_config_catalog
from missionbench.schemas.contract import Contract, load_contract
from missionbench.schemas.episode import Episode, load_episode
from missionbench.schemas.loading import SchemaValidationError
from missionbench.schemas.network_trace import NetworkTrace, load_network_trace
from missionbench.schemas.path import PathSpec, load_path_spec
from missionbench.schemas.platform import PlatformProfile, load_platform_profile
from missionbench.schemas.profile import (
    ProfileCatalog,
    PublicProfileView,
    check_catalog_is_profiled,
    load_profile_catalog,
)
from missionbench.schemas.task_spec import TaskSpec, check_contract_is_supported, load_task_spec
from missionbench.simulator.action_validator import (
    LEGACY_FALLBACK_CONFIG_ID,
    ActionValidator,
    fallback_rejection_reason,
)
from missionbench.simulator.battery_model import BatteryModel, SimpleBatteryModel
from missionbench.simulator.episode_runner import EpisodeRunner
from missionbench.simulator.network_trace import NetworkModel, TraceBasedNetworkModel
from missionbench.simulator.path import ConstantVelocityPath
from missionbench.simulator.records import EpisodeRecord
from missionbench.simulator.state_manager import StateManager
from missionbench.tasks.registry import resolve_task

__all__ = [
    "DEFAULT_EXECUTOR",
    "BenchmarkData",
    "EpisodeResult",
    "SuiteResult",
    "resolve_fallback_config_id",
    "run_episode",
    "run_suite",
]

#: The executor used when none is named. Profile-driven simulation: no GPU, no weights,
#: exactly reproducible, and every number it reports synthetic.
DEFAULT_EXECUTOR: Final = "profile"

_LOGGER: Final = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """One scored episode: the full record, and the metrics derived from it."""

    record: EpisodeRecord
    metrics: EpisodeMetrics


#: Seeds for repeat r of an episode are ``episode.seed + r * SEED_STRIDE``. The stride keeps
#: two episodes from colliding onto the same seed -- and therefore onto identical synthetic
#: predictions -- as long as their declared seeds differ by less than it. Repeat 0 leaves the
#: seed untouched, so a single-repeat suite is exactly the episode as written.
SEED_STRIDE: Final = 10_000


@dataclass(frozen=True, slots=True)
class SuiteResult:
    """A policy's results over a set of episodes."""

    policy_name: str
    #: Provenance of the backend that produced these episodes, read from the episode
    #: records rather than named independently. If a suite somehow mixed backends this is
    #: the sorted set of what actually ran, so it can never quietly claim one when another
    #: was used.
    executor_id: str
    episodes: tuple[EpisodeResult, ...]
    aggregate: AggregateMetrics
    #: How many seeds each episode was run under. See :func:`run_suite`.
    repeats: int = 1

    @staticmethod
    def _executor_id_from(episodes: tuple[EpisodeResult, ...]) -> str:
        ids = sorted({result.record.executor_id for result in episodes})
        return "+".join(ids) if ids else ""

    def to_dict(self, *, include_detail: bool = False) -> dict[str, Any]:
        """Serialise to the V1 result-file shape.

        ``include_detail`` adds the step log and the raw evidence. Both are off by default:
        raw evidence carries ground-truth-derived fields, so a shared result file would
        otherwise disclose the answers.
        """
        return {
            "schema_version": SCHEMA_VERSION,
            "benchmark_version": __version__,
            "policy": self.policy_name,
            "executor_id": self.executor_id,
            "repeats": self.repeats,
            "aggregate": self.aggregate.to_dict(),
            "episodes": [
                {
                    **result.metrics.to_dict(),
                    "record": result.record.to_dict(include_detail=include_detail),
                }
                for result in self.episodes
            ],
        }


class BenchmarkData:
    """Every fixture under a data root, indexed by the IDs that reference it."""

    __slots__ = (
        "_catalog",
        "_ground_truth_dir",
        "_paths",
        "_platforms",
        "_profiles",
        "_replay_sets",
        "_root",
        "_task_specs",
        "_traces",
    )

    def __init__(self, root: Path) -> None:
        if not root.is_dir():
            raise SchemaValidationError(f"benchmark data directory not found: {root}")
        self._root = root
        self._catalog = _single_catalog(root / "configs")
        self._platforms = {
            profile.platform_id: profile
            for profile in _load_all(root / "platforms", load_platform_profile)
        }
        self._profiles = {
            catalog.platform_id: catalog
            for catalog in _load_all(root / "profiles", load_profile_catalog)
        }
        self._paths = {spec.path_id: spec for spec in _load_all(root / "paths", load_path_spec)}
        self._traces = {
            trace.trace_id: trace
            for trace in _load_all(root / "network_traces", load_network_trace)
        }
        self._task_specs = {
            spec.task_id: spec for spec in _load_all(root / "task_specs", load_task_spec)
        }
        self._ground_truth_dir = root / "ground_truth"
        self._replay_sets: dict[str, ReplayRecordSet] = {}
        for record_set in _load_all(root / "predictions", load_replay_record_set):
            if record_set.episode_id in self._replay_sets:
                # Two sets claiming one episode would make replay provenance depend on
                # filesystem order -- exactly the ambiguity this branch exists to remove.
                raise SchemaValidationError(
                    f"two replay record sets declare episode_id {record_set.episode_id!r}; "
                    "each episode may have at most one replay set under predictions/"
                )
            self._replay_sets[record_set.episode_id] = record_set

    @property
    def root(self) -> Path:
        return self._root

    @property
    def catalog(self) -> ConfigCatalog:
        return self._catalog

    @property
    def ground_truth_dir(self) -> Path:
        return self._ground_truth_dir

    def platform(self, platform_id: str) -> PlatformProfile:
        return _lookup(self._platforms, platform_id, "platform")

    def profiles(self, platform_id: str) -> ProfileCatalog:
        return _lookup(self._profiles, platform_id, "configuration profile set for platform")

    def path(self, path_id: str) -> PathSpec:
        return _lookup(self._paths, path_id, "path")

    def trace(self, trace_id: str) -> NetworkTrace:
        return _lookup(self._traces, trace_id, "network trace")

    def task_spec(self, task_id: str) -> TaskSpec:
        return _lookup(self._task_specs, task_id, "task specification")

    def replay_record_set(self, episode_id: str) -> ReplayRecordSet | None:
        """Return the replay records recorded from ``episode_id``, if any were.

        ``None`` rather than an error: an episode without a record set is only a problem
        when the replay executor is actually selected, and that is where the error belongs
        -- with the instruction for how to record one.
        """
        return self._replay_sets.get(episode_id)

    def episodes(self) -> tuple[Episode, ...]:
        return tuple(_load_all(self._root / "episodes", load_episode))

    def contracts(self) -> tuple[Contract, ...]:
        return tuple(_load_all(self._root / "contracts", load_contract))


def resolve_fallback_config_id(
    *,
    episode: Episode,
    catalog: ConfigCatalog,
    privacy_level,
    explicit: str | None = None,
) -> str:
    """Resolve the safe fallback configuration for an episode, by explicit precedence.

    A misbehaving policy's invalid action is replaced by this configuration when there is no
    current one to keep, so it must be resolved and validated *before* the episode starts, and
    it must not silently assume any particular configuration name exists. Precedence:

    1. ``explicit`` (e.g. a ``--fallback-config-id`` CLI override), if supplied.
    2. The episode's ``fallback_config_id``, if supplied.
    3. The episode's ``initial_config_id``, if it is a usable configuration.
    4. The legacy ``CFG_LOCAL_LIGHT``, only if it exists and is usable (backward compatibility).
    5. Otherwise fail, requiring an explicit fallback.

    A fallback *supplied* at (1) or (2) but unusable is an error, not a reason to fall through:
    a named-but-broken fallback is a misconfiguration worth surfacing. Steps (3) and (4) are
    best-effort and fall through when their candidate is not usable. "Usable" means present in
    the catalog, in the episode's allowed pool, and permitted by the privacy level -- never an
    arbitrary "first configuration in the catalog".
    """

    def reject(config_id: str) -> str | None:
        return fallback_rejection_reason(
            config_id,
            catalog=catalog,
            allowed_config_ids=episode.allowed_config_ids,
            privacy_level=privacy_level,
        )

    for source, candidate in (
        ("--fallback-config-id override", explicit),
        (f"episode {episode.episode_id!r} fallback_config_id", episode.fallback_config_id),
    ):
        if candidate is not None:
            problem = reject(candidate)
            if problem is not None:
                raise SchemaValidationError(
                    f"{source} {candidate!r} is unusable as a fallback: {problem}"
                )
            return candidate

    for candidate in (episode.initial_config_id, LEGACY_FALLBACK_CONFIG_ID):
        if candidate is not None and reject(candidate) is None:
            return candidate

    raise SchemaValidationError(
        f"episode {episode.episode_id!r} has no usable safe fallback configuration. Its "
        f"allowed pool is {list(episode.allowed_config_ids)} and the legacy default "
        f"{LEGACY_FALLBACK_CONFIG_ID!r} is not available. Pass --fallback-config-id, or set the "
        "episode's fallback_config_id (or a usable initial_config_id) to a configuration that "
        "is in the catalog, in the allowed pool, and permitted by the privacy level."
    )


def run_episode(
    *,
    data: BenchmarkData,
    episode: Episode,
    contract: Contract,
    policy_name: str = "rule_based",
    policy: Policy | None = None,
    executor_name: str = DEFAULT_EXECUTOR,
    executor: Executor | None = None,
    replay_strict: bool = False,
    battery_model: BatteryModel | None = None,
    network_model: NetworkModel | None = None,
    disclose_profiles: bool = True,
    fallback_config_id: str | None = None,
) -> EpisodeResult:
    """Assemble the components for one episode, run it, and score the result.

    Args:
        disclose_profiles: Whether policies may consult public configuration profiles. An
            explicit benchmark-mode setting, not something a policy can arrange for itself.
        policy: A constructed policy, overriding ``policy_name``. Lets an external policy be
            evaluated without registering it.
        executor_name: Which registered backend to build. See ``executor_registry``.
        executor: An already-constructed backend, overriding ``executor_name``. Its
            provenance is read from the object's ``executor_id``, never from
            ``executor_name`` -- an unregistered backend is recorded as
            ``"unregistered:<ClassName>"`` rather than mislabelled as a registered one.
        replay_strict: Whether a missing replay record raises instead of being recorded as
            a failed inference.
        fallback_config_id: An explicit safe-fallback override. When ``None`` the fallback is
            resolved from the episode by :func:`resolve_fallback_config_id`; when given it takes
            precedence and must itself be a usable configuration.
        battery_model: Overrides ``SimpleBatteryModel``. The seam a recorded-discharge or
            electrochemical model plugs into.
        network_model: Overrides the trace-based model. The seam an external simulator
            adapter plugs into, since it is where "what the link is doing" comes from.
    """
    platform = data.platform(episode.platform_id)
    profiles = data.profiles(episode.platform_id)
    task_spec = data.task_spec(contract.task_id)

    # Cross-document checks, up front. A mismatch discovered mid-episode would surface as a
    # confusing failure on whichever step first touched it.
    check_contract_is_supported(
        task_spec, task_id=contract.task_id, quality_metric=contract.quality_metric
    )
    check_catalog_is_profiled(
        profiles, config_ids=episode.allowed_config_ids, platform_id=episode.platform_id
    )
    # Resolve the safe fallback up front, by explicit precedence, so a data root with its own
    # configuration names is not obliged to contain the legacy CFG_LOCAL_LIGHT. Fails here,
    # before any step runs, if nothing usable can be resolved.
    resolved_fallback = resolve_fallback_config_id(
        episode=episode,
        catalog=data.catalog,
        privacy_level=contract.privacy_level,
        explicit=fallback_config_id,
    )

    task = resolve_task(task_spec)
    ground_truth = task.load_ground_truth(data.ground_truth_dir, episode)
    if ground_truth is None:
        _LOGGER.warning(
            "no ground truth for frame stream %s; quality cannot be scored",
            episode.frame_stream_id,
        )

    resolved_policy = (
        policy
        if policy is not None
        else _build_policy(policy_name, profiles, disclose_profiles=disclose_profiles)
    )
    resolved_executor = (
        executor
        if executor is not None
        else build_executor(
            executor_name,
            ExecutorContext(
                episode_id=episode.episode_id,
                profiles=profiles,
                prediction_source=task.create_prediction_source(ground_truth, profiles),
                replay_record_set=data.replay_record_set(episode.episode_id),
                replay_strict=replay_strict,
            ),
        )
    )

    runner = EpisodeRunner(
        episode=episode,
        contract=contract,
        catalog=data.catalog,
        state_manager=StateManager(
            episode=episode,
            platform=platform,
            path=ConstantVelocityPath.from_spec(
                data.path(episode.path_id), velocity_mps=episode.velocity_mps
            ),
            battery_model=battery_model if battery_model is not None else SimpleBatteryModel(),
        ),
        network_model=(
            network_model
            if network_model is not None
            else TraceBasedNetworkModel(data.trace(episode.network_trace_id))
        ),
        executor=resolved_executor,
        policy=resolved_policy,
        evidence_tracker=task.create_tracker(),
        action_validator=ActionValidator(
            catalog=data.catalog,
            allowed_config_ids=episode.allowed_config_ids,
            privacy_level=contract.privacy_level,
            fallback_config_id=resolved_fallback,
        ),
        policy_name=policy_name,
    )

    record = runner.run()
    evaluation = task.create_evaluator().evaluate(record.evidence, ground_truth, contract)
    return EpisodeResult(
        record=record, metrics=compute_episode_metrics(record, evaluation, contract)
    )


def run_suite(
    *,
    data: BenchmarkData,
    episodes: Sequence[Episode],
    contract: Contract,
    policy_name: str = "rule_based",
    policy: Policy | None = None,
    executor_name: str = DEFAULT_EXECUTOR,
    disclose_profiles: bool = True,
    replay_strict: bool = False,
    repeats: int = 1,
    fallback_config_id: str | None = None,
) -> SuiteResult:
    """Run every episode against one contract and aggregate the results.

    Args:
        repeats: How many seeds to run each episode under. The default of 1 runs the
            episodes exactly as written.

    Why repeats exist
    -----------------
    Mission Success Rate over three episodes can only be 0, 1/3, 2/3 or 1, and the shipped
    quality scores vary by roughly 0.05 between seeds while the margins that decide
    pass/fail are around 0.02. Three episodes therefore cannot resolve the difference
    between two policies: the baselines measure 67 % and 100 % over three episodes and 92 %
    and 96 % over 150, and the second pair is not a significant difference at all.

    Repeats resample the *scene* -- the seed drives synthetic prediction generation -- while
    holding the path, platform, network trace, and targets fixed. That is genuine additional
    sampling, but it is not the same as adding independent episodes: the shipped episodes
    already share one ground-truth stream, so this narrows the interval without removing
    that correlation.

    A policy instance is reused across repeats when one is passed directly, so a stateful
    policy sees the whole grid. Pass a fresh instance per suite if that matters.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats}")

    results = tuple(
        run_episode(
            data=data,
            episode=episode
            if repeat == 0
            else replace(episode, seed=episode.seed + repeat * SEED_STRIDE),
            contract=contract,
            policy_name=policy_name,
            policy=policy,
            executor_name=executor_name,
            disclose_profiles=disclose_profiles,
            replay_strict=replay_strict,
            fallback_config_id=fallback_config_id,
        )
        for episode in episodes
        for repeat in range(repeats)
    )
    return SuiteResult(
        policy_name=policy_name,
        executor_id=SuiteResult._executor_id_from(results),
        episodes=results,
        aggregate=aggregate_metrics([result.metrics for result in results]),
        repeats=repeats,
    )


def _build_policy(policy_name: str, profiles: ProfileCatalog, *, disclose_profiles: bool) -> Policy:
    view = profiles.public_view() if disclose_profiles else PublicProfileView.hidden()
    try:
        return policy_registry.create(policy_name, public_profiles=view)
    except TypeError:
        # A policy that takes no profiles is legitimate -- the static baselines do not.
        return policy_registry.create(policy_name)


def _load_all(directory: Path, loader):
    if not directory.is_dir():
        return []
    return [loader(path) for path in sorted(directory.glob("*.json"))]


def _single_catalog(directory: Path) -> ConfigCatalog:
    catalogs = _load_all(directory, load_config_catalog)
    if len(catalogs) != 1:
        raise SchemaValidationError(
            f"expected exactly one configuration catalog in {directory}, found {len(catalogs)}. "
            "V1 runs one catalog per data root; selecting among several is not yet specified."
        )
    return catalogs[0]


def _lookup(index: dict[str, Any], key: str, kind: str):
    try:
        return index[key]
    except KeyError:
        raise SchemaValidationError(
            f"no {kind} {key!r} in the benchmark data; available: {sorted(index)}"
        ) from None


def episodes_by_id(data: BenchmarkData, episode_ids: Iterable[str]) -> tuple[Episode, ...]:
    """Select episodes by ID, preserving the requested order."""
    index = {episode.episode_id: episode for episode in data.episodes()}
    return tuple(_lookup(index, episode_id, "episode") for episode_id in episode_ids)
