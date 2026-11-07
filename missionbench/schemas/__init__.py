"""Typed benchmark schemas and their JSON loading/validation.

Responsibility
--------------
Define the immutable domain models that every other layer speaks in, and own *all* JSON
parsing and validation for them. Nothing outside this package should call ``json.load``
on a benchmark specification file.

Modules
-------
- ``common``        -- primitives shared across schemas (threshold comparison).
- ``contract``      -- mission contract: what must be achieved, and the hard constraints.
- ``episode``       -- initial episode state: platform, path, trace, allowed configs, seed.
- ``configuration`` -- configuration identity (``config_id``/``model_id``/``strategy``)
                       and the configuration catalog.
- ``platform``      -- UAV platform profile: capacity, flight power, comms energy.
- ``task_spec``     -- task definition: evidence type, matching rule, deduplication.
- ``network_trace`` -- trace data and the single observation a policy may see.
- ``path``          -- the predefined path, reduced to the length the loop consumes.
- ``profile``       -- per-platform configuration costs, and the public view of them
                       that a policy may be shown.
- ``runtime_state`` -- the policy-visible observation built fresh at every decision step.
- ``loading``       -- schema-version gate, field validation, and the seam a future
                       migration layer hooks into.

Boundaries
----------
- Models are frozen dataclasses; the simulator never mutates a schema object in place.
- ``RuntimeState`` must never carry ground truth or future network values. See
  ``docs/architecture.md`` ("Policy-visible vs hidden state").
- Validation is strict: unknown fields are rejected, not silently ignored, so that a
  misspelled key never reads as an intentional default.
- Cross-document consistency (a contract's metric against its task, an episode's power
  mode against its platform) cannot be checked at load time, since neither file can see
  the other. Those checks are explicit functions the runner calls once the pair is
  resolved: ``task_spec.check_contract_is_supported``, ``platform.check_episode_power_mode``.
"""

from missionbench.schemas.common import ComparisonOperator
from missionbench.schemas.configuration import (
    ConfigCatalog,
    Configuration,
    Placement,
    Precision,
    Strategy,
    load_config_catalog,
)
from missionbench.schemas.contract import Contract, PrivacyLevel, load_contract
from missionbench.schemas.episode import Episode, load_episode
from missionbench.schemas.loading import (
    SUPPORTED_SCHEMA_VERSIONS,
    SchemaValidationError,
    SchemaVersionError,
)
from missionbench.schemas.network_trace import (
    NetworkObservation,
    NetworkTrace,
    NetworkTraceSegment,
    load_network_trace,
)
from missionbench.schemas.path import PathSpec, load_path_spec
from missionbench.schemas.platform import (
    PlatformProfile,
    check_episode_power_mode,
    load_platform_profile,
)
from missionbench.schemas.profile import (
    ConfigurationProfile,
    ProfileCatalog,
    PublicProfile,
    PublicProfileView,
    QualityTier,
    check_catalog_is_profiled,
    load_profile_catalog,
)
from missionbench.schemas.runtime_state import EvidenceSummary, RuntimeState
from missionbench.schemas.task_spec import (
    DeduplicationMethod,
    DeduplicationRule,
    MatchingRule,
    TaskSpec,
    check_contract_is_supported,
    load_task_spec,
)

__all__ = [
    "SUPPORTED_SCHEMA_VERSIONS",
    "ComparisonOperator",
    "ConfigCatalog",
    "Configuration",
    "ConfigurationProfile",
    "Contract",
    "DeduplicationMethod",
    "DeduplicationRule",
    "Episode",
    "EvidenceSummary",
    "MatchingRule",
    "NetworkObservation",
    "NetworkTrace",
    "NetworkTraceSegment",
    "PathSpec",
    "Placement",
    "PlatformProfile",
    "Precision",
    "PrivacyLevel",
    "ProfileCatalog",
    "PublicProfile",
    "PublicProfileView",
    "QualityTier",
    "RuntimeState",
    "SchemaValidationError",
    "SchemaVersionError",
    "Strategy",
    "TaskSpec",
    "check_catalog_is_profiled",
    "check_contract_is_supported",
    "check_episode_power_mode",
    "load_config_catalog",
    "load_contract",
    "load_episode",
    "load_network_trace",
    "load_path_spec",
    "load_platform_profile",
    "load_profile_catalog",
    "load_task_spec",
]
