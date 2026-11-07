"""Configuration-selection policies: the subject under evaluation.

Responsibility
--------------
Implement ``Policy.select_config(contract, state, configs) -> str``. A policy sees only the
contract, the policy-visible runtime state, and the allowed configuration descriptions.
Public configuration profiles are visible only when the benchmark mode grants them --
injected at construction, never via an ad-hoc import.

Modules
-------
- ``base``       -- the ``Policy`` protocol, the registry, and ``estimated_latency_s``.
- ``static``     -- ``StaticPolicy``: always the same configuration. The number adaptation
                    has to beat.
- ``rule_based`` -- ``RuleBasedPolicy``: privacy, reachability, communication budget,
                    deadline projection, battery headroom, latency budget, then quality.
                    Explicitly not claimed to be optimal.
- ``registry``   -- name-to-policy mapping for the composition root.

Boundaries
----------
A policy must never import a task evaluator, read ground truth, or parse a ``config_id`` to
deduce behaviour. Selection is driven by typed configuration metadata
(``strategy.placement``, ``strategy.precision``) and, when disclosed, public profiles.

A policy must also tolerate profiles being hidden. Whether they are disclosed is a property
of the run; a policy that crashes without them is not a valid submission.
"""

from missionbench.policies.base import Policy, estimated_latency_s
from missionbench.policies.registry import policy_registry
from missionbench.policies.rule_based import RuleBasedPolicy, RuleBasedSettings
from missionbench.policies.static import StaticPolicy

__all__ = [
    "Policy",
    "RuleBasedPolicy",
    "RuleBasedSettings",
    "StaticPolicy",
    "estimated_latency_s",
    "policy_registry",
]
