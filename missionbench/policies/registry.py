"""The policy registry.

Lets ``--policy rule_based`` resolve without the runner importing a policy, and lets an
externally submitted policy join by registering itself.

The three static baselines name configurations from the shipped catalog. That is a property
of these *fixtures*, not of the policy: ``StaticPolicy`` takes the ID as an argument, so a
different catalog registers its own baselines without touching this module.
"""

from __future__ import annotations

from missionbench.policies.base import Policy, policy_registry
from missionbench.policies.budget_planner import BudgetPlannerPolicy
from missionbench.policies.rule_based import RuleBasedPolicy
from missionbench.policies.static import StaticPolicy

__all__ = ["policy_registry"]


def _always(config_id: str):
    def factory(**_: object) -> Policy:
        return StaticPolicy(config_id)

    return factory


policy_registry.register("always_local_light", _always("CFG_LOCAL_LIGHT"))
policy_registry.register("always_local_strong", _always("CFG_LOCAL_STRONG"))
policy_registry.register("always_remote_strong", _always("CFG_REMOTE_STRONG"))

#: Takes ``public_profiles`` and optionally ``settings``; both are keyword arguments so the
#: composition root decides whether this run discloses profiles.
policy_registry.register("rule_based", RuleBasedPolicy)

#: The projecting baseline (V3 P3): paces the communication budget and projects the
#: final battery from its own observations. Stateful within one episode; the
#: composition root constructs policies per run.
policy_registry.register("budget_planner", BudgetPlannerPolicy)

#: Available for a catalog whose configuration IDs differ from the shipped fixtures.
policy_registry.register("static", StaticPolicy)
