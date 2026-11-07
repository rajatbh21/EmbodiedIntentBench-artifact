"""Static baselines: always select the same configuration.

These exist to answer "what does adaptation actually buy?" A benchmark whose adaptive
policies cannot beat *always use the strong local model* has not demonstrated anything, so
these are the number every other policy is measured against.

A static policy names a configuration, which is not the same as inferring behaviour from an
ID -- the prohibition is on *parsing* an ID to deduce placement or quality. Naming a target
is the whole content of "always use X".

If the named configuration is not in an episode's allowed pool, the policy still returns it.
The action validator records the invalid action and substitutes, and the resulting
invalid-action count is the honest report that this baseline does not apply to that episode.
Silently substituting something else would report a different policy's score under this
policy's name.
"""

from __future__ import annotations

from missionbench.schemas.configuration import ConfigCatalog
from missionbench.schemas.contract import Contract
from missionbench.schemas.runtime_state import RuntimeState

__all__ = ["StaticPolicy"]


class StaticPolicy:
    """Always selects one configuration, whatever the state."""

    __slots__ = ("_config_id",)

    def __init__(self, config_id: str) -> None:
        self._config_id = config_id

    @property
    def config_id(self) -> str:
        return self._config_id

    def select_config(
        self,
        contract: Contract,
        state: RuntimeState,
        configs: ConfigCatalog,
    ) -> str:
        del contract, state, configs
        return self._config_id

    def __repr__(self) -> str:
        return f"StaticPolicy({self._config_id!r})"
