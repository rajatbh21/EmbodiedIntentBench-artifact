"""EmbodiedIntentBench core contracts, execution interfaces, and mission evaluation.

EmbodiedIntentBench evaluates whether execution strategies satisfy structured
mission intents under resource constraints. The supplied UAV perception studies
use model and execution-location selection as a controlled instance of this
mission-level evaluation problem.

This package retains the original synthetic contract/episode runner alongside
interfaces used by later experiments. In that core runner, a policy observes a
structured contract and public runtime state, selects an inference configuration,
and receives mission-level evaluation under modeled time and resource costs.
The core fixtures do not evaluate flight control or open-ended language understanding.

Use ``tools/artifact.py`` for the bundled paper experiments. See
``docs/ARTIFACT.md`` for reproduction and ``docs/EXPERIMENT.md`` for study scope.
"""

__all__ = ["SCHEMA_VERSION", "__version__"]

__version__ = "0.1.0.dev0"

#: Schema version for the original core fixtures (contract, episode, config catalog,
#: platform, task spec, network trace) and their results. Loading these fixtures with
#: another version must fail rather than silently coerce the schema. Later experiment
#: artifacts have their own explicitly named schemas.
#: Migration support, when it arrives, belongs in ``missionbench.schemas.loading``.
SCHEMA_VERSION = "1.0"
