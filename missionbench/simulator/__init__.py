"""The deterministic episode simulator: the benchmark core.

Responsibility
--------------
Orchestrate the one-second decision loop and produce a structured episode record. This
package is the *only* part of the benchmark that owns the loop, and it must stay free of
task-specific logic.

Modules
-------
- ``state_manager``   -- ``SimulationState`` (internal) and ``StateManager``, which owns
                         the step transition and projects the policy-visible ``RuntimeState``.
- ``battery_model``   -- ``BatteryModel`` protocol, ``SimpleBatteryModel``, the joule-based
                         energy accounting, and the component ledger.
- ``network_trace``   -- ``NetworkModel`` protocol and ``TraceBasedNetworkModel`` lookup.
- ``path``            -- ``ConstantVelocityPath``: path progress from elapsed time.
- ``timing``          -- the decision-interval and frame-index rules, stated once.
- ``termination``     -- ``TerminationCondition`` checks (path complete, deadline, battery).
- ``action_validator``-- validates the returned config ID against the catalog, the allowed
                         pool, and the contract's privacy level, and resolves a safe
                         substitute for an invalid action.
- ``episode_runner``  -- the loop itself; arrives once executors, evidence trackers, and
                         policies exist to compose (see docs/branching.md).

Hard rule
---------
``EpisodeRunner`` must not import a task evaluator, contain mask/IoU logic, branch on
``task_id``, or parse meaning out of configuration ID strings. Task behaviour arrives
through the ``TaskDefinition`` resolved from the task registry; resource behaviour arrives
through injected models. Adding a second task must not touch this package.
"""
