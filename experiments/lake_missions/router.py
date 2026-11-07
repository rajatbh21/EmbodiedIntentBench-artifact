"""Public-intent phase adapter; no target IDs, future images or GT access."""

from dataclasses import replace

from experiments.lake_missions.objectives import phase_at


class PhaseRouter:
    def __init__(self, context_router, public_schedule):
        if any("evaluator_target_ids" in p for p in public_schedule):
            raise ValueError("Evaluator targets must not enter the router")
        self.context = context_router
        self.schedule = public_schedule
        self.last_decision = None

    def select(self, observation):
        phase = phase_at(self.schedule, observation.state.current_time_s)
        if phase:
            # Reserve enough phase time for the requested repeated evidence.
            # Quality tiers remain uncalibrated assumptions, not accuracy claims.
            remaining = max(0, phase["deadline_s"] - observation.state.current_time_s)
            per_result_time = remaining / phase["min_hits"]
            state = replace(
                observation.state,
                remaining_deadline_s=min(
                    observation.state.remaining_deadline_s,
                    per_result_time,
                ),
            )
            requirements = replace(
                observation.requirements,
                min_report_recall=phase["required_recall"],
                confirmed_report_deadline_s=phase["deadline_s"],
            )
            observation = replace(observation, state=state, requirements=requirements)
        choice = self.context.select(observation)
        self.last_decision = dict(self.context.last_decision)
        self.last_decision.update(public_phase=phase, phase_adapter="repeated_evidence_time_budget")
        return choice
