"""Phase-local predicted-track support; never a true-target correctness estimate."""

import math

from experiments.mission_routing_v3.policy import support_probability
from experiments.mission_routing_v5.evidence import Policy as PreviousEvidence

PRIOR_MASS = 2.0


def support_rate(prior, hits, choices, config, start, end, now, max_gap):
    """Shrink observed reassociation frequency toward the current RGB recall prior.

    Count only completed actions after this track's first phase-local observation,
    while it was live BEFORE that action. The discovery hit is therefore not an
    extra success trial. Non-reassociation can mean occlusion or exit, not model
    error; this deliberately remains a heuristic and is not calibrated recall.
    """
    if not 0 <= prior <= 1 or max_gap <= 0:
        raise ValueError("Invalid support prior or gap")
    local = sorted({t for t in hits if start <= t < end and t < now})
    successes, attempts = 0, 0
    for capture, selected in sorted(choices.items()):
        if selected != config or not start <= capture < min(end, now):
            continue
        earlier = [t for t in local if t < capture]
        if not earlier or capture - earlier[-1] > max_gap:
            continue
        attempts += 1
        successes += capture in local
    return dict(
        prior=prior,
        successes=successes,
        attempts=attempts,
        support_rate=(PRIOR_MASS * prior + successes) / (PRIOR_MASS + attempts),
    )


class Policy(PreviousEvidence):
    def select(self, observation):
        super().select(observation)
        o = observation
        trace = self.last_decision
        phases = {
            p["phase_id"]: p
            for p in o.intent_contract["mission"]["phases"]
            if p["start_s"] <= o.state.current_time_s < p["end_s"]
        }
        gap = o.tracking["configuration"]["max_gap_s"]
        for candidate in trace["candidates"]:
            utility = 0.0
            for detail in candidate["phases"]:
                phase = phases[detail["phase_id"]]
                pending = []
                for track in o.evidence_history["tracks"]:
                    hits = [
                        t
                        for t in track["capture_times_s"]
                        if phase["start_s"] <= t < phase["end_s"]
                    ]
                    if not hits or (
                        len(hits) >= phase["min_hits"]
                        and hits[-1] - hits[0] >= phase["min_evidence_span_s"]
                    ):
                        continue
                    rate = support_rate(
                        candidate["quality"],
                        hits,
                        self.action_history if self.use_provenance else {},
                        candidate["config_id"],
                        phase["start_s"],
                        phase["end_s"],
                        o.state.current_time_s,
                        gap,
                    )
                    support = support_probability(
                        hits,
                        detail["captures"],
                        rate["support_rate"],
                        phase["min_hits"],
                        phase["min_evidence_span_s"],
                        gap,
                    )
                    pending.append(
                        dict(
                            track_id=track["track_id"],
                            phase_hits=hits,
                            **rate,
                            completion_support=support,
                        )
                    )
                values = [t["completion_support"] for t in pending]
                continuation = (
                    (min(values) if self.use_weakest else sum(values) / len(values))
                    if values
                    else 0.0
                )
                # Discovery remains active even when predicted tracks are supported.
                utility += phase["weight"] * (detail["discovery_support"] + continuation)
                detail.update(confirmation_support=continuation, predicted_track_support=pending)
            candidate["utility"] = utility

        def energy(c):
            return c["projected_energy_j"] if c["projected_energy_j"] is not None else math.inf

        feasible = [c for c in trace["candidates"] if c["feasible"]]
        selected = (
            max(feasible, key=lambda c: (c["utility"], -energy(c), c["config_id"]))
            if feasible
            else min(trace["candidates"], key=lambda c: (energy(c), c["config_id"]))
        )
        trace.update(
            schema_version="prediction_support_evidence_v1",
            selected_config_id=selected["config_id"],
            uses_model_provenance=self.use_provenance,
            prioritizes_weakest_track=self.use_weakest,
            support_prior_mass=PRIOR_MASS,
            support_rate_is_calibrated_probability=False,
            completed_action_history=[
                dict(capture_s=t, config_id=c)
                for t, c in sorted(self.action_history.items())
                if t < o.state.current_time_s
            ],
        )
        return selected["config_id"]
