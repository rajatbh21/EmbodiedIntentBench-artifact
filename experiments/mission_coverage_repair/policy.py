"""Phase-horizon anonymous coverage planning; only current RGB and completed actions.

Each anonymous target type has a development-trained response to each model.
The population ledger predicts exposure, not observed target correctness. No
ground-truth count, identity, future frame, or unexecuted prediction is consulted.
"""

import copy
import math

import numpy as np

from experiments.mission_routing_v2.features import NAMES, encode
from experiments.mission_routing_v2.policy import plain
from experiments.mission_routing_v4.risk import estimate
from experiments.mission_routing_v5.evidence import Policy as Evidence
from experiments.mission_strategy_routing.policy import log_count_survival
from experiments.mission_temporal_routing.policy import repetition_blind
from experiments.v4_benchmark.calibrated_router import model_signature
from experiments.v4_benchmark.scenario import digest

MODES = (
    "coverage_tail",
    "coverage_mean",
    "coverage_global",
    "coverage_short",
    "coverage_no_context",
    "coverage_no_intent",
)


def count_step(mass, probability):
    """Poisson-binomial update with an absorbing completed-count state."""
    mass = np.asarray(mass, float)
    if (
        mass.ndim != 2
        or mass.shape[1] < 2
        or not np.isfinite(mass).all()
        or np.any(mass < 0)
        or not np.allclose(mass.sum(axis=1), 1)
    ):
        raise ValueError("Normalized finite count distributions required")
    p = np.asarray(probability, float)
    if (
        mass.ndim != 2
        or p.shape != (len(mass),)
        or not np.isfinite(p).all()
        or np.any((p < 0) | (p > 1))
    ):
        raise ValueError("Finite type-wise probabilities required")
    result = mass * (1 - p[:, None])
    result[:, 1:] += mass[:, :-1] * p[:, None]
    result[:, -1] += mass[:, -1] * p
    # Floating-point accumulation can exceed one by a few ulps at saturation.
    # Renormalize the nonnegative distribution; do not change the objective.
    return result / result.sum(axis=1, keepdims=True)


def coverage_score(probability, weights, required_recall, *, mean_only=False):
    """Blend mean and geometric support as coverage demand rises from 50% to 100%."""
    p = np.asarray(probability, float)
    weights = np.asarray(weights, float)
    if p.shape != weights.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid completion probabilities")
    if not 0 < required_recall <= 1 or not np.isclose(weights.sum(), 1) or np.any(weights <= 0):
        raise ValueError("Valid coverage fraction and positive normalized weights required")
    blend = 0 if mean_only else max(0, 2 * required_recall - 1)
    return float(
        (1 - blend) * math.log(max(float(weights @ p), 1e-12))
        + blend * (weights @ np.log(np.maximum(p, 1e-12)))
    )


class Policy(Evidence):
    def __init__(self, recall, specs, costs, risk, coverage, *, mode="coverage_tail"):
        if mode not in MODES or coverage["schema_version"] != "anonymous_coverage_v1":
            raise ValueError("Unknown coverage policy/profile")
        if (
            risk["schema_version"] != "mission_false_component_risk_v1"
            or risk["feature_names"] != list(NAMES)
            or set(risk["models"]) != {s.config_id for s in specs}
        ):
            raise ValueError("Risk schema or pool differs")
        super().__init__(
            recall,
            specs,
            costs,
            "no_context_v3" if mode == "coverage_no_context" else "confirmation",
        )
        self.coverage_mode = mode
        self.ids = coverage["config_ids"]
        if set(self.ids) != set(self.specs) or len(self.ids) != len(self.specs):
            raise ValueError("Coverage model pool differs")
        for spec in specs:
            if coverage["signatures"][spec.config_id] != model_signature(spec):
                raise ValueError("Coverage model signature differs")
            if risk["models"][spec.config_id]["signature"] != model_signature(spec):
                raise ValueError("Risk model signature differs")
        self.rates = np.asarray(coverage["rates"], float)
        self.weights = np.asarray(coverage["weights"], float)
        if (
            self.rates.shape != (len(self.weights), len(self.ids))
            or not np.isfinite(self.rates).all()
            or np.any((self.rates <= 0) | (self.rates >= 1))
            or not np.isclose(self.weights.sum(), 1)
            or np.any(self.weights <= 0)
        ):
            raise ValueError("Invalid anonymous type probabilities")
        self.marginal = self.weights @ self.rates
        if mode == "coverage_global":
            self.rates = self.marginal[None, :]
            self.weights = np.ones(1)
        self.risk = copy.deepcopy(risk["models"])
        self.coverage_sha = digest(coverage)
        self.risk_sha = digest(risk)
        self.mass, self.ledger = {}, {}
        self._previous, self._input_sha = None, None

    def select(self, observation):
        o = (
            repetition_blind(observation)
            if self.coverage_mode == "coverage_no_intent"
            else observation
        )
        super().select(o)
        base = copy.deepcopy(self.last_decision)
        now, index = o.state.current_time_s, o.state.frame_id
        input_sha = digest(
            dict(
                base=base,
                index=index,
                now=now,
                config=o.state.current_config_id,
                intent=plain(o.intent_contract),
            )
        )
        if self._previous is not None:
            previous, config, completion, trace = self._previous
            if index == previous:
                if input_sha != self._input_sha:
                    raise ValueError("Repeated decision has different public input")
                self.last_decision = copy.deepcopy(trace)
                return config
            if (
                index < previous
                or now < completion
                or o.state.current_config_id != config
                or o.evidence_history["last_completion_s"] is None
                or abs(o.evidence_history["last_completion_s"] - completion) > 1e-7
            ):
                raise ValueError("Previous selected execution must complete")
        phases = [
            p for p in o.intent_contract["mission"]["phases"] if p["start_s"] <= now < p["end_s"]
        ]
        if len(phases) != 1:
            raise ValueError("Coverage planning requires one current phase")
        phase = phases[0]
        k = phase["min_hits"]
        if k not in (2, 3) or phase["min_evidence_span_s"] > (k - 1) * self.costs.interval_s:
            raise ValueError(
                "Count abstraction requires repetition span implied by distinct captures"
            )
        required = 0.5 if self.coverage_mode == "coverage_no_intent" else phase["required_recall"]
        pid = phase["phase_id"]
        mass = self.mass.get(pid)
        if mass is None:
            mass = np.zeros((len(self.weights), k + 1))
            mass[:, 0] = 1
        before = float(self.ledger.get(pid, 0))
        features = encode(o.rgb)
        context = self.coverage_mode != "coverage_no_context"
        quality = {c["config_id"]: c["quality"] for c in base["candidates"]}
        rates = {c: estimate(self.risk[c]["risk"], features, context=context) for c in quality}
        probabilities = {
            c: np.clip(
                quality[c] * self.rates[:, self.ids.index(c)] / self.marginal[self.ids.index(c)],
                0,
                1,
            )
            for c in quality
        }
        available = base["available_energy_j"]
        horizon = (
            min(phase["end_s"], now + 8)
            if self.coverage_mode == "coverage_short"
            else phase["end_s"]
        )
        for candidate in base["candidates"]:
            config = candidate["config_id"]
            state = mass.copy()
            expected = before
            cursor, previous = index, o.state.current_config_id
            spent = 0.0
            tail = False
            planned = []
            # Repeat a candidate while retaining a feasible full-flight continuation.
            # If continuation would exceed budget, use the certified minimum-energy tail.
            while self._budget.schedule.at(cursor) / 1e9 < phase["end_s"]:
                chosen = config
                if (
                    tail
                    or spent + self._budget.required(cursor, previous, chosen) > available + 1e-8
                ):
                    chosen = self._budget.choices[cursor, previous]
                    tail = True
                if not planned:
                    # A first-action guard is shared with every comparator.
                    chosen = config
                completion, next_index, energy, final = self._budget.transition(
                    cursor, previous, chosen
                )
                capture = self._budget.schedule.at(cursor) / 1e9
                expected += rates[chosen]
                if capture < horizon and completion <= phase["deadline_s"]:
                    state = count_step(state, probabilities[chosen])
                planned.append(dict(capture_id=cursor, config_id=chosen, completion_s=completion))
                spent += energy
                if final:
                    break
                cursor, previous = next_index, chosen
            score = coverage_score(
                state[:, -1],
                self.weights,
                required,
                mean_only=self.coverage_mode == "coverage_mean",
            )
            survival = log_count_survival(expected, phase["max_false_components"])
            candidate.update(
                coverage_log_score=score + survival,
                coverage_only_log_score=score,
                phase_risk_log_survival=survival,
                projected_expected_false_components=expected,
                projected_type_support=state[:, -1].tolist(),
                planned_phase_actions=planned,
            )
        feasible = [c for c in base["candidates"] if c["feasible"]]

        def energy(c):
            return c["projected_energy_j"] if c["projected_energy_j"] is not None else math.inf

        selected = (
            max(feasible, key=lambda c: (c["coverage_log_score"], -energy(c), c["config_id"]))
            if feasible
            else min(base["candidates"], key=lambda c: (energy(c), c["config_id"]))
        )
        config = selected["config_id"]
        after = (
            count_step(mass, probabilities[config])
            if selected["first_completion_s"] <= phase["deadline_s"]
            else mass.copy()
        )
        self.mass[pid] = after
        self.ledger[pid] = before + rates[config]
        base.update(
            schema_version="mission_coverage_routing_numeric_v2",
            coverage_mode=self.coverage_mode,
            selected_config_id=config,
            coverage_profile_sha256=self.coverage_sha,
            risk_profile_sha256=self.risk_sha,
            population_ledger_before=mass.tolist(),
            population_ledger_after=after.tolist(),
            phase_id=pid,
            planning_end_s=horizon,
            expected_false_before=before,
            expected_false_after=self.ledger[pid],
            counts_are_observed_targets=False,
            uses_ground_truth=False,
            uses_prediction_history_for_completion=True,
            uses_track_confirmation=False,
            uses_history=False,
            horizon_s=horizon - now,
            uses_context=context,
            uses_mission_repetition=self.coverage_mode != "coverage_no_intent",
            uses_heterogeneous_types=self.coverage_mode != "coverage_global",
            coverage_is_calibrated=False,
        )
        base = plain(base)
        self.last_decision, self._input_sha = base, input_sha
        self._previous = (index, config, selected["first_completion_s"], copy.deepcopy(base))
        return config
