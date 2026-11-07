"""Published-method-inspired controllers, explicitly not paper reproductions.

Performance prediction resembles ApproxDet/LiteReconfig. Full-information offline
ridge reward fitting is a simple contextual controller comparator, not Chanakya's
network, streaming reward or online training procedure. Runtime has no label access.
"""

from __future__ import annotations

import copy
import math

from missionbench.simulator.action_validator import privacy_permits
from experiments.lake_causal_router import forecast
from experiments.mission_routing_v2.features import NAMES, encode, predict
from experiments.v4_benchmark.calibrated_router import model_signature
from experiments.v4_benchmark.scenario import digest

MODES = (
    "performance_predictor",
    "offline_contextual",
    "intent_history",
    "no_history",
    "no_context",
    "no_intent",
    "resource_only",
)


def history_need(history, phase, now):
    """Prediction support deficit, never an estimate of true target completeness."""
    needs = []
    for track in history["tracks"]:
        hits = [t for t in track["capture_times_s"] if phase["start_s"] <= t < phase["end_s"]]
        if not hits:
            continue
        missing = max(0, phase["min_hits"] - len(hits))
        span_missing = max(0, phase["min_evidence_span_s"] - (hits[-1] - hits[0]))
        if missing or span_missing:
            needs.append(
                dict(
                    hits=len(hits),
                    missing_hits=missing,
                    first_s=hits[0],
                    last_s=hits[-1],
                    missing_span_s=span_missing,
                )
            )
    return needs


class Policy:
    def __init__(self, profile, specs, costs, mode):
        if mode not in MODES or profile["feature_names"] != list(NAMES):
            raise ValueError("Unknown controller or feature schema")
        if set(profile["models"]) != {s.config_id for s in specs}:
            raise ValueError("Wrong configuration pool")
        for s in specs:
            if profile["models"][s.config_id]["signature"] != model_signature(s):
                raise ValueError("Wrong model signature")
        self.models = copy.deepcopy(profile["models"])
        self.profile_sha = digest(profile)
        self.specs = {s.config_id: s for s in specs}
        self.costs, self.mode, self.last_decision = costs, mode, None

    def select(self, observation):
        o, mode = observation, self.mode
        state, public = o.state, o.intent_contract
        features = encode(o.rgb)
        now = state.current_time_s
        use_intent = mode in ("intent_history", "no_history", "no_context")
        use_history = mode in ("intent_history", "no_context", "no_intent")
        use_context = mode not in ("no_context", "resource_only")
        available = (state.battery_frac - o.contract.min_final_battery_frac) * self.costs.capacity_j
        candidates = []
        for config in o.configs:
            if config.strategy.is_remote or not privacy_permits(o.contract.privacy_level, config):
                continue
            resource = o.resources.for_config(config.config_id)
            if (
                o.resources.memory_capacity_mb is not None
                and resource.memory_mb + o.tracking["cost"]["memory_mb"]
                > o.resources.memory_capacity_mb
            ):
                continue
            s, model = self.specs[config.config_id], self.models[config.config_id]
            quality, uncertainty = predict(model["quality"], features)
            if not use_context:
                quality = model["quality"]["global_mean"]
            load = state.current_config_id != s.config_id
            latency = (
                s.mission_latency_s + self.costs.router_latency_s + o.tracking["cost"]["latency_s"]
            )
            slots = forecast(
                self.costs, state.frame_id, latency, resource.load_latency_s if load else 0
            )
            energy = (
                (max(self.costs.duration_s, slots[-1][1]) - now) * self.costs.flight_power_w
                + len(slots)
                * (
                    s.energy_j_per_call
                    + self.costs.router_energy_j
                    + o.tracking["cost"]["energy_j"]
                )
                + (resource.load_energy_j if load else 0)
            )
            cadence = slots[1][0] - slots[0][0] if len(slots) > 1 else latency
            # Performance-only maximizes predicted single-frame accuracy under resource limits.
            utility = quality
            if mode == "offline_contextual":
                # Separate reward head trained from quality / declared capture cadence.
                utility, _ = predict(model["reward"], features)
            elif mode != "performance_predictor":
                utility = quality / max(cadence, self.costs.interval_s)
            phase_details = []
            if use_intent:
                active = [
                    p for p in public["mission"]["phases"] if p["start_s"] <= now < p["end_s"]
                ]
            else:
                active = []
            for phase in active:
                usable = [
                    t for t, end in slots if t < phase["end_s"] and end <= phase["deadline_s"]
                ]
                opportunity = min(1, len(usable) / phase["min_hits"])
                opportunity *= (
                    min(1, (usable[-1] - usable[0]) / max(0.001, phase["min_evidence_span_s"]))
                    if len(usable) > 1
                    else 0
                )
                utility *= phase["weight"] * opportunity
                needs = history_need(o.evidence_history, phase, now) if use_history else []
                # More pending tracks make timely repeat evidence valuable. No stop-on-count.
                supported = 0
                for need in needs:
                    required = max(1, need["missing_hits"])
                    if (
                        len(usable) >= required
                        and usable[required - 1] - need["first_s"] >= phase["min_evidence_span_s"]
                        and usable[0] - need["last_s"] <= o.tracking["configuration"]["max_gap_s"]
                    ):
                        supported += 1
                if needs:
                    utility *= 1 + supported / len(needs)
                phase_details.append(
                    dict(
                        phase_id=phase["phase_id"],
                        opportunities=len(usable),
                        pending_prediction_tracks=len(needs),
                        supportable=supported,
                    )
                )
            if use_history and not use_intent:
                # Global confirmation, without phase schedules or task weights.
                pending = sum(not t["reported"] for t in o.evidence_history["tracks"])
                utility *= 1 + min(1, pending) * min(1, self.costs.interval_s / max(cadence, 0.001))
            candidates.append(
                dict(
                    config_id=s.config_id,
                    quality=quality,
                    uncertainty=uncertainty,
                    utility=utility,
                    projected_energy_j=energy,
                    feasible=energy <= available and slots[0][1] <= o.contract.deadline_s,
                    first_completion_s=slots[0][1],
                    next_capture_s=slots[1][0] if len(slots) > 1 else None,
                    phases=phase_details,
                )
            )
        if not candidates:
            raise ValueError("No legal local configuration")
        feasible = [c for c in candidates if c["feasible"]]
        if not feasible or mode == "resource_only":
            selected = min(
                feasible or candidates, key=lambda c: (c["projected_energy_j"], c["config_id"])
            )
        else:
            selected = max(
                feasible, key=lambda c: (c["utility"], -c["projected_energy_j"], c["config_id"])
            )
        self.last_decision = dict(
            schema_version="mission_routing_decision_v2",
            mode=mode,
            profile_sha256=self.profile_sha,
            selected_config_id=selected["config_id"],
            uses_ground_truth=False,
            features=features.tolist(),
            uses_intent=use_intent,
            uses_history=use_history,
            history=plain(o.evidence_history),
            available_energy_j=available,
            candidates=candidates,
        )
        if not all(math.isfinite(c["utility"]) for c in candidates):
            raise ValueError("Nonfinite utility")
        return selected["config_id"]


def plain(value):
    from collections.abc import Mapping

    if isinstance(value, Mapping):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value
