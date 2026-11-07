"""Auditable public-input heuristic and ablations, not a claim of optimal routing."""

import math

import numpy as np

from experiments.v4_benchmark.intent_visual import visual_features


class ContextRouter:
    def __init__(self, parameters, mode="context"):
        self.settings = parameters
        self.mode = mode
        self.last_decision = None

    def select(self, observation):
        p = self.settings
        state = observation.state
        gray, edges = visual_features(observation.rgb)
        contrast = float(gray.std())
        clutter = float(np.mean(edges > 0.15)) if edges.size else 0.0
        use_intent = self.mode != "context_no_intent"
        use_visual = self.mode != "context_no_visual"
        use_resources = self.mode != "context_no_resources"
        demanding = use_intent and (
            observation.requirements.min_report_recall >= 0.8
            or observation.requirements.max_false_reports == 0
        )
        visual_tier = (
            3 if contrast < 0.08 or clutter > 0.2 else 2 if observation.position_m[2] >= 40 else 1
        )
        desired = max(3 if demanding else 1, visual_tier if use_visual else 1)
        time_available = state.remaining_deadline_s
        if use_intent and observation.confirmed_reports_so_far == 0:
            report_deadline = observation.requirements.confirmed_report_deadline_s
            if report_deadline is not None and report_deadline > state.current_time_s:
                time_available = min(time_available, report_deadline - state.current_time_s)
        remaining = max(0, p["duration_s"] - state.current_time_s)
        energy_left = (state.battery_frac - observation.contract.min_final_battery_frac) * p[
            "capacity_j"
        ]
        options = []
        for config in observation.configs:
            cost = p["costs"][config.config_id]
            resource = observation.resources.for_config(config.config_id)
            loading = state.current_config_id != config.config_id
            latency = cost["latency_s"] + (resource.load_latency_s if loading else 0)
            calls = math.ceil(remaining / max(p["interval_s"], cost["latency_s"]))
            energy = p["flight_power_w"] * remaining + calls * cost["energy_j"]
            energy += resource.load_energy_j if loading else 0
            options.append(
                dict(
                    config_id=config.config_id,
                    rank=cost["rank"],
                    latency_s=latency,
                    memory_fits=resource.memory_mb <= observation.resources.memory_capacity_mb,
                    reserve_fits=energy <= energy_left,
                    deadline_fits=latency <= time_available,
                    projected_energy_j=energy,
                )
            )
        allowed = [
            r
            for r in options
            if r["memory_fits"]
            and (not use_resources or (r["reserve_fits"] and r["deadline_fits"]))
        ]
        qualified = [r for r in allowed if r["rank"] >= desired]
        if qualified:
            chosen = min(qualified, key=lambda r: (r["latency_s"], r["config_id"]))
            reason = "lowest_latency_meeting_heuristic_tier"
        elif allowed:
            chosen = min(allowed, key=lambda r: (-r["rank"], r["latency_s"], r["config_id"]))
            reason = "best_available_heuristic_tier"
        else:
            chosen = min(
                options, key=lambda r: (r["projected_energy_j"], r["latency_s"], r["config_id"])
            )
            reason = "no_feasible_candidate_minimum_projected_energy"
        self.last_decision = dict(
            selected_config_id=chosen["config_id"],
            reason=reason,
            uses_intent=use_intent,
            uses_visual=use_visual,
            uses_resources=use_resources,
            rgb_contrast=contrast,
            edge_fraction=clutter,
            altitude_m=observation.position_m[2],
            desired_heuristic_tier=desired,
            quality_guarantee=False,
            candidates=options,
        )
        return chosen["config_id"]
