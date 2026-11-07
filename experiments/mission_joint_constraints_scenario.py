"""Factor-specific placement costs over immutable RGB, trajectories and model semantics."""

import copy
from dataclasses import asdict, replace
from pathlib import Path

from missionbench.schemas.configuration import (
    ConfigCatalog,
    Configuration,
    Placement,
    Precision,
    Strategy,
)
from missionbench.schemas.profile import PublicProfile, PublicProfileView, QualityTier
from missionbench.v2.network import NetworkRegime, V2NetworkModel
from experiments.lake_causal_router import MissionCosts
from experiments.lake_feasibility_inference import read
from experiments.lake_tracking_runtime import TrackingCost
from experiments.mission_network_emulation import LinkSegment, LinkTrace
from experiments.mission_network_policies import ActionDescriptor, Policy, PolicyCosts
from experiments.mission_routing_v2.flights import load_scenario
from experiments.v4_benchmark.calibrated_router import model_signature
from experiments.v4_benchmark.resources import ResourceConfig


def derivative_documents(original, intent, outer, latency, battery, scale):
    scenario, intent, outer = map(copy.deepcopy, (original, intent, outer))
    for data in (intent["limits"], outer):
        data["privacy_level"] = "remote_allowed"
        data["communication_budget_mb"] = 100.0
    scenario["mission"]["initial_battery_frac"] = battery
    intent["mission"]["initial_battery_frac"] = battery
    for spec in scenario["executor_configs"]:
        spec["mission_latency_s"] = latency[spec["config_id"]] * scale
        spec["energy_j_per_call"] = 15 * spec["mission_latency_s"]
    for public in scenario["public_profiles"]:
        public["expected_latency_ms"] = 1000 * latency[public["config_id"]] * scale
    return scenario, intent, outer


def trace_for(plan, cell, scenario=None):
    # Both public LinkSnapshot and private transport use mission-origin ns.
    return LinkTrace(tuple(LinkSegment(**s) for s in plan["traces"][plan["cases"][cell]["link"]]))


def base_scenario_for(root, plan, cell):
    base = load_scenario(
        Path(root) / "missions" / cell / plan["cases"][cell]["mission"] / "scenario.json",
        plan["cases"][cell]["environment"],
        Path(plan["assets"]),
    )
    return replace(
        base,
        source={
            **base.source,
            "joint_onboard_latency_scale": plan["cases"][cell]["onboard_latency_scale"],
        },
    )


def actions_for(plan, scenario):
    return tuple(
        ActionDescriptor(**a)
        for a in plan["actions_by_scale"][str(scenario.source["joint_onboard_latency_scale"])]
    )


def scenario_for(root, plan, cell):
    base = base_scenario_for(root, plan, cell)
    actions = actions_for(plan, base)
    specs = {s.config_id: s for s in base.executors}
    if {s.config_id: model_signature(s) for s in base.executors} != plan["model_signatures"]:
        raise ValueError("Changed base neural semantics")
    catalog, executors, profiles, resources = [], [], {}, {}
    for action in actions:
        spec = specs[action.source_config_id]
        local = action.placement == "local"
        catalog.append(
            Configuration(
                action.action_id,
                spec.model_strategy_id,
                Strategy(
                    Placement.LOCAL if local else Placement.REMOTE,
                    Precision.FP32,
                    input_compression=None if local else "png_lossless",
                    parameters={"source_config_id": spec.config_id},
                ),
            )
        )
        executors.append(
            replace(
                spec,
                config_id=action.action_id,
                kind=spec.kind if local else "virtual_remote_cached_prediction",
                mission_latency_s=action.latency_s,
                energy_j_per_call=action.energy_j,
            )
        )
        profiles[action.action_id] = PublicProfile(
            action.action_id, action.latency_s * 1000, 0.0, QualityTier(spec.quality_tier)
        )
        resources[action.action_id] = dict(
            memory_mb=action.memory_mb,
            load_latency_s=action.load_latency_s,
            load_energy_j=action.load_energy_j,
        )
    segments = trace_for(plan, cell).segments
    network = V2NetworkModel(
        tuple(
            NetworkRegime(
                "authored_" + str(i),
                s.start_ns / 1e9,
                0 if s.outage else s.upload_bps / 1e6,
                0 if s.outage else s.download_bps / 1e6,
                s.rtt_ns / 1e6,
                0.0,
            )
            for i, s in enumerate(segments)
        )
    )
    resource = ResourceConfig.parse(
        dict(memory_capacity_mb=4096, models=resources), [a.action_id for a in actions]
    )
    source = copy.deepcopy(base.source)
    source.update(
        executor_configs=[asdict(s) for s in executors],
        virtual_placement_actions=[asdict(a) for a in actions],
        network_trace=[r.to_dict() for r in network.regimes],
        resources=resource.to_dict(),
    )
    return replace(
        base,
        executors=tuple(executors),
        catalog=ConfigCatalog(catalog),
        profiles=PublicProfileView(profiles),
        resources=resource,
        network=network,
        source=source,
    )


def load_profiles(root, plan):
    return {k: read(Path(root) / p) for k, p in plan["profile_paths"].items()}


def tracking_cost(plan):
    return TrackingCost(**plan["tracking_cost"])


def policy_costs(plan, scenario):
    return PolicyCosts(MissionCosts.from_scenario(scenario), **plan["policy_costs"])


def policy_for(root, plan, name, scenario):
    profiles = load_profiles(root, plan)
    # Local actions keep the original base IDs and original neural kind/parameters.
    local_ids = {a.action_id for a in actions_for(plan, scenario) if a.placement == "local"}
    base_specs = tuple(s for s in scenario.executors if s.config_id in local_ids)
    return Policy(
        name,
        actions_for(plan, scenario),
        base_specs,
        profiles["recall"],
        profiles["risk"],
        policy_costs(plan, scenario),
    )
