"""Baseline policies: the static references and the rule-based heuristic."""

from __future__ import annotations

import dataclasses

import pytest

from missionbench.policies import (
    Policy,
    RuleBasedPolicy,
    RuleBasedSettings,
    StaticPolicy,
    estimated_latency_s,
    policy_registry,
)
from missionbench.registry import RegistryError
from missionbench.schemas import (
    EvidenceSummary,
    NetworkObservation,
    PrivacyLevel,
    PublicProfileView,
    RuntimeState,
)

GOOD = NetworkObservation(bandwidth_mbps=20.0, rtt_ms=30.0, packet_loss_frac=0.0)
DEGRADED = NetworkObservation(bandwidth_mbps=8.0, rtt_ms=70.0, packet_loss_frac=0.01)
POOR = NetworkObservation(bandwidth_mbps=2.0, rtt_ms=150.0, packet_loss_frac=0.03)
DISCONNECTED = NetworkObservation(bandwidth_mbps=0.0, rtt_ms=0.0, packet_loss_frac=1.0)


def _state(
    *,
    network: NetworkObservation = GOOD,
    battery_frac: float = 0.80,
    time_s: float = 0.0,
    path_progress: float = 0.0,
    communication_mb: float = 0.0,
    remaining_deadline_s: float = 960.0,
) -> RuntimeState:
    return RuntimeState(
        current_time_s=time_s,
        frame_id=int(time_s),
        battery_frac=battery_frac,
        power_mode="15W",
        network=network,
        current_config_id=None,
        remaining_deadline_s=remaining_deadline_s,
        cumulative_energy_j=0.0,
        cumulative_communication_mb=communication_mb,
        path_progress=path_progress,
        evidence_summary=EvidenceSummary(
            predicted_unique_targets=0, processed_frames=0, mean_prediction_confidence=0.0
        ),
    )


@pytest.fixture
def rule_based(profiles) -> RuleBasedPolicy:
    return RuleBasedPolicy(public_profiles=profiles.public_view())


# --- static baselines ---------------------------------------------------------------------------


def test_a_static_policy_ignores_everything(catalog, contract) -> None:
    policy = StaticPolicy("CFG_LOCAL_STRONG")
    for network in (GOOD, DEGRADED, POOR, DISCONNECTED):
        assert (
            policy.select_config(contract, _state(network=network, battery_frac=0.01), catalog)
            == "CFG_LOCAL_STRONG"
        )


def test_a_static_policy_reports_honestly_when_it_does_not_apply(catalog, contract) -> None:
    """Substituting silently would report a different policy's score under this one's name."""
    policy = StaticPolicy("CFG_LOCAL_STRONG")
    restricted = catalog.subset(["CFG_LOCAL_LIGHT"])
    assert policy.select_config(contract, _state(), restricted) == "CFG_LOCAL_STRONG"


# --- latency estimation -------------------------------------------------------------------------


def test_a_policy_estimates_remote_latency_from_what_it_observes(catalog, profiles) -> None:
    view = profiles.public_view()
    remote = catalog.get("CFG_REMOTE_STRONG")
    fast = estimated_latency_s(remote, view.get(remote.config_id), GOOD)
    slow = estimated_latency_s(remote, view.get(remote.config_id), POOR)

    assert fast == pytest.approx(0.6 + 0.03 + 0.1)  # upload + RTT + compute
    assert slow == pytest.approx(6.0 + 0.15 + 0.1)
    assert slow > fast


def test_the_estimate_is_approximate_not_the_executors_number(catalog, profiles) -> None:
    """Download size is not disclosed, so a policy estimates rather than knows."""
    from missionbench.executor import ProfileExecutor
    from missionbench.executor.base import ExecutionRequest

    remote = catalog.get("CFG_REMOTE_STRONG")
    estimate = estimated_latency_s(remote, profiles.public_view().get(remote.config_id), GOOD)
    actual = (
        ProfileExecutor(profiles)
        .execute(
            ExecutionRequest(
                episode_id="E",
                frame_id=0,
                configuration=remote,
                network=GOOD,
                current_time_s=0.0,
                seed=1,
            )
        )
        .latency_s
    )

    assert estimate < actual, "the undisclosed download leg makes the estimate optimistic"
    assert estimate == pytest.approx(actual, rel=0.05), "but it is close enough to act on"


def test_local_latency_needs_no_network(catalog, profiles) -> None:
    local = catalog.get("CFG_LOCAL_STRONG")
    view = profiles.public_view()
    assert estimated_latency_s(local, view.get(local.config_id), DISCONNECTED) == pytest.approx(
        0.45
    )


def test_there_is_no_estimate_without_a_profile(catalog) -> None:
    assert estimated_latency_s(catalog.get("CFG_LOCAL_LIGHT"), None, GOOD) is None


# --- rule-based: quality preference -------------------------------------------------------------


def test_it_takes_the_best_quality_it_can_afford(rule_based, catalog, contract) -> None:
    assert rule_based.select_config(contract, _state(), catalog) == "CFG_REMOTE_STRONG"


def test_it_abandons_remote_as_bandwidth_collapses(rule_based, catalog, contract) -> None:
    """At 2 Mbps one upload spans six frames, and every skipped frame is a target unseen."""
    assert rule_based.select_config(contract, _state(network=GOOD), catalog) == "CFG_REMOTE_STRONG"
    assert rule_based.select_config(contract, _state(network=POOR), catalog) == "CFG_LOCAL_STRONG"


def test_the_latency_ceiling_is_tunable(profiles, catalog, contract) -> None:
    patient = RuleBasedPolicy(
        public_profiles=profiles.public_view(),
        settings=RuleBasedSettings(max_latency_intervals=10.0),
    )
    assert patient.select_config(contract, _state(network=POOR), catalog) == "CFG_REMOTE_STRONG"


# --- rule-based: reachability -------------------------------------------------------------------


def test_it_does_not_select_remote_into_a_dead_link(rule_based, catalog, contract) -> None:
    chosen = rule_based.select_config(contract, _state(network=DISCONNECTED), catalog)
    assert catalog.get(chosen).strategy.placement.value == "local"


# --- rule-based: privacy ------------------------------------------------------------------------


def test_it_respects_local_only(rule_based, catalog, local_only_contract) -> None:
    chosen = rule_based.select_config(local_only_contract, _state(), catalog)
    assert catalog.get(chosen).strategy.placement.value == "local"


def test_it_respects_features_only(rule_based, catalog, contract) -> None:
    features_only = dataclasses.replace(contract, privacy_level=PrivacyLevel.FEATURES_ONLY)
    chosen = rule_based.select_config(features_only, _state(), catalog)
    assert catalog.get(chosen).strategy.placement.value == "local", (
        "the shipped remote configuration transmits raw input"
    )


def test_an_impossible_pool_still_returns_an_action(rule_based, contract, catalog) -> None:
    """Every option violates privacy. Returning one lets the validator record the truth."""
    remote_only = catalog.subset(["CFG_REMOTE_STRONG"])
    local_only = dataclasses.replace(contract, privacy_level=PrivacyLevel.LOCAL_ONLY)
    assert rule_based.select_config(local_only, _state(), remote_only) == "CFG_REMOTE_STRONG"


# --- rule-based: communication budget -----------------------------------------------------------


def test_it_stops_using_remote_once_the_budget_is_spent(rule_based, catalog, contract) -> None:
    plenty = _state(communication_mb=0.0)
    nearly_gone = _state(communication_mb=contract.communication_budget_mb - 1.0)

    assert rule_based.select_config(contract, plenty, catalog) == "CFG_REMOTE_STRONG"
    assert (
        catalog.get(
            rule_based.select_config(contract, nearly_gone, catalog)
        ).strategy.placement.value
        == "local"
    )


def test_it_holds_back_a_reserve(profiles, catalog, contract) -> None:
    """Spending to the last megabyte would leave a late step with no legal remote option."""
    policy = RuleBasedPolicy(
        public_profiles=profiles.public_view(),
        settings=RuleBasedSettings(communication_reserve_frac=0.5),
    )
    half_spent = _state(communication_mb=contract.communication_budget_mb * 0.5)
    assert (
        catalog.get(policy.select_config(contract, half_spent, catalog)).strategy.placement.value
        == "local"
    )


def test_a_zero_budget_rules_out_remote_from_the_start(rule_based, catalog, contract) -> None:
    no_budget = dataclasses.replace(contract, communication_budget_mb=0.0)
    assert (
        catalog.get(rule_based.select_config(no_budget, _state(), catalog)).strategy.placement.value
        == "local"
    )


# --- rule-based: deadline and battery pressure --------------------------------------------------


def test_deadline_pressure_switches_to_the_fastest_option(rule_based, catalog, contract) -> None:
    """Projected from progress so far: half the deadline gone, a tenth of the path flown."""
    behind = _state(time_s=480.0, path_progress=0.10, remaining_deadline_s=480.0)
    assert rule_based.select_config(contract, behind, catalog) == "CFG_LOCAL_LIGHT"


def test_progress_on_schedule_does_not_trigger_the_fast_path(rule_based, catalog, contract) -> None:
    on_track = _state(time_s=480.0, path_progress=0.55, remaining_deadline_s=480.0)
    assert rule_based.select_config(contract, on_track, catalog) == "CFG_REMOTE_STRONG"


def test_battery_pressure_switches_to_the_fastest_option(rule_based, catalog, contract) -> None:
    """Latency is a weak proxy for energy -- public profiles disclose no energy at all."""
    low = _state(battery_frac=contract.min_final_battery_frac + 0.01)
    assert rule_based.select_config(contract, low, catalog) == "CFG_LOCAL_LIGHT"


def test_a_healthy_battery_does_not_trigger_the_fast_path(rule_based, catalog, contract) -> None:
    assert (
        rule_based.select_config(contract, _state(battery_frac=0.80), catalog) != "CFG_LOCAL_LIGHT"
    )


# --- rule-based: degradation and determinism ----------------------------------------------------


def test_it_still_runs_without_profiles(catalog, contract) -> None:
    """Whether profiles are disclosed is a property of the run, not a policy's assumption."""
    blind = RuleBasedPolicy(public_profiles=PublicProfileView.hidden())
    chosen = blind.select_config(contract, _state(), catalog)
    assert chosen in catalog


def test_a_blind_policy_still_avoids_a_dead_link(catalog, contract) -> None:
    """Reachability needs no profile, so the one rule that must survive does."""
    blind = RuleBasedPolicy()
    chosen = blind.select_config(contract, _state(network=DISCONNECTED), catalog)
    assert catalog.get(chosen).strategy.placement.value == "local"


def test_selection_is_deterministic(rule_based, catalog, contract) -> None:
    state = _state(network=DEGRADED, communication_mb=100.0)
    choices = {rule_based.select_config(contract, state, catalog) for _ in range(10)}
    assert len(choices) == 1


def test_a_policy_never_sees_more_than_the_episode_allows(rule_based, catalog, contract) -> None:
    restricted = catalog.subset(["CFG_LOCAL_LIGHT", "CFG_LOCAL_STRONG"])
    assert rule_based.select_config(contract, _state(), restricted) in restricted
    assert "CFG_REMOTE_STRONG" not in restricted


# --- the registry -------------------------------------------------------------------------------


def test_every_v1_baseline_is_registered() -> None:
    assert policy_registry.names() == (
        "always_local_light",
        "always_local_strong",
        "always_remote_strong",
        "budget_planner",
        "rule_based",
        "static",
    )


def test_baselines_build_from_the_registry(catalog, contract) -> None:
    assert (
        policy_registry.create("always_local_light").select_config(contract, _state(), catalog)
        == "CFG_LOCAL_LIGHT"
    )


def test_the_rule_based_policy_takes_its_profiles_from_the_composition_root(
    profiles, catalog, contract
) -> None:
    policy = policy_registry.create("rule_based", public_profiles=profiles.public_view())
    assert policy.select_config(contract, _state(), catalog) == "CFG_REMOTE_STRONG"


def test_the_static_policy_is_reusable_for_another_catalog(catalog, contract) -> None:
    policy = policy_registry.create("static", config_id="CFG_LOCAL_STRONG")
    assert policy.select_config(contract, _state(), catalog) == "CFG_LOCAL_STRONG"


def test_an_unknown_policy_says_what_is_available() -> None:
    with pytest.raises(RegistryError, match="unknown policy 'rulebased'; available: \\["):
        policy_registry.create("rulebased")


def test_a_new_policy_needs_no_change_to_anything_else(catalog, contract) -> None:
    """Extensibility criterion 2: conforming to the protocol is the whole requirement."""

    class LastInCatalogPolicy:
        def select_config(self, contract, state, configs) -> str:
            return configs.ids()[-1]

    policy: Policy = LastInCatalogPolicy()
    assert policy.select_config(contract, _state(), catalog) == "CFG_REMOTE_STRONG"
