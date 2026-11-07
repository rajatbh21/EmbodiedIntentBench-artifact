"""Per-schema loading, invariants, and cross-document checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from missionbench.schemas import (
    ComparisonOperator,
    ConfigCatalog,
    Placement,
    Precision,
    PrivacyLevel,
    SchemaValidationError,
    check_contract_is_supported,
    check_episode_power_mode,
    load_config_catalog,
    load_contract,
    load_episode,
    load_network_trace,
    load_platform_profile,
    load_task_spec,
)
from missionbench.schemas.configuration import Configuration, Strategy
from missionbench.schemas.platform import JOULES_PER_WATT_HOUR

# --- contract --------------------------------------------------------------------


def test_contract_fixture_loads(data_dir: Path) -> None:
    contract = load_contract(data_dir / "contracts" / "contract_001.json")
    assert contract.contract_id == "CONTRACT_001"
    assert contract.task_id == "HUMAN_SEARCH_SEGMENTATION"
    assert contract.quality_metric == "target_f1"
    assert contract.quality_operator is ComparisonOperator.GREATER_EQUAL
    assert contract.quality_threshold == 0.85
    assert contract.deadline_s == 960.0
    assert contract.communication_budget_mb == 400.0
    assert contract.min_final_battery_frac == 0.20
    assert contract.privacy_level is PrivacyLevel.REMOTE_ALLOWED


def test_contract_quality_satisfied_uses_its_own_operator(data_dir: Path) -> None:
    contract = load_contract(data_dir / "contracts" / "contract_001.json")
    threshold = contract.quality_threshold
    assert contract.quality_satisfied(threshold + 0.04)
    assert contract.quality_satisfied(threshold), "the boundary passes for >="
    assert not contract.quality_satisfied(threshold - 0.01)


def test_contract_is_frozen(data_dir: Path) -> None:
    contract = load_contract(data_dir / "contracts" / "contract_001.json")
    with pytest.raises(AttributeError):
        contract.deadline_s = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("deadline_s", 0.0, "must be > 0.0"),
        ("deadline_s", -1.0, "must be > 0.0"),
        ("communication_budget_mb", -0.1, "must be >= 0.0"),
        ("min_final_battery_frac", 1.5, "must be <= 1.0"),
        ("min_final_battery_frac", -0.1, "must be >= 0.0"),
        ("quality_operator", "=>", "must be one of"),
        ("privacy_level", "public", "must be one of"),
    ],
)
def test_contract_rejects_out_of_range_values(
    write_json, valid_contract_payload, field: str, value: object, message: str
) -> None:
    valid_contract_payload[field] = value
    with pytest.raises(SchemaValidationError, match=message):
        load_contract(write_json(valid_contract_payload))


def test_all_privacy_levels_are_loadable(write_json, valid_contract_payload) -> None:
    for level in PrivacyLevel:
        valid_contract_payload["privacy_level"] = level.value
        assert load_contract(write_json(valid_contract_payload)).privacy_level is level


# --- task specification -----------------------------------------------------------


def test_task_spec_fixture_loads(data_dir: Path) -> None:
    spec = load_task_spec(data_dir / "task_specs" / "human_search_segmentation.json")
    assert spec.task_id == "HUMAN_SEARCH_SEGMENTATION"
    assert spec.target_type == "person"
    assert spec.matching_rule.metric == "mask_iou"
    assert spec.matching_rule.threshold == 0.50
    assert spec.deduplication.method == "ground_truth_track_id"
    assert spec.supported_quality_metrics == ("target_recall", "target_precision", "target_f1")


def test_matching_rule_decides_a_match_at_the_threshold(data_dir: Path) -> None:
    rule = load_task_spec(data_dir / "task_specs" / "human_search_segmentation.json").matching_rule
    assert rule.matches(0.50)
    assert rule.matches(0.51)
    assert not rule.matches(0.49)


def test_contract_and_task_spec_consistency_is_checked(data_dir: Path) -> None:
    spec = load_task_spec(data_dir / "task_specs" / "human_search_segmentation.json")
    contract = load_contract(data_dir / "contracts" / "contract_001.json")

    check_contract_is_supported(
        spec, task_id=contract.task_id, quality_metric=contract.quality_metric
    )

    with pytest.raises(SchemaValidationError, match="does not support quality metric"):
        check_contract_is_supported(spec, task_id=spec.task_id, quality_metric="mIoU")
    with pytest.raises(SchemaValidationError, match="contract targets task"):
        check_contract_is_supported(spec, task_id="OTHER_TASK", quality_metric="target_f1")


# --- configuration catalog --------------------------------------------------------


def test_config_catalog_fixture_loads(data_dir: Path) -> None:
    catalog = load_config_catalog(data_dir / "configs" / "config_catalog_001.json")
    assert catalog.ids() == ("CFG_LOCAL_LIGHT", "CFG_LOCAL_STRONG", "CFG_REMOTE_STRONG")

    light = catalog.get("CFG_LOCAL_LIGHT")
    assert light.model_id == "LIGHT_INSTANCE_SEG"
    assert light.strategy.placement is Placement.LOCAL
    assert light.strategy.precision is Precision.INT8
    assert light.strategy.input_compression is None
    assert not light.strategy.is_remote

    remote = catalog.get("CFG_REMOTE_STRONG")
    assert remote.strategy.placement is Placement.REMOTE
    assert remote.strategy.input_compression == "jpeg_q75"
    assert remote.strategy.is_remote


def test_catalog_iteration_order_follows_declaration(data_dir: Path) -> None:
    """Deterministic iteration: a scanning policy must not depend on dict luck."""
    catalog = load_config_catalog(data_dir / "configs" / "config_catalog_001.json")
    assert [config.config_id for config in catalog] == list(catalog.ids())


def test_catalog_membership_and_unknown_lookup(data_dir: Path) -> None:
    catalog = load_config_catalog(data_dir / "configs" / "config_catalog_001.json")
    assert "CFG_LOCAL_LIGHT" in catalog
    assert "CFG_NOT_REAL" not in catalog
    assert len(catalog) == 3
    with pytest.raises(KeyError, match="unknown config_id"):
        catalog.get("CFG_NOT_REAL")


def test_catalog_subset_restricts_the_visible_pool(data_dir: Path) -> None:
    catalog = load_config_catalog(data_dir / "configs" / "config_catalog_001.json")
    subset = catalog.subset(["CFG_REMOTE_STRONG", "CFG_LOCAL_LIGHT"])
    # Order follows the parent catalog, not the request, so the pool is request-independent.
    assert subset.ids() == ("CFG_LOCAL_LIGHT", "CFG_REMOTE_STRONG")
    assert "CFG_LOCAL_STRONG" not in subset
    with pytest.raises(SchemaValidationError, match="unknown config_id"):
        catalog.subset(["CFG_NOT_REAL"])


def test_catalog_rejects_duplicate_config_ids() -> None:
    strategy = Strategy(placement=Placement.LOCAL, precision=Precision.INT8)
    duplicate = Configuration(config_id="CFG_A", model_id="M", strategy=strategy)
    with pytest.raises(SchemaValidationError, match="duplicate config_id"):
        ConfigCatalog([duplicate, duplicate])


def test_strategy_parameters_are_read_only() -> None:
    """Configurations are policy-visible; a policy must not be able to edit the catalog."""
    strategy = Strategy(
        placement=Placement.LOCAL, precision=Precision.INT8, parameters={"pruning_ratio": 0.5}
    )
    assert strategy.parameters["pruning_ratio"] == 0.5
    with pytest.raises(TypeError):
        strategy.parameters["pruning_ratio"] = 0.9  # type: ignore[index]


def test_strategy_parameters_default_to_empty_and_accept_future_dimensions(
    write_json,
) -> None:
    payload = {
        "schema_version": "1.0",
        "configs": [
            {
                "config_id": "CFG_A",
                "model_id": "M",
                "strategy": {"placement": "local", "precision": "fp32"},
            },
            {
                "config_id": "CFG_B",
                "model_id": "M",
                "strategy": {
                    "placement": "remote",
                    "precision": "fp16",
                    "parameters": {"split_point": 7, "input_resolution": "640x480"},
                },
            },
        ],
    }
    catalog = load_config_catalog(write_json(payload))
    assert catalog.get("CFG_A").strategy.parameters == {}
    # A new strategy dimension needs no schema change: it rides in `parameters`.
    assert catalog.get("CFG_B").strategy.parameters["split_point"] == 7


@pytest.mark.parametrize(
    ("placement", "precision"),
    [("sideways", "fp16"), ("local", "fp8")],
)
def test_catalog_rejects_unknown_strategy_enums(write_json, placement: str, precision: str) -> None:
    payload = {
        "schema_version": "1.0",
        "configs": [
            {
                "config_id": "CFG_A",
                "model_id": "M",
                "strategy": {"placement": placement, "precision": precision},
            }
        ],
    }
    with pytest.raises(SchemaValidationError, match="must be one of"):
        load_config_catalog(write_json(payload))


# --- platform ---------------------------------------------------------------------


def test_platform_fixture_loads_and_converts_capacity(data_dir: Path) -> None:
    platform = load_platform_profile(data_dir / "platforms" / "synthetic_uav_platform_001.json")
    assert platform.platform_id == "UAV_PLATFORM_001"
    assert platform.battery_capacity_wh == 100.0
    assert platform.battery_capacity_j == 100.0 * JOULES_PER_WATT_HOUR == 360_000.0
    assert platform.flight_power_w == 180.0
    assert platform.communication_energy_j_per_mb == 0.5
    assert platform.supported_power_modes == ("15W",)


def test_platform_power_mode_check(data_dir: Path) -> None:
    platform = load_platform_profile(data_dir / "platforms" / "synthetic_uav_platform_001.json")
    check_episode_power_mode(platform, power_mode="15W")
    with pytest.raises(SchemaValidationError, match="does not support power mode"):
        check_episode_power_mode(platform, power_mode="30W")


# --- network trace ----------------------------------------------------------------


def test_network_trace_fixture_loads(data_dir: Path) -> None:
    trace = load_network_trace(data_dir / "network_traces" / "synthetic_network_degrading_001.json")
    assert trace.trace_id == "NETWORK_DEGRADING_001"
    assert len(trace.segments) == 3
    assert (trace.start_s, trace.end_s) == (0.0, 960.0)
    assert trace.segments[0].observation.bandwidth_mbps == 20.0
    assert trace.segments[2].observation.rtt_ms == 150.0


def test_trace_segments_are_half_open_so_boundaries_are_unambiguous(
    write_json, valid_trace_payload
) -> None:
    trace = load_network_trace(write_json(valid_trace_payload))
    first, second = trace.segments
    assert first.contains(0.0) and first.contains(9.999)
    assert not first.contains(10.0), "the boundary belongs to exactly one segment"
    assert second.contains(10.0)
    assert not second.contains(20.0), "the trace end is exclusive"


def test_trace_with_a_gap_is_rejected(write_json, valid_trace_payload) -> None:
    valid_trace_payload["segments"][1]["start_s"] = 12
    with pytest.raises(SchemaValidationError, match="gap between segment 0"):
        load_network_trace(write_json(valid_trace_payload))


def test_trace_with_an_overlap_is_rejected(write_json, valid_trace_payload) -> None:
    valid_trace_payload["segments"][1]["start_s"] = 8
    with pytest.raises(SchemaValidationError, match="overlap between segment 0"):
        load_network_trace(write_json(valid_trace_payload))


def test_trace_segment_must_have_positive_duration(write_json, valid_trace_payload) -> None:
    valid_trace_payload["segments"][0]["end_s"] = 0
    with pytest.raises(SchemaValidationError, match="must be > 0"):
        load_network_trace(write_json(valid_trace_payload))


def test_disconnected_observation_is_recognised(data_dir: Path) -> None:
    trace = load_network_trace(
        data_dir / "network_traces" / "synthetic_network_disconnecting_001.json"
    )
    assert trace.segments[1].observation.is_disconnected
    assert not trace.segments[0].observation.is_disconnected


# --- episode ----------------------------------------------------------------------


def test_episode_fixture_loads(data_dir: Path) -> None:
    episode = load_episode(data_dir / "episodes" / "episode_001.json")
    assert episode.episode_id == "EPISODE_001"
    assert episode.platform_id == "UAV_PLATFORM_001"
    assert episode.network_trace_id == "NETWORK_DEGRADING_001"
    assert episode.initial_battery_frac == 0.80
    assert episode.initial_config_id is None
    assert episode.seed == 42
    assert episode.allowed_config_ids == (
        "CFG_LOCAL_LIGHT",
        "CFG_LOCAL_STRONG",
        "CFG_REMOTE_STRONG",
    )


def test_episode_initial_config_must_be_allowed(data_dir: Path, write_json) -> None:
    payload = {
        "schema_version": "1.0",
        "episode_id": "E",
        "platform_id": "P",
        "path_id": "PATH",
        "frame_stream_id": "S",
        "initial_altitude_m": 40.0,
        "velocity_mps": 5.0,
        "initial_battery_frac": 0.8,
        "power_mode": "15W",
        "network_trace_id": "T",
        "allowed_config_ids": ["CFG_A", "CFG_B"],
        "initial_config_id": "CFG_C",
        "seed": 1,
    }
    with pytest.raises(SchemaValidationError, match="is not in allowed_config_ids"):
        load_episode(write_json(payload))

    payload["initial_config_id"] = "CFG_B"
    assert load_episode(write_json(payload)).initial_config_id == "CFG_B"
