"""Every shipped fixture loads, and cross-file references resolve.

Loaders validate one document at a time; nothing in a single file can catch an episode
naming a network trace that does not exist. These tests close that gap, so a fixture set
cannot drift into referencing something absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from missionbench import SCHEMA_VERSION
from missionbench.executor import load_replay_records
from missionbench.schemas import (
    load_config_catalog,
    load_contract,
    load_episode,
    load_network_trace,
    load_path_spec,
    load_platform_profile,
    load_profile_catalog,
    load_task_spec,
)
from missionbench.tasks.human_search_segmentation import load_human_search_ground_truth

LOADERS_BY_DIRECTORY = {
    "contracts": load_contract,
    "episodes": load_episode,
    "configs": load_config_catalog,
    "platforms": load_platform_profile,
    "task_specs": load_task_spec,
    "network_traces": load_network_trace,
    "paths": load_path_spec,
    "profiles": load_profile_catalog,
    "predictions": load_replay_records,
    "ground_truth": load_human_search_ground_truth,
}


def _fixture_paths(data_dir: Path, directory: str) -> list[Path]:
    return sorted((data_dir / directory).glob("*.json"))


@pytest.mark.parametrize("directory", sorted(LOADERS_BY_DIRECTORY))
def test_every_fixture_in_the_directory_loads(data_dir: Path, directory: str) -> None:
    paths = _fixture_paths(data_dir, directory)
    assert paths, f"expected at least one fixture in data/{directory}"
    for path in paths:
        LOADERS_BY_DIRECTORY[directory](path)


@pytest.mark.parametrize("directory", sorted(LOADERS_BY_DIRECTORY))
def test_every_fixture_declares_the_supported_schema_version(
    data_dir: Path, directory: str
) -> None:
    for path in _fixture_paths(data_dir, directory):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload.get("schema_version") == SCHEMA_VERSION, path


def test_the_three_required_network_conditions_are_present(data_dir: Path) -> None:
    trace_ids = {
        load_network_trace(path).trace_id for path in _fixture_paths(data_dir, "network_traces")
    }
    assert trace_ids == {
        "NETWORK_STABLE_001",
        "NETWORK_DEGRADING_001",
        "NETWORK_DISCONNECTING_001",
    }


def test_episode_references_resolve(data_dir: Path) -> None:
    platform_ids = {
        load_platform_profile(path).platform_id for path in _fixture_paths(data_dir, "platforms")
    }
    trace_ids = {
        load_network_trace(path).trace_id for path in _fixture_paths(data_dir, "network_traces")
    }
    path_ids = {load_path_spec(path).path_id for path in _fixture_paths(data_dir, "paths")}
    known_config_ids: set[str] = set()
    for path in _fixture_paths(data_dir, "configs"):
        known_config_ids.update(load_config_catalog(path).ids())

    for path in _fixture_paths(data_dir, "episodes"):
        episode = load_episode(path)
        assert episode.platform_id in platform_ids, path
        assert episode.network_trace_id in trace_ids, path
        assert episode.path_id in path_ids, path
        assert set(episode.allowed_config_ids) <= known_config_ids, path


def test_contract_references_resolve(data_dir: Path) -> None:
    specs = {
        spec.task_id: spec
        for spec in (load_task_spec(path) for path in _fixture_paths(data_dir, "task_specs"))
    }
    for path in _fixture_paths(data_dir, "contracts"):
        contract = load_contract(path)
        assert contract.task_id in specs, path
        spec = specs[contract.task_id]
        assert spec.supports_metric(contract.quality_metric), path
        assert contract.evidence_type == spec.evidence_type, path


def test_every_network_trace_covers_the_longest_contract_deadline(data_dir: Path) -> None:
    """A trace that ends before the deadline would make late observations undefined."""
    longest_deadline = max(
        load_contract(path).deadline_s for path in _fixture_paths(data_dir, "contracts")
    )
    for path in _fixture_paths(data_dir, "network_traces"):
        trace = load_network_trace(path)
        assert trace.start_s == 0.0, path
        assert trace.end_s >= longest_deadline, path


def test_the_path_can_be_flown_within_every_deadline(data_dir: Path) -> None:
    """Path completion must be a reachable outcome, not one the deadline pre-empts."""
    velocities = {load_episode(path).velocity_mps for path in _fixture_paths(data_dir, "episodes")}
    deadlines = [load_contract(path).deadline_s for path in _fixture_paths(data_dir, "contracts")]

    for path in _fixture_paths(data_dir, "paths"):
        spec = load_path_spec(path)
        for velocity in velocities:
            duration_s = spec.length_m / velocity
            assert duration_s <= min(deadlines), (
                f"{path} takes {duration_s:.0f} s at {velocity} m/s, past the tightest "
                f"deadline of {min(deadlines):.0f} s"
            )


def test_no_episode_is_doomed_on_battery_by_flight_alone(data_dir: Path) -> None:
    """Flight energy is not policy-controllable, so it must never decide the outcome.

    If merely flying the path put an episode under a contract's reserve, that episode
    would fail no matter what the policy selected, and would measure nothing.
    """
    platforms = {
        profile.platform_id: profile
        for profile in (
            load_platform_profile(path) for path in _fixture_paths(data_dir, "platforms")
        )
    }
    paths = {
        spec.path_id: spec
        for spec in (load_path_spec(p) for p in _fixture_paths(data_dir, "paths"))
    }
    floors = [
        load_contract(path).min_final_battery_frac for path in _fixture_paths(data_dir, "contracts")
    ]

    for path in _fixture_paths(data_dir, "episodes"):
        episode = load_episode(path)
        platform = platforms[episode.platform_id]
        duration_s = paths[episode.path_id].length_m / episode.velocity_mps
        flight_frac = platform.flight_power_w * duration_s / platform.battery_capacity_j
        remaining = episode.initial_battery_frac - flight_frac
        assert remaining > max(floors), (
            f"{path}: flight alone leaves {remaining:.3f}, at or below the strictest "
            f"reserve {max(floors):.2f}; the episode could not be passed by any policy"
        )


def test_the_mission_is_long_enough_for_battery_to_be_an_observation(data_dir: Path) -> None:
    """The battery must traverse a real span, or battery-aware policy logic is dead code.

    At the original 50 s mission scale the battery moved by 0.025, so a rule such as
    "below 30 %, switch to the light configuration" could never fire and the benchmark
    could not tell a battery-aware policy from a battery-blind one. See docs/v1_spec.md §11
    ("Mission scale").
    """
    minimum_span = 0.30

    platforms = {
        profile.platform_id: profile
        for profile in (
            load_platform_profile(path) for path in _fixture_paths(data_dir, "platforms")
        )
    }
    paths = {
        spec.path_id: spec
        for spec in (load_path_spec(p) for p in _fixture_paths(data_dir, "paths"))
    }

    for path in _fixture_paths(data_dir, "episodes"):
        episode = load_episode(path)
        platform = platforms[episode.platform_id]
        duration_s = paths[episode.path_id].length_m / episode.velocity_mps
        span = platform.flight_power_w * duration_s / platform.battery_capacity_j
        assert span >= minimum_span, (
            f"{path}: the battery only moves {span:.3f} over the mission; "
            f"battery-conditioned behaviour would be untestable"
        )


def test_every_selectable_configuration_is_profiled(data_dir: Path) -> None:
    """A missing profile would surface on whichever step first selected that configuration."""
    catalogs = [load_config_catalog(path) for path in _fixture_paths(data_dir, "configs")]
    known_config_ids = {config_id for catalog in catalogs for config_id in catalog.ids()}

    for path in _fixture_paths(data_dir, "profiles"):
        profiles = load_profile_catalog(path)
        assert set(profiles) <= known_config_ids, f"{path} profiles an unknown configuration"

    platform_ids = {
        load_platform_profile(path).platform_id for path in _fixture_paths(data_dir, "platforms")
    }
    for path in _fixture_paths(data_dir, "profiles"):
        catalog = load_profile_catalog(path)
        assert catalog.platform_id in platform_ids, path
        assert set(catalog) == known_config_ids, (
            f"{path} does not profile every selectable configuration; "
            f"missing {sorted(known_config_ids - set(catalog))}"
        )


def test_every_episode_has_ground_truth_for_its_frame_stream(data_dir: Path) -> None:
    """Ground truth is keyed on the stream, so episodes flying the same scene share it."""
    streams = {
        load_human_search_ground_truth(path).frame_stream_id
        for path in _fixture_paths(data_dir, "ground_truth")
    }
    for path in _fixture_paths(data_dir, "episodes"):
        assert load_episode(path).frame_stream_id in streams, path


def test_ground_truth_targets_fall_inside_the_mission(data_dir: Path) -> None:
    """A target visible only after the path ends could never be found by any policy."""
    paths = {
        spec.path_id: spec
        for spec in (load_path_spec(p) for p in _fixture_paths(data_dir, "paths"))
    }
    episodes = [load_episode(path) for path in _fixture_paths(data_dir, "episodes")]
    shortest_mission_frames = min(
        int(paths[episode.path_id].length_m / episode.velocity_mps) for episode in episodes
    )

    for path in _fixture_paths(data_dir, "ground_truth"):
        for target in load_human_search_ground_truth(path).targets:
            assert target.last_frame_id < shortest_mission_frames, (
                f"{path}: {target.track_id} is only visible after the mission ends"
            )


def test_measurement_bearing_fixtures_are_marked_synthetic(data_dir: Path) -> None:
    """Profiles, traces, predictions, and ground truth carry invented numbers.

    Their filenames say so, so that a value lifted out of this repository cannot be
    mistaken for a measurement. Specification files (contracts, episodes, task specs,
    catalogs) state choices rather than measurements and are exempt.
    """
    for directory in ("platforms", "network_traces", "profiles", "predictions", "ground_truth"):
        for path in _fixture_paths(data_dir, directory):
            assert path.name.startswith("synthetic_"), (
                f"{path} carries invented numbers and must be named synthetic_*"
            )
