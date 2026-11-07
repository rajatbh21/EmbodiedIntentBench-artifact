"""Shared test fixtures.

Tests read the real files under ``data/`` rather than inventing parallel fixtures, so a
fixture that drifts out of conformance fails the suite instead of quietly diverging from
what the benchmark actually ships.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """The repository's benchmark data directory."""
    assert DATA_DIR.is_dir(), f"expected benchmark data at {DATA_DIR}"
    return DATA_DIR


@pytest.fixture
def write_json(tmp_path: Path) -> Callable[[dict[str, Any]], Path]:
    """Return a helper that writes a payload to a temporary JSON file and returns its path.

    Used to exercise malformed documents without keeping invalid files in ``data/``.
    """
    counter = 0

    def _write(payload: Any, *, name: str | None = None) -> Path:
        nonlocal counter
        counter += 1
        path = tmp_path / (name or f"document_{counter}.json")
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    return _write


@pytest.fixture
def contract(data_dir: Path):
    """The default example contract: target_f1 >= 0.80, 60 s, 50 MB, 20 %, remote_allowed."""
    from missionbench.schemas import load_contract

    return load_contract(data_dir / "contracts" / "contract_001.json")


@pytest.fixture
def local_only_contract(data_dir: Path):
    from missionbench.schemas import load_contract

    return load_contract(data_dir / "contracts" / "contract_002_local_only.json")


@pytest.fixture
def platform(data_dir: Path):
    """100 Wh capacity, 180 W flight power, 0.5 J/MB communication."""
    from missionbench.schemas import load_platform_profile

    return load_platform_profile(data_dir / "platforms" / "synthetic_uav_platform_001.json")


@pytest.fixture
def catalog(data_dir: Path):
    """The three-configuration V1 catalog."""
    from missionbench.schemas import load_config_catalog

    return load_config_catalog(data_dir / "configs" / "config_catalog_001.json")


@pytest.fixture
def profiles(data_dir: Path):
    """Synthetic per-configuration costs for UAV_PLATFORM_001."""
    from missionbench.schemas import load_profile_catalog

    return load_profile_catalog(data_dir / "profiles" / "synthetic_profiles_uav_platform_001.json")


@pytest.fixture
def episode(data_dir: Path):
    """EPISODE_001: 80 % battery, degrading network, no initial configuration."""
    from missionbench.schemas import load_episode

    return load_episode(data_dir / "episodes" / "episode_001.json")


@pytest.fixture
def path_spec(data_dir: Path):
    """PATH_001: 250 m, i.e. 50 s at the episode velocity of 5 m/s."""
    from missionbench.schemas import load_path_spec

    return load_path_spec(data_dir / "paths" / "path_001.json")


# ---------------------------------------------------------------------------
# Behaviour-test scale
#
# Tests of *behaviour* -- transitions, termination, timing -- use these round
# numbers rather than the shipped fixtures. The shipped mission is 900 s over
# 4500 m against a 960 s deadline, which turns every expected value into an
# awkward fraction and, worse, means rescaling the benchmark rewrites tests that
# have nothing to do with scale. A 50 s path against a 60 s deadline keeps the
# arithmetic checkable by eye.
#
# That the *shipped* fixtures have the intended scale is asserted separately, in
# test_fixture_integrity.py.
# ---------------------------------------------------------------------------

BEHAVIOUR_PATH_LENGTH_M = 250.0
BEHAVIOUR_VELOCITY_MPS = 5.0  # => a 50 s path
BEHAVIOUR_DEADLINE_S = 60.0


@pytest.fixture
def synthetic_contract(contract):
    """The example contract at behaviour-test scale: a 60 s deadline."""
    import dataclasses

    return dataclasses.replace(contract, deadline_s=BEHAVIOUR_DEADLINE_S)


@pytest.fixture
def make_state_manager(episode, platform):
    """Factory for state managers at behaviour-test scale, with overridable parts."""
    import dataclasses

    from missionbench.simulator.battery_model import SimpleBatteryModel
    from missionbench.simulator.path import ConstantVelocityPath
    from missionbench.simulator.state_manager import StateManager

    def _make(
        *,
        path_length_m: float = BEHAVIOUR_PATH_LENGTH_M,
        velocity_mps: float = BEHAVIOUR_VELOCITY_MPS,
        battery_model: Any = None,
        **episode_overrides: Any,
    ):
        return StateManager(
            episode=dataclasses.replace(episode, **episode_overrides),
            platform=platform,
            path=ConstantVelocityPath(length_m=path_length_m, velocity_mps=velocity_mps),
            battery_model=battery_model if battery_model is not None else SimpleBatteryModel(),
        )

    return _make


@pytest.fixture
def state_manager(make_state_manager):
    """A state manager at behaviour-test scale, with the simple battery model."""
    return make_state_manager()


@pytest.fixture
def valid_contract_payload() -> dict[str, Any]:
    """A minimal conforming contract document, for tests that mutate one field at a time."""
    return {
        "schema_version": "1.0",
        "contract_id": "CONTRACT_TEST",
        "task_id": "HUMAN_SEARCH_SEGMENTATION",
        "evidence_type": "instance_mask_set",
        "quality_metric": "target_f1",
        "quality_operator": ">=",
        "quality_threshold": 0.80,
        "deadline_s": 60.0,
        "communication_budget_mb": 50.0,
        "min_final_battery_frac": 0.20,
        "privacy_level": "remote_allowed",
    }


@pytest.fixture
def valid_trace_payload() -> dict[str, Any]:
    """A minimal conforming two-segment network trace document."""
    return {
        "schema_version": "1.0",
        "trace_id": "NETWORK_TEST",
        "segments": [
            {
                "start_s": 0,
                "end_s": 10,
                "bandwidth_mbps": 20.0,
                "rtt_ms": 30.0,
                "packet_loss_frac": 0.0,
            },
            {
                "start_s": 10,
                "end_s": 20,
                "bandwidth_mbps": 5.0,
                "rtt_ms": 90.0,
                "packet_loss_frac": 0.02,
            },
        ],
    }
