"""Trace-based network observation lookup."""

from __future__ import annotations

from pathlib import Path

import pytest

from missionbench.schemas.network_trace import (
    NetworkObservation,
    NetworkTrace,
    NetworkTraceSegment,
    load_network_trace,
)
from missionbench.simulator.network_trace import TraceBasedNetworkModel


@pytest.fixture
def degrading_model(data_dir: Path) -> TraceBasedNetworkModel:
    return TraceBasedNetworkModel(
        load_network_trace(data_dir / "network_traces" / "synthetic_network_degrading_001.json")
    )


@pytest.mark.parametrize(
    ("time_s", "expected_bandwidth"),
    [
        (0.0, 20.0),
        (160.0, 20.0),
        (319.999, 20.0),
        (320.0, 8.0),  # boundary belongs to the segment that starts there
        (639.999, 8.0),
        (640.0, 2.0),
        (959.999, 2.0),
    ],
)
def test_observation_follows_the_segment_in_force(
    degrading_model: TraceBasedNetworkModel, time_s: float, expected_bandwidth: float
) -> None:
    assert degrading_model.observe(time_s).bandwidth_mbps == expected_bandwidth


def test_lookup_returns_the_whole_observation(degrading_model: TraceBasedNetworkModel) -> None:
    observation = degrading_model.observe(400.0)
    assert observation == NetworkObservation(bandwidth_mbps=8.0, rtt_ms=70.0, packet_loss_frac=0.01)


def test_times_past_the_trace_end_clamp_to_the_last_segment(
    degrading_model: TraceBasedNetworkModel,
) -> None:
    """A slow inference can push the clock past the trace; that must not abort the run."""
    assert degrading_model.observe(960.0) == degrading_model.observe(959.9)
    assert degrading_model.observe(10_000.0).bandwidth_mbps == 2.0


def test_negative_times_clamp_to_the_first_segment(
    degrading_model: TraceBasedNetworkModel,
) -> None:
    assert degrading_model.observe(-1.0).bandwidth_mbps == 20.0


def test_lookup_is_deterministic(degrading_model: TraceBasedNetworkModel) -> None:
    assert [degrading_model.observe(400.0) for _ in range(5)].count(
        degrading_model.observe(400.0)
    ) == 5


def test_single_segment_trace(data_dir: Path) -> None:
    model = TraceBasedNetworkModel(
        load_network_trace(data_dir / "network_traces" / "synthetic_network_stable_001.json")
    )
    assert model.observe(0.0).bandwidth_mbps == 20.0
    assert model.observe(959.0).bandwidth_mbps == 20.0
    assert model.change_times_s() == ()


def test_disconnection_is_observable(data_dir: Path) -> None:
    model = TraceBasedNetworkModel(
        load_network_trace(data_dir / "network_traces" / "synthetic_network_disconnecting_001.json")
    )
    assert not model.observe(399.0).is_disconnected
    assert model.observe(400.0).is_disconnected
    assert model.observe(719.0).is_disconnected
    assert not model.observe(720.0).is_disconnected


def test_change_times_record_transitions_for_later_adaptation_analysis(
    degrading_model: TraceBasedNetworkModel,
) -> None:
    assert degrading_model.change_times_s() == (320.0, 640.0)


def test_change_times_ignore_segments_with_identical_conditions() -> None:
    """A trace split for authoring convenience is not a network change."""
    observation = NetworkObservation(bandwidth_mbps=10.0, rtt_ms=40.0, packet_loss_frac=0.0)
    trace = NetworkTrace(
        trace_id="T",
        segments=(
            NetworkTraceSegment(start_s=0.0, end_s=10.0, observation=observation),
            NetworkTraceSegment(start_s=10.0, end_s=20.0, observation=observation),
        ),
    )
    assert TraceBasedNetworkModel(trace).change_times_s() == ()


def test_model_exposes_its_trace_id(degrading_model: TraceBasedNetworkModel) -> None:
    assert degrading_model.trace_id == "NETWORK_DEGRADING_001"
