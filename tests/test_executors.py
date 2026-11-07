"""Execution backends: the cost model, failure handling, replay, and the registry.

Latency figures are worked by hand from the formula in docs/v1_spec.md §7 rather than from
the implementation, because the remote latency model is the mechanism that turns a network
condition into a mission cost -- the single most important number this layer produces.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from missionbench.executor import (
    DEFAULT_REMOTE_TIMEOUT_S,
    ExecutionRequest,
    ExecutionResult,
    FailureReason,
    ProfileExecutor,
    RealSegmentationExecutor,
    ReplayExecutor,
    executor_registry,
    load_replay_records,
    remote_latency_s,
    transfer_time_s,
)
from missionbench.registry import Registry, RegistryError
from missionbench.schemas import NetworkObservation, SchemaValidationError

GOOD = NetworkObservation(bandwidth_mbps=20.0, rtt_ms=30.0, packet_loss_frac=0.0)
DEGRADED = NetworkObservation(bandwidth_mbps=8.0, rtt_ms=70.0, packet_loss_frac=0.01)
POOR = NetworkObservation(bandwidth_mbps=2.0, rtt_ms=150.0, packet_loss_frac=0.03)
DISCONNECTED = NetworkObservation(bandwidth_mbps=0.0, rtt_ms=0.0, packet_loss_frac=1.0)


@pytest.fixture
def request_for(catalog):
    def _make(config_id: str, network: NetworkObservation = GOOD, *, frame_id: int = 0):
        return ExecutionRequest(
            episode_id="EPISODE_001",
            frame_id=frame_id,
            configuration=catalog.get(config_id),
            network=network,
            current_time_s=float(frame_id),
            seed=42,
        )

    return _make


# --- latency arithmetic ----------------------------------------------------------------


def test_transfer_time_converts_megabytes_to_megabits() -> None:
    assert transfer_time_s(1.0, 8.0) == 1.0  # 8 megabits over an 8 Mbps link
    assert transfer_time_s(1.5, 20.0) == 0.6


def test_remote_latency_sums_its_four_parts() -> None:
    latency = remote_latency_s(
        upload_mb=1.5, download_mb=0.05, server_latency_ms=100.0, bandwidth_mbps=20.0, rtt_ms=30.0
    )
    # 0.6 upload + 0.03 RTT + 0.1 server + 0.02 download
    assert latency == pytest.approx(0.75)


# --- ProfileExecutor: local -------------------------------------------------------------


def test_local_execution_uses_the_profiled_compute_time(profiles, request_for) -> None:
    result = ProfileExecutor(profiles).execute(request_for("CFG_LOCAL_LIGHT"))
    assert result.success
    assert result.latency_s == pytest.approx(0.100)
    assert result.onboard_energy_j == 3.0
    assert result.communication_mb == 0.0


def test_local_execution_ignores_the_network(profiles, request_for) -> None:
    executor = ProfileExecutor(profiles)
    latencies = [
        executor.execute(request_for("CFG_LOCAL_STRONG", network)).latency_s
        for network in (GOOD, DEGRADED, POOR, DISCONNECTED)
    ]
    assert latencies == pytest.approx([0.450] * 4)


def test_local_execution_still_works_while_disconnected(profiles, request_for) -> None:
    """The point of a local configuration: it is the option that survives a link loss."""
    result = ProfileExecutor(profiles).execute(request_for("CFG_LOCAL_LIGHT", DISCONNECTED))
    assert result.success


# --- ProfileExecutor: remote ------------------------------------------------------------


@pytest.mark.parametrize(
    ("network", "expected_latency_s"),
    [
        # upload 8*1.5/bw  +  rtt  +  0.1 server  +  download 8*0.05/bw
        (GOOD, 0.60 + 0.030 + 0.1 + 0.020),  # 20 Mbps,  30 ms => 0.75 s
        (DEGRADED, 1.50 + 0.070 + 0.1 + 0.050),  #  8 Mbps,  70 ms => 1.72 s
        (POOR, 6.00 + 0.150 + 0.1 + 0.200),  #  2 Mbps, 150 ms => 6.45 s
    ],
)
def test_remote_latency_tracks_the_observed_network(
    profiles, request_for, network: NetworkObservation, expected_latency_s: float
) -> None:
    result = ProfileExecutor(profiles).execute(request_for("CFG_REMOTE_STRONG", network))
    assert result.success
    assert result.latency_s == pytest.approx(expected_latency_s)


def test_a_degraded_link_pushes_remote_inference_past_the_decision_interval(
    profiles, request_for
) -> None:
    """This is the mechanism the benchmark measures: the network becomes a mission cost."""
    executor = ProfileExecutor(profiles)
    assert executor.execute(request_for("CFG_REMOTE_STRONG", GOOD)).latency_s < 1.0
    assert executor.execute(request_for("CFG_REMOTE_STRONG", DEGRADED)).latency_s > 1.0
    assert executor.execute(request_for("CFG_REMOTE_STRONG", POOR)).latency_s > 6.0


def test_remote_execution_counts_both_directions(profiles, request_for) -> None:
    result = ProfileExecutor(profiles).execute(request_for("CFG_REMOTE_STRONG"))
    assert (result.upload_mb, result.download_mb) == (1.5, 0.05)
    assert result.communication_mb == pytest.approx(1.55)


def test_packet_loss_is_recorded_but_not_modelled(profiles, request_for) -> None:
    """V1 applies no retransmission penalty; saying so in the log beats leaving it implicit."""
    executor = ProfileExecutor(profiles)
    lossless = executor.execute(
        request_for(
            "CFG_REMOTE_STRONG",
            NetworkObservation(bandwidth_mbps=8.0, rtt_ms=70.0, packet_loss_frac=0.0),
        )
    )
    lossy = executor.execute(request_for("CFG_REMOTE_STRONG", DEGRADED))

    assert lossy.latency_s == lossless.latency_s
    assert lossy.metadata["packet_loss_frac"] == 0.01


# --- ProfileExecutor: remote failure ------------------------------------------------------


def test_remote_execution_fails_without_bandwidth(profiles, request_for) -> None:
    result = ProfileExecutor(profiles).execute(request_for("CFG_REMOTE_STRONG", DISCONNECTED))

    assert not result.success
    assert result.failure_reason is FailureReason.NETWORK_UNAVAILABLE
    assert result.prediction is None
    assert result.latency_s == DEFAULT_REMOTE_TIMEOUT_S
    assert result.communication_mb == 0.0, "nothing reached the server, so nothing is charged"
    assert result.onboard_energy_j == 1.0, "the frame was still captured and encoded"


def test_the_remote_timeout_is_configurable(profiles, request_for) -> None:
    executor = ProfileExecutor(profiles, remote_timeout_s=5.0)
    assert executor.execute(request_for("CFG_REMOTE_STRONG", DISCONNECTED)).latency_s == 5.0


def test_a_failed_execution_never_raises(profiles, request_for) -> None:
    """Losing the link is a situation the policy must handle, not a crash."""
    result = ProfileExecutor(profiles).execute(request_for("CFG_REMOTE_STRONG", DISCONNECTED))
    assert isinstance(result, ExecutionResult)


# --- predictions and determinism -----------------------------------------------------------


def test_the_executor_runs_without_any_prediction_source(profiles, request_for) -> None:
    """Resource-only mode: costs and failures are exercised with no task involved."""
    result = ProfileExecutor(profiles).execute(request_for("CFG_LOCAL_LIGHT"))
    assert result.success
    assert result.prediction is None


def test_predictions_come_from_the_injected_source(profiles, request_for) -> None:
    class StubSource:
        def prediction_for(self, request: ExecutionRequest) -> object:
            return {"frame": request.frame_id, "config": request.configuration.config_id}

    result = ProfileExecutor(profiles, predictions=StubSource()).execute(
        request_for("CFG_LOCAL_LIGHT", frame_id=7)
    )
    assert result.prediction == {"frame": 7, "config": "CFG_LOCAL_LIGHT"}


def test_a_failed_remote_execution_asks_for_no_prediction(profiles, request_for) -> None:
    class ExplodingSource:
        def prediction_for(self, request: ExecutionRequest) -> object:
            raise AssertionError("must not be consulted when execution failed")

    result = ProfileExecutor(profiles, predictions=ExplodingSource()).execute(
        request_for("CFG_REMOTE_STRONG", DISCONNECTED)
    )
    assert not result.success


def test_execution_is_deterministic(profiles, request_for) -> None:
    executor = ProfileExecutor(profiles)
    first = executor.execute(request_for("CFG_REMOTE_STRONG", DEGRADED))
    second = executor.execute(request_for("CFG_REMOTE_STRONG", DEGRADED))
    assert first.to_dict() == second.to_dict()


def test_behaviour_follows_placement_not_the_configuration_id(profiles, catalog) -> None:
    """Nothing may infer "remote" from the ID; the typed strategy field decides."""
    import dataclasses

    from missionbench.schemas.configuration import Placement

    remote = catalog.get("CFG_REMOTE_STRONG")
    relabelled = dataclasses.replace(remote, config_id="CFG_LOCAL_LIGHT")
    assert relabelled.strategy.placement is Placement.REMOTE

    request = ExecutionRequest(
        episode_id="E",
        frame_id=0,
        configuration=relabelled,
        network=DISCONNECTED,
        current_time_s=0.0,
        seed=1,
    )
    result = ProfileExecutor(profiles).execute(request)
    assert not result.success, "a locally-named remote configuration must still be remote"


# --- ExecutionResult invariants --------------------------------------------------------------


def test_a_result_may_not_be_both_successful_and_failed() -> None:
    with pytest.raises(ValueError, match="must not carry a failure_reason"):
        ExecutionResult(
            success=True,
            latency_s=0.1,
            onboard_energy_j=1.0,
            failure_reason=FailureReason.TIMEOUT,
        )


def test_a_failed_result_must_say_why() -> None:
    with pytest.raises(ValueError, match="must state a failure_reason"):
        ExecutionResult(success=False, latency_s=0.1, onboard_energy_j=1.0)


def test_result_metadata_is_read_only() -> None:
    result = ExecutionResult(success=True, latency_s=0.1, onboard_energy_j=1.0, metadata={"a": 1})
    with pytest.raises(TypeError):
        result.metadata["a"] = 2  # type: ignore[index]


def test_result_serialises_for_the_step_log(profiles, request_for) -> None:
    payload = ProfileExecutor(profiles).execute(request_for("CFG_REMOTE_STRONG")).to_dict()
    assert payload["success"] is True
    assert payload["communication_mb"] == pytest.approx(1.55)
    assert payload["failure_reason"] is None
    assert payload["has_prediction"] is False


# --- ReplayExecutor --------------------------------------------------------------------------


@pytest.fixture
def replay_records(data_dir: Path):
    return load_replay_records(data_dir / "predictions" / "synthetic_replay_example.json")


def test_replay_serves_the_recorded_outcome(replay_records, request_for) -> None:
    result = ReplayExecutor(replay_records).execute(request_for("CFG_REMOTE_STRONG", frame_id=0))
    assert result.success
    assert result.latency_s == pytest.approx(0.750)
    assert result.communication_mb == pytest.approx(1.55)
    assert result.prediction == {
        "instances": [{"prediction_id": "P0", "predicted_target_id": "T_A", "confidence": 0.91}]
    }


def test_replay_passes_the_payload_through_untouched(replay_records, request_for) -> None:
    """The payload's shape belongs to the task; this layer must not interpret it."""
    result = ReplayExecutor(replay_records).execute(request_for("CFG_LOCAL_LIGHT", frame_id=0))
    assert result.prediction == {"instances": []}


def test_replay_reproduces_a_recorded_failure(replay_records, request_for) -> None:
    result = ReplayExecutor(replay_records).execute(request_for("CFG_REMOTE_STRONG", frame_id=1))
    assert not result.success
    assert result.failure_reason is FailureReason.NETWORK_UNAVAILABLE
    assert result.prediction is None
    assert result.latency_s == pytest.approx(2.0)


def test_a_missing_record_is_a_failed_inference_not_a_crash(replay_records, request_for) -> None:
    """A slow configuration skips frames, so a policy can reach one nobody precomputed."""
    result = ReplayExecutor(replay_records).execute(request_for("CFG_LOCAL_LIGHT", frame_id=999))
    assert not result.success
    assert result.failure_reason is FailureReason.NO_PREDICTION_AVAILABLE


def test_strict_mode_treats_a_gap_as_a_fixture_bug(replay_records, request_for) -> None:
    with pytest.raises(SchemaValidationError, match="frame 999"):
        ReplayExecutor(replay_records, strict=True).execute(
            request_for("CFG_LOCAL_LIGHT", frame_id=999)
        )


def test_replay_record_set_rejects_duplicates(write_json) -> None:
    record = {
        "frame_id": 0,
        "config_id": "CFG_A",
        "success": True,
        "latency_ms": 1.0,
        "onboard_energy_j": 1.0,
    }
    payload = {"schema_version": "1.0", "records": [record, dict(record)]}
    with pytest.raises(SchemaValidationError, match="duplicate record"):
        load_replay_records(write_json(payload))


@pytest.mark.parametrize(
    ("success", "failure_reason", "message"),
    [
        (True, "timeout", "successful record has no failure_reason"),
        (False, None, "failed record must state a failure_reason"),
    ],
)
def test_replay_records_must_be_internally_consistent(
    write_json, success: bool, failure_reason: str | None, message: str
) -> None:
    record = {
        "frame_id": 0,
        "config_id": "CFG_A",
        "success": success,
        "latency_ms": 1.0,
        "onboard_energy_j": 1.0,
    }
    if failure_reason is not None:
        record["failure_reason"] = failure_reason
    with pytest.raises(SchemaValidationError, match=message):
        load_replay_records(write_json({"schema_version": "1.0", "records": [record]}))


# --- the real-model stub ---------------------------------------------------------------------


def test_the_real_executor_stub_fails_loudly(request_for) -> None:
    """A missing backend is a configuration error, not a mission condition to measure.

    Returning a failed result for every frame would look like a benchmark outcome.
    """
    # Construction raises, so selecting it fails before step 0 rather than after a mission.
    with pytest.raises(NotImplementedError, match="not implemented in V1"):
        RealSegmentationExecutor()


# --- the registry -----------------------------------------------------------------------------


def test_every_v1_backend_is_registered() -> None:
    assert executor_registry.names() == ("profile", "real_segmentation", "replay")


def test_the_registry_builds_a_working_executor(profiles, request_for) -> None:
    from missionbench.executor.registry import ExecutorContext, build_executor

    executor = build_executor("profile", ExecutorContext(episode_id="E", profiles=profiles))
    assert executor.executor_id == "profile"
    assert executor.execute(request_for("CFG_LOCAL_LIGHT")).success


def test_an_unknown_name_says_what_is_available() -> None:
    with pytest.raises(RegistryError, match=r"unknown executor 'proflie'; available: \["):
        executor_registry.create("proflie")


def test_a_duplicate_registration_is_refused() -> None:
    """Silently replacing an implementation would be undebuggable from a result file."""
    registry: Registry[object] = Registry("thing")
    registry.register("a", object)
    with pytest.raises(RegistryError, match="already registered"):
        registry.register("a", object)


# --- replaceability ------------------------------------------------------------------------------


def test_two_backends_are_interchangeable_behind_the_protocol(
    profiles, replay_records, request_for
) -> None:
    """Extensibility criterion 3: swap ProfileExecutor for ReplayExecutor, change nothing else.

    A caller written against the protocol drives either backend without knowing which it has.
    """

    def run(executor) -> list[bool]:
        return [
            executor.execute(request_for("CFG_LOCAL_LIGHT", frame_id=frame)).success
            for frame in (0, 1)
        ]

    assert run(ProfileExecutor(profiles)) == [True, True]
    assert run(ReplayExecutor(replay_records)) == [True, True]
