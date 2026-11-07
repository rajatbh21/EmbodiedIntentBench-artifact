"""Paid perception deadlines include all overhead and gate every cache access."""

from dataclasses import replace

import numpy as np
import pytest

from experiments.lake_causal_router import MissionCosts
from experiments.mission_network_emulation import LinkSegment, LinkTrace
from experiments.mission_network_policies import ActionDescriptor, PolicyCosts
from experiments.mission_network_study_v1.executors import prepare_payload
from experiments.mission_task_deadlines.contract import DeadlineProfile
from experiments.mission_task_deadlines.perception import execute_perception

MS, SECOND = 1_000_000, 1_000_000_000


class Cache:
    def __init__(self, selected):
        self.selected, self.reads = selected, []

    def __getitem__(self, key):
        self.reads.append(key)
        if self.selected is None:
            raise AssertionError("A discarded output must never access a cached prediction")
        return self.selected


def setup():
    rgb = np.arange(45, dtype=np.uint8).reshape(3, 5, 3)
    prepared = prepare_payload(rgb)
    mask = np.arange(15).reshape(3, 5) % 3 == 0
    cache = Cache(mask)
    costs = PolicyCosts(MissionCosts(120, 2, 108000, 120, 0.05, 0.75, 10 * MS))
    # Each payload serialization takes exactly one millisecond, with zero RTT.
    trace = LinkTrace(
        (
            LinkSegment(
                0, 30 * SECOND, len(prepared.payload) * 8000, prepared.response_bytes * 8000, 0
            ),
        )
    )
    action = ActionDescriptor(
        "LOCAL", "SOURCE", "local", "synthetic_test_signature", 0.1, 1.5, 0.01, 0.15, 100
    )
    return dict(
        action=action,
        rgb=rgb,
        prepared=prepared,
        predictions=cache,
        release_ns=0,
        profile=DeadlineProfile("unit_test", 200 * MS, 200 * MS, 30 * MS),
        mission_deadline_ns=20 * SECOND,
        current_action_id=None,
        trace=trace,
        costs=costs,
        router_latency_ns=50 * MS,
        router_energy_j=0.75,
        step_ns=10 * MS,
    )


def remote(settings):
    return replace(settings["action"], action_id="REMOTE", placement="remote", energy_j=0.0)


def set_budget(settings, budget_ns):
    settings["profile"] = replace(settings["profile"], perception_ns=budget_ns)


def test_local_deadline_includes_preparation_routing_loading_service_and_exact_boundary():
    settings = setup()
    set_budget(settings, 180 * MS)
    out = execute_perception(**settings)
    event = out.event
    assert event["decision_ns"] == 20 * MS and event["dispatch_ns"] == 80 * MS
    assert event["raw_finish_ns"] == event["usable_finish_ns"] == event["deadline_ns"] == 180 * MS
    assert event["admitted"] and event["task_deadline_met"] and event["status"] == "completed"
    assert out.nonflight_energy_j == pytest.approx(0.3 + 0.75 + 0.15 + 1.5)
    assert settings["predictions"].reads == [settings["prepared"].rgb_sha256]
    assert out.resident_action_id == "LOCAL" and out.communication_mb == 0
    assert not event["network_dispatched"]


def test_one_nanosecond_late_is_discarded_without_cache_access():
    settings = setup()
    set_budget(settings, 180 * MS - 1)
    settings["predictions"] = Cache(None)
    out = execute_perception(**settings)
    assert out.event["status"] == "task_deadline_miss" and out.event["lateness_ns"] == 1
    assert out.mask is None and not out.event["prediction_cache_read"]
    assert settings["predictions"].reads == [] and out.nonflight_energy_j == pytest.approx(2.7)


def test_tick_rounding_can_make_an_otherwise_timely_local_output_late():
    settings = setup()
    set_budget(settings, 185 * MS)
    settings["action"] = replace(settings["action"], latency_s=0.1001)
    settings["predictions"] = Cache(None)
    out = execute_perception(**settings)
    assert out.event["raw_finish_ns"] == 180_100_000 < out.event["deadline_ns"]
    assert out.completed_ns == 190 * MS and out.event["lateness_ns"] == 5 * MS
    assert out.mask is None and settings["predictions"].reads == []


def test_local_service_still_runs_and_is_fully_paid_when_overhead_exhausts_budget():
    settings = setup()
    set_budget(settings, 50 * MS)
    settings["predictions"] = Cache(None)
    out = execute_perception(**settings)
    assert out.event["dispatch_ns"] == 80 * MS > out.event["deadline_ns"]
    assert out.completed_ns == 180 * MS
    assert out.event["costs"][
        "compute_energy_j"
    ] == 1.5 and out.nonflight_energy_j == pytest.approx(2.7)
    assert out.resident_action_id == "LOCAL" and settings["predictions"].reads == []


@pytest.mark.parametrize("budget_ns", [80 * MS, 79 * MS])
def test_remote_exhausted_budget_never_dispatches_even_at_exact_bound(budget_ns):
    settings = setup()
    settings["action"] = remote(settings)
    settings["predictions"] = Cache(None)
    set_budget(settings, budget_ns)
    out = execute_perception(**settings)
    assert out.event["status"] == "task_deadline_miss" and out.event["deadline_missed"]
    assert out.event["pre_dispatch_budget_exhausted"] and not out.event["network_dispatched"]
    assert out.event["transport"] is None and out.event["radio"] is None
    assert out.event["deadline_stop_reason"] == "task_deadline_miss"
    assert out.completed_ns == 80 * MS and out.nonflight_energy_j == pytest.approx(1.2)
    assert out.communication_mb == 0 and settings["predictions"].reads == []


def test_remote_response_is_exact_and_immutable_after_all_paid_stages():
    settings = setup()
    settings["action"] = remote(settings)
    out = execute_perception(**settings)
    assert out.event["transport"]["finish_ns"] == 182 * MS
    assert out.event["raw_finish_ns"] == 192 * MS and out.completed_ns == 200 * MS
    assert out.event["admitted"] and out.event["deadline_stop_reason"] is None
    assert out.event["transport_deadline_constraint"] == "task_deadline_miss"
    assert np.array_equal(out.mask, settings["predictions"].selected)
    assert settings["prepared"].response_bytes == 2  # non-byte-aligned, 15-pixel mask
    assert len(settings["predictions"].reads) == 1
    assert out.communication_mb == (len(settings["prepared"].payload) + 2) / 1e6
    assert out.nonflight_energy_j == pytest.approx(1.2 + 0.15 + 0.054)
    with pytest.raises(ValueError):
        out.mask.setflags(write=True)
    settings["predictions"].selected[:] = False
    assert out.mask.any()


def test_completed_remote_response_is_discarded_when_decode_exceeds_deadline():
    settings = setup()
    settings["action"] = remote(settings)
    settings["predictions"] = Cache(None)
    set_budget(settings, 190 * MS)
    out = execute_perception(**settings)
    assert out.event["transport"]["status"] == "completed" and out.event["output_arrived"]
    assert out.event["transport"]["finish_ns"] == 182 * MS < out.event["deadline_ns"]
    assert out.event["raw_finish_ns"] == 192 * MS and out.completed_ns == 200 * MS
    assert (
        out.event["status"] == "task_deadline_miss"
        and out.event["costs"]["decode_energy_j"] == 0.15
    )
    assert settings["predictions"].reads == [] and out.mask is None
    assert out.event["deadline_stop_reason"] is None  # response transport itself completed


@pytest.mark.parametrize("deadline_delta_ns", [0, -1])
def test_remote_exact_response_completion_at_inclusive_deadline(deadline_delta_ns):
    settings = setup()
    settings["action"] = replace(remote(settings), load_latency_s=0, load_energy_j=0)
    settings["costs"] = replace(settings["costs"], encode_s=0, decode_s=0)
    settings["router_latency_ns"] = 0
    settings["router_energy_j"] = 0
    settings["step_ns"] = 1
    set_budget(settings, 102 * MS + deadline_delta_ns)
    if deadline_delta_ns:
        settings["predictions"] = Cache(None)
    out = execute_perception(**settings)
    assert out.completed_ns == 102 * MS + deadline_delta_ns
    assert out.event["admitted"] is (deadline_delta_ns == 0)
    assert out.event["status"] == ("completed" if deadline_delta_ns == 0 else "task_deadline_miss")
    assert len(settings["predictions"].reads) == (1 if deadline_delta_ns == 0 else 0)


@pytest.mark.parametrize(
    ("task_budget", "mission_deadline", "expected_status", "finish", "deadline_missed"),
    [
        (2 * SECOND, 20 * SECOND, "task_deadline_miss", 2 * SECOND, True),
        (20 * SECOND, 25 * SECOND, "transport_timeout", 8_080_000_000, False),
        (3 * SECOND, SECOND, "mission_deadline_exceeded", SECOND, False),
    ],
)
def test_task_deadline_transport_timeout_and_mission_bound_remain_distinct(
    task_budget, mission_deadline, expected_status, finish, deadline_missed
):
    settings = setup()
    settings["action"] = remote(settings)
    settings["predictions"] = Cache(None)
    settings["trace"] = LinkTrace((LinkSegment(0, 30 * SECOND, 1, 1, 0, True),))
    settings["mission_deadline_ns"] = mission_deadline
    set_budget(settings, task_budget)
    out = execute_perception(**settings)
    assert out.event["status"] == expected_status and out.completed_ns == finish
    assert out.event["deadline_missed"] is deadline_missed
    assert out.event["costs"]["radio_energy_j"] == pytest.approx((finish - 80 * MS) / SECOND * 0.5)
    assert out.communication_mb == 0 and settings["predictions"].reads == []


def test_outage_after_partial_upload_preserves_actual_transferred_bytes_and_radio_cost():
    settings = setup()
    settings["action"] = remote(settings)
    settings["predictions"] = Cache(None)
    set_budget(settings, 100 * MS)
    settings["trace"] = LinkTrace(
        (
            LinkSegment(0, 85 * MS, len(settings["prepared"].payload) * 800, 1, 0),
            LinkSegment(85 * MS, SECOND, 0, 0, 0, True),
        )
    )
    out = execute_perception(**settings)
    assert out.event["status"] == "task_deadline_miss"
    assert out.communication_mb == len(settings["prepared"].payload) / 2e6
    assert out.event["transport"]["downloaded_bits_numerator"] == 0
    assert out.event["costs"]["radio_energy_j"] == pytest.approx(0.005 * 2 + 0.015 * 0.5)
    assert out.nonflight_energy_j == pytest.approx(1.2175)
    assert settings["predictions"].reads == [] and out.mask is None


def test_trace_exhaustion_cannot_read_the_lost_output_or_charge_decode():
    settings = setup()
    settings["action"] = remote(settings)
    settings["predictions"] = Cache(None)
    settings["trace"] = LinkTrace((replace(settings["trace"].segments[0], end_ns=100 * MS),))
    set_budget(settings, SECOND)
    out = execute_perception(**settings)
    assert out.event["status"] == "trace_exhausted"
    assert not out.event["output_arrived"] and out.event["costs"]["decode_energy_j"] == 0
    assert settings["predictions"].reads == [] and out.mask is None


def test_abstention_still_pays_preparation_router_time_and_energy_without_loading():
    settings = setup()
    settings["action"] = None
    settings["predictions"] = Cache(None)
    settings["current_action_id"] = "PREVIOUS"
    out = execute_perception(**settings)
    assert out.event["status"] == "abstained" and not out.event["admitted"]
    assert out.completed_ns == 70 * MS and out.nonflight_energy_j == pytest.approx(1.05)
    assert out.resident_action_id == "PREVIOUS" and not out.event["reset_slot"]
    assert out.event["costs"]["load_latency_ns"] == out.event["costs"]["load_energy_j"] == 0
    assert settings["predictions"].reads == [] and out.mask is None


def test_resident_action_is_not_loaded_again_and_absolute_release_does_not_reset_clock():
    settings = setup()
    settings["current_action_id"] = settings["action"].action_id
    settings["release_ns"] = 5 * SECOND
    set_budget(settings, 170 * MS)
    out = execute_perception(**settings)
    assert out.event["admitted"] and out.completed_ns == out.event["deadline_ns"] == 5_170_000_000
    assert out.event["costs"]["load_energy_j"] == 0 and not out.event["reset_slot"]
    assert out.nonflight_energy_j == pytest.approx(2.55)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.update(prepared=replace(s["prepared"], response_bytes=1)),
        lambda s: s.update(prepared=replace(s["prepared"], payload=b"")),
        lambda s: s.update(prepared=replace(s["prepared"], payload=bytearray(b"mutable"))),
        lambda s: s.update(prepared=replace(s["prepared"], rgb_sha256="another_rgb")),
        lambda s: s.update(router_latency_ns=True),
        lambda s: s.update(step_ns=0),
        lambda s: s.update(router_energy_j=float("nan")),
        lambda s: s.update(current_action_id=""),
    ],
)
def test_invalid_requests_fail_before_prediction_cache_access(mutate):
    settings = setup()
    mutate(settings)
    with pytest.raises(ValueError):
        execute_perception(**settings)
    assert settings["predictions"].reads == []
