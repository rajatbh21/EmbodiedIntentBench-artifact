"""Invocation deadlines discard only late-stage state and retain paid work."""

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from experiments.mission_pipeline_routing.catalog import (
    Algorithm,
    Capability,
    Catalog,
    Costs,
    Task,
    default_catalog,
)
from experiments.mission_pipeline_routing.runtime import (
    CompletedPrediction,
    PublicContext,
    PublicPose,
    Selection,
)
from experiments.mission_pipeline_routing.verification import RuleVerdict, VerificationFactory
from experiments.mission_task_deadlines.contract import DeadlineProfile
from experiments.mission_task_deadlines.pipeline import DeadlinePipeline

MS = 1_000_000
SECOND = 1_000_000_000
IMAGE = "ASSOC_IMAGE_IOU"
REPEAT = "VERIFY_REPEAT2"


def profile(association_ms=200, verification_ms=30):
    return DeadlineProfile("unit_test", SECOND, association_ms * MS, verification_ms * MS)


def prediction(capture_ns=0, empty=False):
    mask = np.zeros((32, 32), dtype=np.bool_)
    if not empty:
        mask[13:19, 13:19] = True
    return CompletedPrediction(
        "selected_perception",
        capture_ns,
        capture_ns + 100 * MS,
        mask,
        PublicPose((0.0, 0.0, 4.0), (0.0, 0.0, 0.0, 1.0), np.pi / 2),
    )


def context(capture_ns=0, **overrides):
    return replace(
        PublicContext(
            capture_ns + 100 * MS,
            0.65,
            20 * SECOND,
            "search",
            0,
            20 * SECOND,
            20 * SECOND,
            2,
            2 * SECOND,
            4096,
            2400,
        ),
        **overrides,
    )


def run(pipeline, capture_ns=0, association=IMAGE, verification=REPEAT, **overrides):
    return pipeline.execute(
        Selection(association, verification),
        prediction(capture_ns),
        context(capture_ns, **overrides),
    )


def learned_factory(tmp_path, rule_type, *, memory_mb=8):
    path = tmp_path / "synthetic_unit_test_only.json"
    path.write_text('{"unit_test_only": true}\n')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    descriptor = Algorithm(
        "VERIFY_SYNTHETIC",
        Task.REPORT_VERIFICATION,
        "unit_test_rule",
        frozenset({Capability.TRACK_EVIDENCE}),
        frozenset(),
        Costs(3 * MS, 0.045, memory_mb, 10 * MS, 0.15),
        "learned_evidence_model",
        digest,
    )
    factory = VerificationFactory(descriptor, rule_type, ((str(path), digest),))
    return factory, Catalog([*default_catalog().algorithms, descriptor]), path


class Accept:
    def decide(self, features):
        return RuleVerdict(True, "unit_test_accept")


def test_each_task_has_its_own_release_budget_and_tick_boundary():
    pipeline = DeadlinePipeline(profile=profile(80, 10))
    out = run(pipeline)
    association, verification = out["stages"]
    assert association["release_ns"] == 100 * MS
    assert association["natural_finish_ns"] == 175 * MS
    assert association["usable_finish_ns"] == association["deadline_ns"] == 180 * MS
    assert verification["release_ns"] == 180 * MS
    assert verification["natural_finish_ns"] == 186 * MS
    assert verification["usable_finish_ns"] == verification["deadline_ns"] == 190 * MS
    assert association["committed"] and verification["committed"]
    assert (
        out["status"] == "completed" and pipeline.final_time_ns == out["completed_ns"] == 190 * MS
    )
    assert out["charged_energy_j"] == pytest.approx(0.75 + 0.3 + 0.075 + 0.075 + 0.015)
    assert verification["costs"]["routing_latency_ns"] == 0


def test_raw_finish_within_budget_does_not_bypass_usable_deadline():
    pipeline = DeadlinePipeline(profile=profile(75, 30))
    out = run(pipeline)
    stage = out["stages"][0]
    assert stage["raw_finish_ns"] == stage["deadline_ns"] == 175 * MS
    assert stage["usable_finish_ns"] == 180 * MS and stage["lateness_ns"] == 5 * MS
    assert out["status"] == "task_deadline_miss"
    assert not stage["committed"] and pipeline.snapshot(180 * MS)["tracks"] == ()


def test_waiting_after_input_availability_consumes_association_budget():
    pipeline = DeadlinePipeline(profile=profile(80, 10))
    out = run(pipeline, now_ns=111 * MS)
    stage = out["stages"][0]
    assert stage["release_ns"] == 100 * MS and stage["start_ns"] == 111 * MS
    assert stage["costs"]["queue_wait_ns"] == 11 * MS
    assert stage["deadline_ns"] == 180 * MS and not stage["on_time"]


def test_late_association_does_not_commit_or_dispatch_verifier_but_load_persists():
    pipeline = DeadlinePipeline(profile=profile(60, 30))
    late = run(pipeline)
    skipped = late["stages"][1]
    assert late["completed_ns"] == 180 * MS
    assert not skipped["dispatched"] and skipped["release_ns"] is None
    assert skipped["costs"]["charged_energy_j"] == 0
    assert late["candidates"] == late["emitted_reports"] == []
    assert pipeline.executed_calls == {}
    assert pipeline.stage_dispatches == {IMAGE: 1}
    assert pipeline.snapshot(late["completed_ns"])["verification_id"] is None
    assert late["charged_energy_j"] == pytest.approx(1.125)
    next_out = run(pipeline, 2 * SECOND)
    assert next_out["stages"][0]["committed"]
    assert next_out["stages"][0]["costs"]["load_latency_ns"] == 0
    assert next_out["stages"][1]["costs"]["load_latency_ns"] == 5 * MS
    assert next_out["candidates"][0]["hits"] == 1
    assert pipeline.late_jobs == pipeline.completed_jobs == 1


def test_late_verification_keeps_timely_tracks_and_does_not_suppress_future_report(tmp_path):
    factory, catalog, _ = learned_factory(tmp_path, Accept)
    pipeline = DeadlinePipeline(
        profile=profile(200, 10),
        catalog=catalog,
        verification_factories={"VERIFY_SYNTHETIC": factory},
    )
    first = run(pipeline, verification="VERIFY_SYNTHETIC")
    assert first["stages"][0]["committed"] and not first["stages"][1]["committed"]
    assert first["stages"][1]["lateness_ns"] == 10 * MS
    assert first["selected_rulecalls"] == 0 and pipeline._verification.reported_tracks == set()
    assert pipeline._association.sources == {0: "selected_perception"}
    second = run(pipeline, 2 * SECOND, verification="VERIFY_SYNTHETIC")
    assert second["stages"][1]["costs"]["load_latency_ns"] == 0
    assert second["candidates"][0]["hits"] == 2
    assert second["selected_rulecalls"] == 1 and len(second["emitted_reports"]) == 1
    assert second["emitted_reports"][0]["report_id"] == 1
    assert pipeline.executed_calls == {IMAGE: 2, "VERIFY_SYNTHETIC": 1}
    assert pipeline.stage_dispatches == {IMAGE: 2, "VERIFY_SYNTHETIC": 2}


def test_missed_verification_never_calls_rule_even_with_sufficient_evidence():
    pipeline = DeadlinePipeline(profile=profile(200, 5))
    for capture in (0, 2 * SECOND, 4 * SECOND):
        out = run(pipeline, capture)
        assert out["stages"][0]["committed"] and not out["stages"][1]["committed"]
        assert out["selected_rulecalls"] == 0 and not out["emitted_reports"]
    assert pipeline._verification.calls == pipeline._verification.rule_calls == 0
    assert pipeline._association.calls == 3 and pipeline.late_jobs == 3


def test_strict_intent_floor_is_enforced_before_selected_model(tmp_path):
    factory, catalog, _ = learned_factory(tmp_path, Accept)
    pipeline = DeadlinePipeline(
        profile=profile(), catalog=catalog, verification_factories={"VERIFY_SYNTHETIC": factory}
    )
    for capture in (0, 2 * SECOND):
        out = run(
            pipeline,
            capture,
            verification="VERIFY_SYNTHETIC",
            required_hits=3,
            required_span_ns=4 * SECOND,
        )
        assert out["selected_rulecalls"] == 0 and out["emitted_reports"] == []
    out = run(
        pipeline,
        4 * SECOND,
        verification="VERIFY_SYNTHETIC",
        required_hits=3,
        required_span_ns=4 * SECOND,
    )
    assert out["selected_rulecalls"] == 1 and len(out["emitted_reports"]) == 1


@pytest.mark.parametrize("deadline_name", ["deadline_ns", "phase_deadline_ns"])
def test_timely_task_cannot_waive_outer_or_phase_deadline(deadline_name):
    pipeline = DeadlinePipeline(profile=profile())
    run(pipeline)
    out = run(pipeline, 2 * SECOND, **{deadline_name: 2_150_000_000})
    assert all(stage["on_time"] for stage in out["stages"])
    assert out["emitted_reports"] == [] and out["selected_rulecalls"] == 0
    assert out["verification_decisions"][0]["reason"] == "completion_after_public_deadline"


def test_empty_prediction_is_not_successful_evidence():
    pipeline = DeadlinePipeline(profile=profile())
    out = pipeline.execute(Selection(IMAGE, REPEAT), prediction(empty=True), context())
    assert out["status"] == "completed"
    assert out["candidates"] == out["emitted_reports"] == []
    assert pipeline.snapshot(out["completed_ns"])["tracks"] == () and out["selected_rulecalls"] == 0


def test_failed_association_discards_partial_temporal_state_and_does_not_dispatch_verifier(
    monkeypatch,
):
    from experiments.mission_task_deadlines.pipeline import AssociationExecutor

    def broken(self, selected_prediction, completion_ns):
        self.sources[selected_prediction.capture_ns] = "partial_state_must_not_escape"
        raise ValueError("synthetic_association_failure")

    monkeypatch.setattr(AssociationExecutor, "update", broken)
    pipeline = DeadlinePipeline(profile=profile())
    out = run(pipeline)
    assert out["status"] == "execution_failed" and pipeline.failed_jobs == 1
    assert not out["stages"][0]["committed"] and not out["stages"][1]["dispatched"]
    assert pipeline._association.sources == {} and pipeline._verification is None
    assert pipeline.final_time_ns == 180 * MS and pipeline.cumulative_energy_j > 0
    assert out["emitted_reports"] == [] and pipeline.executed_calls == {IMAGE: 1}


def test_failed_rule_state_copy_keeps_paid_time_and_has_no_forward_or_report(tmp_path):
    class NotCopyable(Accept):
        def __deepcopy__(self, memo):
            raise ValueError("synthetic_copy_failure")

    factory, catalog, _ = learned_factory(tmp_path, NotCopyable)
    pipeline = DeadlinePipeline(
        profile=profile(), catalog=catalog, verification_factories={"VERIFY_SYNTHETIC": factory}
    )
    out = run(pipeline, verification="VERIFY_SYNTHETIC")
    assert out["status"] == "execution_failed" and out["stages"][0]["committed"]
    assert pipeline.final_time_ns == 200 * MS and out["selected_rulecalls"] == 0
    assert out["emitted_reports"] == [] and pipeline.cumulative_energy_j > 0


def test_phase_change_cannot_reuse_old_phase_evidence():
    pipeline = DeadlinePipeline(profile=profile())
    run(pipeline)
    out = run(pipeline, 2 * SECOND, phase_id="next", phase_start_ns=2 * SECOND)
    assert out["candidates"][0]["hits"] == 1
    assert out["selected_rulecalls"] == 0 and out["emitted_reports"] == []


def test_memory_capability_and_cross_task_rejections_are_atomic():
    pipeline = DeadlinePipeline(profile=profile())
    initial = dict(pipeline.snapshot(0))
    with pytest.raises(ValueError, match="memory"):
        run(pipeline, memory_capacity_mb=2401)
    with pytest.raises(ValueError, match="capabilities"):
        run(pipeline, association="ASSOC_TERRAIN_WORLD")
    with pytest.raises(ValueError, match="requested task"):
        run(pipeline, association=REPEAT)
    assert dict(pipeline.snapshot(0)) == initial
    assert pipeline.cumulative_energy_j == 0 and pipeline.stage_dispatches == {}


def test_checkpoint_change_rejects_before_switch_or_payment(tmp_path):
    factory, catalog, path = learned_factory(tmp_path, Accept)
    pipeline = DeadlinePipeline(
        profile=profile(), catalog=catalog, verification_factories={"VERIFY_SYNTHETIC": factory}
    )
    run(pipeline)
    before, energy = dict(pipeline.snapshot(2_100_000_000)), pipeline.cumulative_energy_j
    path.write_text("tampered")
    with pytest.raises(ValueError, match="checkpoint changed"):
        run(pipeline, 2 * SECOND, association="ASSOC_FLAT_WORLD", verification="VERIFY_SYNTHETIC")
    assert (
        dict(pipeline.snapshot(2_100_000_000)) == before and pipeline.cumulative_energy_j == energy
    )


def test_factory_descriptor_cost_substitution_is_atomic(tmp_path):
    factory, _, _ = learned_factory(tmp_path, Accept)
    substituted = replace(factory.descriptor, costs=Costs(0, 0, 0, 0, 0))
    pipeline = DeadlinePipeline(
        profile=profile(),
        catalog=Catalog([*default_catalog().algorithms, substituted]),
        verification_factories={"VERIFY_SYNTHETIC": factory},
    )
    with pytest.raises(ValueError, match="identity, capabilities or costs"):
        run(pipeline, verification="VERIFY_SYNTHETIC")
    assert pipeline.cumulative_energy_j == 0 and pipeline.final_time_ns == 0


def test_failed_rule_keeps_paid_busy_time_and_timely_association_without_partial_report(tmp_path):
    class Broken:
        def decide(self, features):
            raise ValueError("synthetic_failure")

    factory, catalog, _ = learned_factory(tmp_path, Broken)
    pipeline = DeadlinePipeline(
        profile=profile(), catalog=catalog, verification_factories={"VERIFY_SYNTHETIC": factory}
    )
    run(pipeline, verification="VERIFY_SYNTHETIC")
    before_energy = pipeline.cumulative_energy_j
    out = run(pipeline, 2 * SECOND, verification="VERIFY_SYNTHETIC")
    assert out["status"] == "execution_failed" and pipeline.failed_jobs == 1
    assert out["stages"][0]["committed"] and not out["stages"][1]["committed"]
    assert out["selected_rulecalls"] == 1 and out["emitted_reports"] == []
    assert pipeline._association.sources == {
        0: "selected_perception",
        2 * SECOND: "selected_perception",
    }
    assert pipeline._verification.reported_tracks == set()
    assert (
        pipeline.cumulative_energy_j > before_energy
        and pipeline.final_time_ns == out["completed_ns"]
    )
    with pytest.raises(ValueError, match="reused capture"):
        run(pipeline, 2 * SECOND, verification="VERIFY_SYNTHETIC", now_ns=out["completed_ns"])


def test_snapshot_is_immutable_and_cannot_read_prior_to_paid_finish():
    pipeline = DeadlinePipeline(profile=profile())
    out = run(pipeline)
    with pytest.raises(ValueError, match="availability"):
        pipeline.snapshot(out["completed_ns"] - 1)
    snapshot = pipeline.snapshot(out["completed_ns"])
    with pytest.raises(TypeError):
        snapshot["emitted_reports"] = 99
    with pytest.raises(TypeError):
        snapshot["tracks"][0]["capture_times_ns"] = ()
    assert snapshot["ground_truth_access"] is False


def test_future_reused_capture_and_capture_during_late_busy_period_are_rejected():
    pipeline = DeadlinePipeline(profile=profile(60, 30))
    with pytest.raises(ValueError, match="Future prediction"):
        run(pipeline, now_ns=0)
    out = run(pipeline)
    with pytest.raises(ValueError, match="reused capture"):
        run(pipeline, now_ns=out["completed_ns"])
    with pytest.raises(ValueError, match="busy"):
        run(pipeline, out["completed_ns"] - 1)
    run(pipeline, out["completed_ns"])


def test_verifier_switch_preserves_tracks_but_not_suppression_and_ids_never_repeat():
    pipeline = DeadlinePipeline(profile=profile())
    run(pipeline)
    first = run(pipeline, 2 * SECOND)["emitted_reports"][0]
    next_out = run(pipeline, 4 * SECOND, verification="VERIFY_REPEAT3")
    second = next_out["emitted_reports"][0]
    assert next_out["candidates"][0]["hits"] == 3
    assert next_out["stages"][1]["reset_slot"]
    assert first["report_id"] == 1 and second["report_id"] == 2
    assert next_out["selected_verifier_rule_calls"] == 2


def test_resident_previous_verifier_is_included_in_peak_memory_check(tmp_path):
    factory, catalog, _ = learned_factory(tmp_path, Accept, memory_mb=100)
    pipeline = DeadlinePipeline(
        profile=profile(), catalog=catalog, verification_factories={"VERIFY_SYNTHETIC": factory}
    )
    run(pipeline, verification="VERIFY_SYNTHETIC")
    energy = pipeline.cumulative_energy_j
    # New association + old 100MB verifier coexist before the small new verifier loads.
    with pytest.raises(ValueError, match="memory"):
        run(pipeline, 2 * SECOND, association="ASSOC_FLAT_WORLD", memory_capacity_mb=2508)
    assert pipeline.cumulative_energy_j == energy


def test_depleted_public_battery_rejects_before_any_task_or_payment():
    pipeline = DeadlinePipeline(profile=profile())
    with pytest.raises(ValueError, match="depleted battery"):
        run(pipeline, battery_frac=0)
    assert pipeline.cumulative_energy_j == 0 and pipeline.stage_dispatches == {}
    assert pipeline.final_time_ns == 0 and pipeline.snapshot(0)["tracks"] == ()


def test_battery_exhaustion_at_association_finish_never_dispatches_verifier():
    pipeline = DeadlinePipeline(profile=profile(), battery_capacity_j=1.125)
    out = run(pipeline, battery_frac=1)
    assert out["status"] == "battery_depleted"
    assert out["remaining_after_association_j"] == 0
    assert out["stages"][0]["committed"]
    assert out["stages"][1]["status"] == "not_dispatched_battery_depleted"
    assert out["stages"][1]["costs"]["charged_energy_j"] == 0
    assert out["completed_ns"] == 180 * MS and pipeline.cumulative_energy_j == 1.125
    assert pipeline._verification is None and out["selected_rulecalls"] == 0
    assert pipeline.stage_dispatches == {IMAGE: 1} and pipeline.battery_depleted_jobs == 1


def test_association_battery_guard_includes_elapsed_flight_but_does_not_charge_it_twice():
    pipeline = DeadlinePipeline(profile=profile(), battery_capacity_j=10, flight_power_w=120)
    out = run(pipeline, battery_frac=1)
    assert out["remaining_after_association_j"] == pytest.approx(10 - 1.125 - 0.08 * 120)
    assert out["battery_depleted_before_verification"]
    assert out["status"] == "battery_depleted" and out["emitted_reports"] == []
    assert pipeline.cumulative_energy_j == 1.125  # caller pays baseline flight exactly once


@pytest.mark.parametrize(
    "kwargs",
    [
        {"battery_capacity_j": 0},
        {"battery_capacity_j": True},
        {"battery_capacity_j": float("inf")},
        {"flight_power_w": -1},
        {"flight_power_w": True},
    ],
)
def test_invalid_battery_guard_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        DeadlinePipeline(profile=profile(), **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"step_ns": 0},
        {"step_ns": True},
        {"routing_latency_ns": -1},
        {"routing_energy_j": float("nan")},
    ],
)
def test_invalid_runtime_costs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        DeadlinePipeline(profile=profile(), **kwargs)
