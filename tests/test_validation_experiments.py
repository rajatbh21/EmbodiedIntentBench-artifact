"""Checks for the benchmark-validation experiments (experiments/validation).

Runtime checks execute a few cells of the committed replay bundle directly
(no copy, CPU only). Artifact checks validate the committed summary under
examples/validation and skip if it is absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "data/replay"
SUMMARY = ROOT / "examples/validation/summary.json"

pytest.importorskip("numpy")


@pytest.fixture(scope="module")
def ws():
    from experiments.validation import runtime

    runtime.install_memoization()
    return runtime.Workspace(BUNDLE)


@pytest.fixture(scope="module")
def references():
    return {r["run_id"]: r for r in json.loads((BUNDLE / "expected_rows.json").read_text())}


def cell(ws, run_id):
    return next(i for i in ws.plan["grid"] if i["run_id"] == run_id)


SAMPLE = (
    "sector_revisit/sector_7/urgency_recovery/degraded/ample/risk_current",
    "shoreline_loop/sector_3/urgency_increase/good/residual_250j/fixed_local_mask",
    "wide_area_search/sector_3/no_change/degraded/ample/fixed_remote_mask",
)


@pytest.mark.parametrize("run_id", SAMPLE)
def test_instrumented_runtime_reproduces_reference_rows_and_outcome(ws, references, run_id):
    from experiments.intent_mission_v2.runtime import outcome as frozen
    from experiments.validation import contracts, runtime
    from tools import replay

    res = runtime.run_case(ws, cell(ws, run_id))
    assert replay.compare_rows([res["row"]], [references[run_id]]) == []
    assert res["pipeline_failures"] == 0
    case = ws.plan["cases"][res["row"]["flight"]]
    ref = contracts.contract_for(case, res["row"]["battery"], "confirm")
    ours, theirs = (
        contracts.outcome(ref, res["events"], res["result"]),
        frozen(ref, res["events"], res["result"]),
    )
    for key in ("dynamic_mission_success", "checks", "failed_constraints"):
        assert ours[key] == theirs[key]


def test_variants_change_only_declared_fields_and_stay_redacted():
    from experiments.validation import contracts

    case = json.loads((BUNDLE / "study.json").read_text())["cases"]["sector_revisit/sector_3"]
    base = contracts.contract_for(case, 0.95, "confirm").document()
    for name in contracts.VARIANTS:
        c = contracts.contract_for(case, 0.95, name)
        doc, public = c.document(), c.public_document()
        assert "evaluator_target_ids" not in json.dumps(public)
        assert not any(t in json.dumps(public) for t in case["evaluator_target_ids"])
        assert doc["evaluator_target_ids"] == base["evaluator_target_ids"]
        assert doc["limits"] == base["limits"] and doc["duration_s"] == base["duration_s"]
    strict = contracts.contract_for(case, 0.95, "confirm_strict").document()["mission"]["phases"][0]
    assert (strict["min_hits"], strict["min_evidence_span_s"]) == (3, 4.0)
    revisit = contracts.contract_for(case, 0.95, "revisit").document()["mission"]["phases"]
    assert [(p["start_s"], p["end_s"]) for p in revisit] == [(0.0, 18.0), (18.0, 36.0)]
    tc = contracts.contract_for(case, 0.95, "time_critical").document()
    assert tc["reporting"]["deadline_s"] == tc["mission"]["phases"][0]["deadline_s"] == 36.0
    assert not contracts.VARIANTS["survey"]["rerun_supported"]


def test_no_skip_processes_every_capture_without_dropping_evidence(ws):
    from experiments.validation import runtime

    item = cell(ws, "sector_revisit/sector_7/urgency_recovery/degraded/ample/fixed_local_mask")
    causal = runtime.run_case(ws, item)
    ablated = runtime.run_case(ws, item, protocol="no_skip")
    assert causal["row"]["skipped_opportunity_count"] > 0
    assert ablated["row"]["decision_count"] == ablated["row"]["eligible_opportunity_count"]
    assert ablated["row"]["skipped_opportunity_count"] == 0
    assert ablated["pipeline_failures"] == 0
    # Every processed capture is still charged; queued work is not free.
    assert ablated["row"]["nonflight_energy_j"] > causal["row"]["nonflight_energy_j"]
    # The tracker relaxation is scoped to the no-skip run.
    from experiments.lake_tracking import StaticReportTracker

    assert StaticReportTracker.update.__qualname__ == "StaticReportTracker.update"


def test_latency_multiplier_scales_latency_and_local_energy_only(ws):
    from experiments.validation import runtime

    actions = ws.load_case(ws.plan["grid"][0])["actions"]
    scaled = runtime.scale_actions(actions, 2)
    for a, b in zip(actions, scaled, strict=True):
        assert b.latency_s == pytest.approx(2 * a.latency_s)
        assert b.energy_j == pytest.approx(2 * a.energy_j if a.placement == "local" else a.energy_j)
        assert (b.load_latency_s, b.load_energy_j, b.memory_mb) == (
            a.load_latency_s,
            a.load_energy_j,
            a.memory_mb,
        )
    with pytest.raises(ValueError):
        runtime.scale_actions(actions, 0)


def test_scripted_schedule_reproduces_a_policy_run(ws):
    from experiments.validation import execution, hindsight, runtime

    item = cell(ws, "shoreline_loop/sector_3/no_change/good/ample/risk_current")
    res = runtime.run_case(ws, item)
    schedule = ws.load_case(item)["scenario"].schedule
    indices = sorted(
        {e["observation_id"] for e in res["events"]} | set(res["result"]["skipped_observation_ids"])
    )
    release_ns = {i: schedule.at(i) - schedule.epoch_ns for i in indices}
    compact = execution.compact_events(res["events"])
    script = hindsight.script_from_events(compact, release_ns)
    replayed = runtime.run_case(ws, item, policy=execution.ScriptedPolicy(script))
    assert [e["selected_action_id"] for e in replayed["events"]] == [
        e["selected_action_id"] for e in res["events"]
    ]
    assert replayed["outcome"]["checks"] == res["outcome"]["checks"]


class _Frame:
    def __init__(self, rgb, ids):
        self.rgb, self.object_ids = rgb, ids


def test_leakage_observer_flags_injected_leaks():
    from experiments.mission_network_emulation import LinkSnapshot
    from experiments.validation.leakage import LeakageObserver

    frames = {
        0: _Frame(np.zeros((2, 2, 3), np.uint8), np.zeros((2, 2), np.uint8)),
        1: _Frame(np.ones((2, 2, 3), np.uint8), np.zeros((2, 2), np.uint8)),
    }
    obs = LeakageObserver(["p_1"], frames, "no_change")
    inputs = dict(
        release_ns=0,
        current_action_id=None,
        battery_frac=0.9,
        cumulative_communication_mb=0.0,
        link=LinkSnapshot(5, 1, 1, 1, False),
        payload_bytes=1,
        response_bytes=1,
        intent=dict(
            current_intent=dict(version=0, effective_from_ns=0, perception_budget_ns=8 * 10**9),
            evaluator_target_ids=["p_1"],
        ),
        rgb=frames[1].rgb,
    )
    obs.policy_input(index=0, decision_ns=5, rgb_key="k0", update=None, inputs=inputs)
    obs.selected(index=0, action_id="A", rgb_key="k0")
    obs.mask_access("B", "k1")
    kinds = {v["kind"] for v in obs.summary()["violations"]}
    assert {
        "forbidden_intent_key",
        "target_identity_in_intent",
        "rgb_not_current_capture",
        "rgb_shares_other_capture",
        "unselected_prediction_read",
        "prediction_for_other_capture",
    } <= kinds


def test_bounds_are_monotone_and_relax_correctly():
    from experiments.validation.bounds import bound_result

    phase = dict(
        phase_id="p",
        start_s=0.0,
        end_s=10.0,
        deadline_s=12.0,
        min_hits=2,
        min_evidence_span_s=2.0,
        required_recall=2 / 3,
        max_false_components=2,
    )
    doc = dict(mission=dict(phases=[phase]), reporting=dict(min_report_recall=2 / 3))
    caps = [
        dict(
            t=float(t),
            visible=["a", "b"],
            detectable=["a", "b"],
            matched={"m": ["a"] if t in (0, 2) else []},
        )
        for t in range(0, 10, 2)
    ]
    table = dict(focal=["a", "b", "c"], captures=caps)
    assert bound_result(table, doc, "sensor")["passed"]
    union = bound_result(table, doc, "model_union")
    assert not union["passed"] and union["reportable"] == ["a"]


@pytest.mark.skipif(not SUMMARY.exists(), reason="validation summary not generated")
def test_committed_summary_is_internally_consistent():
    s = json.loads(SUMMARY.read_text())
    p = s["parity"]
    assert (
        p["runs"] == 648 and p["exact_row_parity"] and p["outcome_parity"] and p["leakage_passed"]
    )
    assert p["downstream_execution_failures"] == 0 and p["calibration"]["passed"]
    assert s["counts"]["recordings"] == 6
    # No released policy may succeed where a bound certifies infeasibility.
    assert all(g["successes_in_certified_infeasible"] == 0 for g in s["exp2"]["gap"])
    # Causal 1x and the reference are the same 648 executions.
    causal_1x = next(d for d in s["exp3"]["disagreement"] if d["multiplier"] == 1.0)
    assert causal_1x["both_succeed"] + causal_1x["only_causal_succeeds"] == 218


def test_sensitivity_grid_is_the_declared_grid_and_contains_the_reference():
    from experiments.validation import contracts

    settings = contracts.sweep_settings()
    assert len(settings) == contracts.CONFIG["sensitivity_map"]["settings"] == 144
    assert len({tuple(s.values()) for s in settings}) == 144
    case = json.loads((BUNDLE / "study.json").read_text())["cases"]["sector_revisit/sector_3"]
    base = contracts.contract_for(case, 0.95, "confirm").document()
    ref = next(s for s in settings if tuple(s.values()) == (2, 2, 2, True, 44.0))
    assert contracts.sweep_document(base, ref) == base
    loose = next(s for s in settings if tuple(s.values()) == (1, 1, None, False, 36.0))
    doc = contracts.sweep_document(base, loose)
    phase = doc["mission"]["phases"][0]
    assert (phase["min_hits"], phase["min_evidence_span_s"], phase["required_recall"]) == (
        1,
        0.0,
        1 / 3,
    )
    assert phase["max_false_components"] == float("inf")
    assert doc["reporting"]["min_report_recall"] == 0.0
    assert doc["evaluator_target_ids"] == base["evaluator_target_ids"]


@pytest.mark.skipif(not SUMMARY.exists(), reason="validation summary not generated")
def test_sensitivity_receipt_reproduces_declared_anchor_points():
    s = json.loads(SUMMARY.read_text())["sensitivity"]
    assert s["settings"] == 144 and s["receipt"]["runs"] == 648
    assert all(s["receipt"]["consistency"].values())
