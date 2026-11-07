"""Audit the claims supported by a completed routing cohort, without retuning it.

This is an evaluator-only release report. It refuses incomplete or changed
validated artifacts, distinguishes executions from independent scenes, and never
treats a relaxed feasibility bound as a realizable routing policy.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.mission_routing_v2.flights import verify
from experiments.mission_routing_v2.train import checked_manifest
from experiments.mission_routing_v2_physics import terrain_geometry, validate_alignment
from experiments.mission_routing_v2_route_validation import audit


def claim_audit(rows, ablations, feasibility):
    """Compute descriptive paired claims; no significance or causal gain inference."""
    primary = [r for r in rows if r["power_w"] == 15]
    keys = sorted({(r["flight"], r["mission"]) for r in primary})
    groups = []
    for flight, mission in keys:
        group = [r for r in primary if (r["flight"], r["mission"]) == (flight, mission)]
        impossible = feasibility[flight]["missions"][mission]["latency_certified_infeasible"]
        if impossible and any(r["full_success"] for r in group):
            raise ValueError("Full success contradicts a certified infeasibility bound")
        groups.append(
            dict(
                flight=flight,
                mission=mission,
                full_success_discriminates=len({r["full_success"] for r in group}) > 1,
                all_policies_fail=not any(r["full_success"] for r in group),
                phases_range=[
                    min(r["phases_succeeded"] for r in group),
                    max(r["phases_succeeded"] for r in group),
                ],
                report_recall_range=[
                    min(r["report_recall"] for r in group),
                    max(r["report_recall"] for r in group),
                ],
                optimistic_bound_certifies_infeasible=impossible,
            )
        )
    effects = {}
    for name in ("no_history", "no_context", "no_intent"):
        selected = [a for a in ablations if a["ablation"] == name and a["power_w"] == 15]
        if len(selected) != len(keys):
            raise ValueError("Incomplete primary ablation grid")
        effects[name] = dict(
            paired_cases=len(selected),
            changed_schedules=sum(not a["identical_schedule"] for a in selected),
            schedule_change_proves_benefit=False,
        )
    outcomes = []
    for flight, mission in keys:
        group = {
            r["policy"]: r for r in primary if (r["flight"], r["mission"]) == (flight, mission)
        }
        if "intent_history" not in group:
            continue
        full = group["intent_history"]
        for name in ("no_history", "no_context", "no_intent"):
            other = group[name]
            outcomes.append(
                dict(
                    flight=flight,
                    mission=mission,
                    ablation=name,
                    full_success_delta=int(full["full_success"]) - int(other["full_success"]),
                    successful_phases_delta=full["phases_succeeded"] - other["phases_succeeded"],
                    report_recall_delta=full["report_recall"] - other["report_recall"],
                    final_battery_delta=full["final_battery"] - other["final_battery"],
                )
            )
    return dict(
        primary_executions=len(primary),
        primary_successes=sum(r["full_success"] for r in primary),
        paired_cases=len(keys),
        full_success_discriminating_cases=sum(g["full_success_discriminates"] for g in groups),
        all_fail_cases=sum(g["all_policies_fail"] for g in groups),
        certified_infeasible_cases=sum(g["optimistic_bound_certifies_infeasible"] for g in groups),
        failure_counts=dict(Counter(c for r in primary for c in r["failed_constraints"])),
        ablations=effects,
        paired_ablation_outcomes=outcomes,
        groups=groups,
    )


def validated_rows(root, protocol, validation):
    """Fail closed on missing grid cells, stale checksums or changed run artifacts."""
    expected_files = {
        f"summary_{flight}_{power}.json"
        for flight in protocol["cases"]
        for power in protocol["active_cpu_power_sensitivity_w"]
    }
    if not validation["complete"] or set(validation["source_summaries_sha256"]) != expected_files:
        raise ValueError("Incomplete validated summary grid")
    rows = []
    for filename in sorted(expected_files):
        if sha(root / filename) != validation["source_summaries_sha256"][filename]:
            raise ValueError("Changed validated summary")
        summary = read(root / filename)
        if not summary["complete"] or summary["protocol_sha256"] != sha(root / "protocol.json"):
            raise ValueError("Wrong summary protocol")
        rows.extend(summary["cases"])
    actual = [(r["flight"], r["mission"], r["power_w"], r["policy"]) for r in rows]
    expected = {
        (f, m, w, p)
        for f in protocol["cases"]
        for m in protocol["missions"]
        for w in protocol["active_cpu_power_sensitivity_w"]
        for p in protocol["policies"]
    }
    if len(actual) != len(expected) or set(actual) != expected or validation["cases"] != len(rows):
        raise ValueError("Missing or duplicated validated policy cases")
    for row in rows:
        for name, key in (
            ("outcome.json", "outcome_sha256"),
            ("execution/events.json", "events_sha256"),
            ("execution/results.json", "results_sha256"),
        ):
            if sha(root / row["path"] / name) != row[key]:
                raise ValueError("Changed validated run artifact")
    return rows


def witness_audit(root, protocol):
    """Optional evaluator witnesses never enter primary policy execution counts."""
    folder = root / "witnesses"
    if not folder.exists():
        return None
    plan = read(folder / "protocol.json")
    summary = read(folder / "summary.json")
    if not summary["complete"] or summary["protocol_sha256"] != sha(folder / "protocol.json"):
        raise ValueError("Incomplete witness search")
    if plan["protocol_sha256"] != sha(root / "protocol.json") or not plan["gt_aware"]:
        raise ValueError("Wrong witness protocol")
    if plan["source_sha256"] != sha(Path(__file__).with_name("mission_routing_v2_witness.py")):
        raise ValueError("Changed witness implementation")
    if plan["search_sha256"] != sha(Path(__file__).with_name("lake_hindsight_search.py")):
        raise ValueError("Changed witness search")
    rows = summary["cases"]
    keys = [(r["flight"], r["mission"]) for r in rows]
    expected = {(f, m) for f in protocol["cases"] for m in protocol["missions"]}
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Incomplete witness grid")
    for row in rows:
        destination = folder / row["flight"] / row["mission"]
        if sha(destination / "search.json") != row["search_sha256"]:
            raise ValueError("Changed witness search result")
        if "outcome_sha256" in row:
            for name, key in (
                ("outcome.json", "outcome_sha256"),
                ("execution/events.json", "events_sha256"),
                ("execution/results.json", "results_sha256"),
            ):
                if sha(destination / "replay" / name) != row[key]:
                    raise ValueError("Changed witness replay")
            outcome = read(destination / "replay/outcome.json")
            if outcome["mission_success"] != row["full_contract_witness"]:
                raise ValueError("False witness summary")
        elif row["full_contract_witness"]:
            raise ValueError("Full witness has no paid replay")
    return dict(
        gt_aware=True,
        deployable_policy=False,
        excluded_from_policy_grid=True,
        witnessed_cases=sum(r["full_contract_witness"] for r in rows),
        cases=len(rows),
        summary_sha256=sha(folder / "summary.json"),
    )


def run(root):
    p = verify(root)
    for field, key, filename in (
        ("physics_alignment", "implementation_sha256", "mission_routing_v2_physics.py"),
        ("cohort_design", "designer_sha256", "mission_routing_v2_corridor.py"),
        ("route_revision", "reviser_sha256", "mission_routing_v2_route_validation.py"),
    ):
        if field in p and p[field][key] != sha(Path(__file__).with_name(filename)):
            raise ValueError("Changed physical cohort derivation: " + filename)
    validation = read(root / "validation.json")
    if validation["validator_sha256"] != p["source_sha256"]["validate.py"]:
        raise ValueError("Unexpected validator implementation")
    rows = validated_rows(root, p, validation)
    route = audit(root)
    if not route["all_passed"]:
        raise ValueError("Physical route failed")
    geometry = terrain_geometry(Path(p["assets"]) / "harmonic_terrain/model.sdf")
    aligned = all(
        geometry["visual_" + k] == geometry["collision_" + k] for k in ("size", "position", "uri")
    )
    if "physics_alignment" in p:
        validate_alignment(Path(p["assets"]))
        if (
            sha(Path(p["assets"]) / "manifest.json")
            != p["physics_alignment"]["derived_assets_manifest_sha256"]
        ):
            raise ValueError("Changed corrected assets")
    feasibility = read(root / "feasibility_diagnostics.json")
    policies = read(root / "policy_diagnostics.json")
    if feasibility["protocol_sha256"] != sha(root / "protocol.json") or policies[
        "validation_sha256"
    ] != sha(root / "validation.json"):
        raise ValueError("Stale diagnostics")
    captures, calls = 0, 0
    for flight in p["cases"]:
        folder = root / flight / "predictions"
        manifest = checked_manifest(folder, root / "protocol.json")
        if manifest["recording_sha256"] != sha(root / flight / "recording/recording.json"):
            raise ValueError("Changed physical recording")
        diagnostic = feasibility["flights"][flight]
        if diagnostic["source_manifest_sha256"] != sha(folder / "manifest.json") or diagnostic[
            "outcomes_sha256"
        ] != sha(root / flight / "diagnostic_outcomes.json"):
            raise ValueError("Changed feasibility evidence")
        captures += manifest["capture_count"]
        calls += read(folder / "completion.json")["actual_neural_calls"]
    claims = claim_audit(rows, policies["ablations"], feasibility["flights"])
    witnesses = witness_audit(root, p)
    result = dict(
        schema_version="mission_routing_v2_release_v1",
        scope=p["scope"],
        protocol_sha256=sha(root / "protocol.json"),
        implementation_sha256=sha(Path(__file__)),
        evidence_sha256={
            name: sha(root / name)
            for name in (
                "validation.json",
                "physical_route_validation.json",
                "feasibility_diagnostics.json",
                "policy_diagnostics.json",
            )
        },
        total_executions=len(rows),
        physical_captures=captures,
        actual_neural_calls=calls,
        distinct_layouts=len({c["lineage"]["layout"] for c in p["cases"].values()}),
        distinct_routes=len({c["lineage"]["route"] for c in p["cases"].values()}),
        visual_collision_aligned=aligned,
        fixed_flight_execution_audited=True,
        new_map_generalization_established=False,
        onboard_energy_measured=False,
        population_policy_superiority_established=False,
        real_mission_log_replay=False,
        claims=claims,
        evaluator_witnesses=witnesses,
    )
    save(root / "benchmark_status.json", result)
    lines = [
        "# Benchmark evidence and remaining gates",
        "",
        f"{len(rows)} validated policy executions reuse {captures} physical captures and "
        f"{calls} actual neural inference calls. They are not {len(rows)} independent trials.",
        "",
        f"This cohort contains {result['distinct_layouts']} distinct target layout(s) and "
        f"{result['distinct_routes']} distinct physical route(s) on one known visual map. "
        "Day/dusk recordings of shared geometry are paired lighting conditions.",
        "",
        f"Visual/collision heightfields aligned: **{aligned}**. "
        "Original-map legacy runs must not be relabelled as corrected-map experiments.",
        "",
        f"Primary condition (15 W assumed compute power): **{claims['primary_successes']}/"
        f"{claims['primary_executions']} full mission successes**. "
        f"Full success distinguishes policies in **{claims['full_success_discriminating_cases']}/"
        f"{claims['paired_cases']}** matched flight/intent cases. "
        "This is a descriptive result, not a success-rate estimate over a mission population.",
        "",
        "## Matched operating conditions",
        "",
        "| Flight | Intent | Success distinguishes policies | All policies fail | "
        "Certified infeasible | Phase range | Report recall range |",
        "|---|---|---|---|---|---|---|",
    ]
    for g in claims["groups"]:
        lines.append(
            f"| {g['flight']} | {g['mission']} | {g['full_success_discriminates']} | "
            f"{g['all_policies_fail']} | {g['optimistic_bound_certifies_infeasible']} | "
            f"{g['phases_range']} | {g['report_recall_range']} |"
        )
    if witnesses is not None:
        lines += [
            "",
            f"GT-aware paid schedules establish full-contract feasibility in "
            f"**{witnesses['witnessed_cases']}/{witnesses['cases']}** cases. "
            "These are evaluator-only witnesses, excluded from all policy counts and "
            "rankings. Search failures remain inconclusive. [Witness details](WITNESSES.md).",
        ]
    lines += [
        "",
        "## Input ablations",
        "",
        "| Removed input | Changed schedules | Primary paired cases |",
        "|---|---:|---:|",
    ]
    for name, effect in claims["ablations"].items():
        lines.append(f"| {name} | {effect['changed_schedules']} | {effect['paired_cases']} |")
    lines += [
        "",
        "Different schedules establish input use, not improved mission outcomes. "
        "Identical schedules establish no incremental behavioral effect in these cases.",
        "",
        "Each delta below is full intent/history policy minus its ablation, at 15 W. "
        "Positive values favor the full policy for that metric only; tradeoffs remain explicit.",
        "",
        "| Flight | Intent | Ablation | Success delta | Phase delta | Report recall delta | "
        "Battery fraction delta |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for d in claims["paired_ablation_outcomes"]:
        lines.append(
            f"| {d['flight']} | {d['mission']} | {d['ablation']} | "
            f"{d['full_success_delta']:+d} | {d['successful_phases_delta']:+d} | "
            f"{d['report_recall_delta']:+.3f} | {d['final_battery_delta']:+.4f} |"
        )
    lines += [
        "",
        "## What remains",
        "",
        "1. Separate witnessed-feasible operating conditions from certified-infeasible stress "
        "cases. Passing an optimistic bound is not a full mission witness. Keep negative results.",
        "2. Evaluate complementary models and causal history features on development data, "
        "then freeze a new version before new evaluation flights. Do not tune on this cohort.",
        "3. Add independent maps, layouts and trajectories with documented real-mission "
        "provenance. Split and quantify uncertainty by mission/scene, not by correlated frames.",
        "4. Replace declared energy, memory and load assumptions with onboard measurements; "
        "stress pose error, moving targets and sensing failures.",
        "5. Implement faithful published comparators before making method-superiority claims. "
        "Current lightweight comparators are adaptations, not paper reproductions.",
        "",
        "The benchmark is an audited fixed-flight simulation harness. Population generalization, "
        "real-mission-log replay and deployed energy performance remain unestablished.",
        "",
        "[All policy results](REPORT.md) · [Feasibility](FEASIBILITY.md) · "
        "[Ablations](POLICY_DIAGNOSTICS.md) · [Machine-readable status](benchmark_status.json)",
        "",
    ]
    (root / "BENCHMARK_STATUS.md").write_text("\n".join(lines))
    print(
        {
            k: claims[k]
            for k in (
                "primary_successes",
                "primary_executions",
                "full_success_discriminating_cases",
                "all_fail_cases",
                "certified_infeasible_cases",
            )
        }
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
