"""Predeclared matched effects and two-factor interactions; no outcome selection."""

from __future__ import annotations

import itertools
from collections import defaultdict
from pathlib import Path

from experiments.mission_joint_constraints import FACTOR_KEYS, expected_local_groups, verify
from experiments.mission_measured_cost_study_v2 import require
from experiments.mission_runtime_profile import digest, read, save, sha

VARYING = ("mission", "initial_battery_frac", "link", "onboard_latency_scale")
LEVELS = dict(
    mission="missions",
    initial_battery_frac="batteries",
    link="links",
    onboard_latency_scale="onboard_latency_scales",
)
METRICS = (
    "full_success",
    "phases_succeeded",
    "expected_transport_timeouts",
    "report_recall",
    "final_battery",
    "communication_mb",
)
ABLATION_CONTROLS = (
    "mission_evidence_network_blind",
    "mission_evidence_local_only",
    "snapshot_quality_2s",
)


def lookup_key(row):
    return tuple(row[k] for k in (*FACTOR_KEYS, "policy"))


def behavior(events):
    return [(e["observation_id"], e["execution"]["config_id"]) for e in events]


def behavior_difference(left, right):
    a, b = dict(left), dict(right)
    shared = sorted(a.keys() & b.keys())
    return dict(
        action_schedule_changed=left != right,
        capture_schedule_changed=list(a) != list(b),
        matched_captures=len(shared),
        matched_capture_action_changes=sum(a[k] != b[k] for k in shared),
    )


def matched_effects(plan, rows, actions):
    """Hold every other factor, including recording and policy, exactly fixed."""
    pairs, interactions = [], []
    lookup = {lookup_key(r): r for r in rows}
    require(len(lookup) == len(rows) == plan["expected_replays"], "Incomplete matched grid")
    for factor in VARYING:
        for low, high in itertools.combinations(plan["factors"][LEVELS[factor]], 2):
            for left in rows:
                if left[factor] != low:
                    continue
                right = lookup[lookup_key(dict(left, **{factor: high}))]
                pairs.append(
                    dict(
                        factor=factor,
                        low=low,
                        high=high,
                        held={k: left[k] for k in (*FACTOR_KEYS, "policy") if k != factor},
                        left_path=left["path"],
                        right_path=right["path"],
                        delta={m: right[m] - left[m] for m in METRICS},
                        **behavior_difference(actions[left["path"]], actions[right["path"]]),
                        full_success_changed=left["full_success"] != right["full_success"],
                        phase_count_changed=left["phases_succeeded"] != right["phases_succeeded"],
                    )
                )
    for first, second in itertools.combinations(VARYING, 2):
        for a0, a1 in itertools.combinations(plan["factors"][LEVELS[first]], 2):
            for b0, b1 in itertools.combinations(plan["factors"][LEVELS[second]], 2):
                for row in rows:
                    if row[first] != a0 or row[second] != b0:
                        continue
                    quartet = [
                        lookup[lookup_key(dict(row, **{first: a, second: b}))]
                        for a, b in ((a0, b0), (a1, b0), (a0, b1), (a1, b1))
                    ]
                    r00, r10, r01, r11 = quartet
                    interactions.append(
                        dict(
                            factors=[first, second],
                            levels=[[a0, a1], [b0, b1]],
                            held={
                                k: row[k]
                                for k in (*FACTOR_KEYS, "policy")
                                if k not in (first, second)
                            },
                            paths=[r["path"] for r in quartet],
                            difference_in_differences={
                                m: (r11[m] - r01[m]) - (r10[m] - r00[m]) for m in METRICS
                            },
                        )
                    )
    return pairs, interactions


def aggregate(rows):
    return dict(
        conditions=len(rows),
        full_successes=sum(r["full_success"] for r in rows),
        phases_succeeded=sum(r["phases_succeeded"] for r in rows),
        phase_denominator=4 * len(rows),
        decisions=sum(r["calls"] for r in rows),
        expected_timeouts=sum(r["expected_transport_timeouts"] for r in rows),
        mean_report_recall=sum(r["report_recall"] for r in rows) / len(rows),
        minimum_final_battery=min(r["final_battery"] for r in rows),
        below_reserve=sum(r["final_battery"] < 0.2 for r in rows),
    )


def policy_contrasts(rows, actions):
    """Network-aware minus each predeclared control within the identical condition."""
    lookup = {lookup_key(r): r for r in rows}
    pairs = []
    for aware in rows:
        if aware["policy"] != "mission_evidence_network":
            continue
        for control in ABLATION_CONTROLS:
            key = lookup_key(dict(aware, policy=control))
            if key not in lookup:
                continue
            other = lookup[key]
            pairs.append(
                dict(
                    policy="mission_evidence_network",
                    control=control,
                    held={k: aware[k] for k in FACTOR_KEYS},
                    aware_path=aware["path"],
                    control_path=other["path"],
                    delta={m: aware[m] - other[m] for m in METRICS},
                    **behavior_difference(actions[other["path"]], actions[aware["path"]]),
                )
            )
    return pairs


def finalize(root):
    root = Path(root)
    require(not (root / "completion.json").exists(), "Preserve completed analysis")
    plan = verify(root)
    index, audit = read(root / "run_index.json"), read(root / "validation.json")
    require(
        audit["complete"]
        and audit["complete_execution_grid"]
        and audit["executions"] == len(index["cases"]) == plan["expected_replays"]
        and audit["fixed_local_invariance_groups"] == expected_local_groups(plan),
        "Incomplete independently audited factorial grid",
    )
    require(
        audit["plan_sha256"] == index["plan_sha256"] == sha(root / "plan.json")
        and audit["run_index_sha256"] == sha(root / "run_index.json"),
        "Changed audit binding",
    )
    auditor = Path(__file__).with_name("mission_joint_constraints_audit.py")
    require(
        audit["validator_sha256"]
        == sha(auditor)
        == plan["implementation_sha256"][str(auditor.resolve())],
        "Unfrozen auditor",
    )
    for field in (
        "canonical_outcomes_verified",
        "public_states_and_resource_flags_reconstructed",
        "raw_selected_mask_scores_reconstructed",
        "report_identity_support_reconstructed",
        "per_event_cost_clock_partial_bytes_verified",
        "no_mission_runner_called",
    ):
        require(audit[field] is True, "Incomplete reconstruction: " + field)
    for relative, expected in audit["source_output_sha256"].items():
        require(sha(root / relative) == expected, "Changed audited output")
    rows, actions, outcomes, enriched = index["cases"], {}, {}, []
    for row in rows:
        destination = root / row["path"]
        events = read(destination / "execution/events.json")
        outcome = read(destination / "outcome.json")
        require(
            sha(destination / "execution/events.json") == row["events_sha256"]
            and sha(destination / "outcome.json") == row["outcome_sha256"],
            "Changed outcome",
        )
        actions[row["path"]] = behavior(events)
        outcomes[row["path"]] = outcome
        enriched.append(
            dict(
                row,
                remote_decisions=sum(e["placement"] == "remote" for e in events),
                action_schedule_sha256=digest(actions[row["path"]]),
                strict_original_success=outcome["strict_original_endpoint_success"],
            )
        )
    pairs, interactions = matched_effects(plan, rows, actions)
    ablations = policy_contrasts(rows, actions)
    save(
        root / "matched_contrasts.json",
        dict(
            plan_sha256=sha(root / "plan.json"),
            run_index_sha256=sha(root / "run_index.json"),
            direction="high-minus-low in declared level order; "
            "intent names have no numeric ordering",
            pair_count=len(pairs),
            pairs=pairs,
            interaction_count=len(interactions),
            interactions=interactions,
            policy_contrast_count=len(ablations),
            policy_contrasts=ablations,
            no_independent_trial_or_population_inference=True,
        ),
    )
    policy_rows = []
    for policy in plan["policies"]:
        selected = [r for r in enriched if r["policy"] == policy]
        margins = {}
        for factor in VARYING:
            margins[factor] = [
                dict(level=level, **aggregate([r for r in selected if r[factor] == level]))
                for level in plan["factors"][LEVELS[factor]]
            ]
        policy_rows.append(
            dict(
                policy=policy,
                **aggregate(selected),
                margins=margins,
                remote_decisions=sum(r["remote_decisions"] for r in selected),
            )
        )
    contrasts = []
    for factor in VARYING:
        for policy in plan["policies"]:
            selected = [p for p in pairs if p["factor"] == factor and p["held"]["policy"] == policy]
            contrasts.append(
                dict(
                    factor=factor,
                    policy=policy,
                    pairs=len(selected),
                    action_schedule_changed=sum(p["action_schedule_changed"] for p in selected),
                    capture_schedule_changed=sum(p["capture_schedule_changed"] for p in selected),
                    full_success_changed=sum(p["full_success_changed"] for p in selected),
                    phase_count_changed=sum(p["phase_count_changed"] for p in selected),
                    cancellation_count_changed=sum(
                        p["delta"]["expected_transport_timeouts"] != 0 for p in selected
                    ),
                    matched_capture_action_changes=sum(
                        p["matched_capture_action_changes"] for p in selected
                    ),
                )
            )
    grouped = defaultdict(list)
    for row in enriched:
        grouped[row["cell"]].append(row)
    infeasible = [
        dict(
            cell=cell,
            any_tested_policy_succeeds=any(r["full_success"] for r in members),
            known_sensor_infeasible=plan["cases"][cell]["known_sensor_infeasible"],
            **{k: members[0][k] for k in FACTOR_KEYS},
        )
        for cell, members in grouped.items()
    ]
    endpoint_changes = [
        r["path"] for r in enriched if r["full_success"] != r["strict_original_success"]
    ]
    summary = dict(
        schema_version="mission_joint_constraints_summary_v1",
        complete=True,
        executions=len(rows),
        condition_cells=len(grouped),
        policies=policy_rows,
        decisions=audit["events"],
        reports=audit["reports"],
        expected_transport_timeouts=audit["canceled_predictions"],
        fixed_local_invariance_groups=audit["fixed_local_invariance_groups"],
        remote_compute_invariance_groups=audit["remote_compute_invariance_groups"],
        matched_pairs=len(pairs),
        two_factor_interactions=len(interactions),
        matched_policy_contrasts=len(ablations),
        policy_ablation_summary=[
            dict(
                control=control,
                conditions=sum(p["control"] == control for p in ablations),
                full_success_wins=sum(
                    p["control"] == control and p["delta"]["full_success"] > 0 for p in ablations
                ),
                full_success_losses=sum(
                    p["control"] == control and p["delta"]["full_success"] < 0 for p in ablations
                ),
                phase_count_wins=sum(
                    p["control"] == control and p["delta"]["phases_succeeded"] > 0
                    for p in ablations
                ),
                phase_count_losses=sum(
                    p["control"] == control and p["delta"]["phases_succeeded"] < 0
                    for p in ablations
                ),
            )
            for control in ABLATION_CONTROLS
        ],
        contrast_sensitivity=contrasts,
        cells=infeasible,
        executions_complete=enriched,
        all_tested_policies_fail_cells=sum(not r["any_tested_policy_succeeds"] for r in infeasible),
        feasibility_caveat=plan["feasibility_scope"]
        + " No paid oracle witness was run for these new cells.",
        endpoint_changed_paths=endpoint_changes,
        endpoint_caveat="Recoverable transport cancellation vs original strict execution endpoint; "
        "this label difference is not a policy improvement.",
        matched_contrasts_sha256=sha(root / "matched_contrasts.json"),
        plan_sha256=sha(root / "plan.json"),
        run_index_sha256=sha(root / "run_index.json"),
        validation_sha256=sha(root / "validation.json"),
        scope=plan["scope"],
    )
    save(root / "summary.json", summary)
    lines = [
        "# Joint intent/resource factorial development study",
        "",
        plan["scope"],
        "",
        f"Completed {len(rows)} executions over {len(grouped)} matched condition cells; "
        f"{audit['events']:,} decisions and {audit['reports']} reports "
        "independently reconstructed.",
        "",
        "| Policy | Full success | Completed phases | Cancellations | Below reserve |",
        "|---|---:|---:|---:|---:|",
    ]
    for p in policy_rows:
        lines.append(
            f"| {p['policy']} | {p['full_successes']}/{p['conditions']} | "
            f"{p['phases_succeeded']}/{p['phase_denominator']} | "
            f"{p['expected_timeouts']} | {p['below_reserve']} |"
        )
    lines += [
        "",
        "## Matched one-factor effects",
        "",
        "Action/schedule changes are response sensitivity, not evidence of beneficial adaptation. "
        "All pairs hold the other factors and recording fixed. Three link levels produce three "
        "pairwise contrasts per context; these overlapping comparisons are not independent trials.",
        "",
        "| Factor | Policy | Pairs | Action/schedule changes | Success changes | Phase changes |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for c in contrasts:
        lines.append(
            f"| {c['factor']} | {c['policy']} | {c['pairs']} | "
            f"{c['action_schedule_changed']} | {c['full_success_changed']} | "
            f"{c['phase_count_changed']} |"
        )
    lines += [
        "",
        "## Complete interactions and controls",
        "",
        f"The machine-readable matched_contrasts.json retains all {len(pairs)} one-factor pairs "
        f"and {len(interactions)} two-factor differences in differences, with every source path. "
        f"It also retains {len(ablations)} within-cell network-aware minus link-blind, "
        "local-only and snapshot-quality policy contrasts. "
        "No interaction or policy was selected after observing its outcome.",
        "",
        f"All {audit['fixed_local_invariance_groups']} fixed-local controls are invariant across "
        f"links. All {audit['remote_compute_invariance_groups']} fixed-remote controls are "
        "invariant across onboard compute factors. This checks placement isolation.",
        "",
        f"All tested policies fail in {summary['all_tested_policies_fail_cells']}/{len(grouped)} "
        "condition cells. Strict inspection is already sensor-certified infeasible in the "
        "source development protocol; all strict cells are integration/stress controls. Other "
        "all-failure cells lack a paid witness and are not certified infeasible. Intent changes "
        "also change "
        "requirements, and cannot alone demonstrate beneficial intent use.",
        "",
        f"The recoverable endpoint changes {len(endpoint_changes)} labels relative to the "
        "original strict endpoint. These are separately enumerated, not routing improvements.",
        "",
        "Four model weight families and nine perception configurations remain the model "
        "pool. This experiment does not add navigation, planning, language understanding or "
        "other task heads. Network/service, power and compute scaling remain declared models.",
    ]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    save(
        root / "completion.json",
        dict(
            complete=True,
            executions=len(rows),
            decisions=audit["events"],
            plan_sha256=sha(root / "plan.json"),
            run_index_sha256=sha(root / "run_index.json"),
            validation_sha256=sha(root / "validation.json"),
            summary_sha256=sha(root / "summary.json"),
            matched_contrasts_sha256=sha(root / "matched_contrasts.json"),
            report_sha256=sha(root / "REPORT.md"),
        ),
    )
    return summary
