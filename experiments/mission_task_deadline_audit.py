"""Post hoc receipt audit for the frozen per-invocation deadline study.

This checker does not import its executor, deadline contract, policy, scheduler,
or score reconstruction. It independently checks arithmetic, output gating,
selected-mask manifest identities, resource totals and report evidence guards.
It does not rerun neural inference, association geometry or the mission evaluator,
and it does not establish that the authored network/power assumptions are real.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from fractions import Fraction
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def close(actual, expected, message):
    require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8), message)


def ns(seconds):
    return round(seconds * 1_000_000_000)


def check_deadline(row, release, budget, step):
    require(row["release_ns"] == release, "Incorrect task release")
    require(row["budget_ns"] == budget, "Incorrect invocation budget")
    require(row["deadline_ns"] == release + budget, "Deadline restarted or wrong origin")
    raw = row["raw_finish_ns"]
    require(type(raw) is int and raw >= release, "Invalid actual task finish")
    usable = ((raw + step - 1) // step) * step
    require(row["usable_finish_ns"] == usable, "Incorrect usable availability rounding")
    require(row["on_time"] == (usable <= release + budget), "Incorrect inclusive deadline")
    require(row["lateness_ns"] == max(0, usable - release - budget), "Incorrect lateness")
    if row.get("committed", row.get("admitted", False)):
        require(row["on_time"], "Late output committed")
    return usable


def check_transport(job, costs):
    previous = job["start_ns"]
    radio = 0.0
    uploaded = downloaded = Fraction(0)
    durations = Counter()
    for event in job["events"]:
        start, end, phase = event["start_ns"], event["end_ns"], event["phase"]
        require(start == previous and end >= start, "Transport event gap or reversed time")
        previous = end
        duration = end - start
        durations[phase] += duration
        bits = Fraction(event["bits_numerator"], event["bits_denominator"])
        require(bits >= 0, "Negative transferred payload")
        if phase in ("upload_tx", "download_tx"):
            require(
                bits <= Fraction(duration * event["rate_bps"], 1_000_000_000),
                "Transferred bits exceed serialization capacity",
            )
            power = costs["tx_power_w"] if phase == "upload_tx" else costs["rx_power_w"]
            if phase == "upload_tx":
                uploaded += bits
            else:
                downloaded += bits
        else:
            require(bits == 0, "Bytes transferred in a waiting phase")
            power = costs["wait_power_w"]
        radio += duration / 1e9 * power
    require(previous == job["finish_ns"], "Unaccounted transport tail")
    require(job["elapsed_ns"] == previous - job["start_ns"], "Wrong transport elapsed time")
    for phase, duration in job["phase_durations_ns"].items():
        require(durations[phase] == duration, "Wrong transport phase duration")
    for direction, bits in (("uploaded", uploaded), ("downloaded", downloaded)):
        require(
            bits
            == Fraction(job[direction + "_bits_numerator"], job[direction + "_bits_denominator"]),
            "Wrong byte ledger",
        )
    require(
        uploaded <= job["requested_upload_bits"] and downloaded <= job["requested_download_bits"],
        "Extra payload bytes",
    )
    if job["prediction_eligible"]:
        require(
            job["status"] == "completed"
            and uploaded == job["requested_upload_bits"]
            and downloaded == job["requested_download_bits"],
            "Incomplete response admitted",
        )
    return float(uploaded + downloaded) / 8e6, radio


def metrics_row(row, perception=False):
    return dict(
        attempts=1,
        on_time=int(
            row["task_deadline_met"] if perception else row["on_time"] and row["committed"]
        ),
        deadline_misses=int(row["deadline_missed"] if perception else not row["on_time"]),
        discarded_outputs=int(not row["admitted"] if perception else not row["committed"]),
        lateness_ns=row["lateness_ns"],
    )


def audit_run(directory, item, plan, parent, manifest):
    events, result, outcome = (
        read(directory / (name + ".json")) for name in ("events", "results", "outcome")
    )
    intent = read(directory / "intent.json")
    scenario = read(
        Path(plan["parent_root"]) / "missions" / item["cell"] / item["mission"] / "scenario.json"
    )
    mission = scenario["mission"]
    capacity, power = mission["battery_capacity_wh"] * 3600, mission["flight_power_w"]
    interval, duration = ns(mission["observation_interval_s"]), ns(mission["duration_s"])
    profile = next(
        p for p in plan["factors"]["deadline_profiles"] if p["name"] == item["deadline_profile"]
    )
    actions = {a["action_id"]: a for a in parent["actions_by_scale"]["1.0"]}
    costs = parent["policy_costs"]
    require(
        result["status"] == "completed" and result["error"] is None,
        "Unsuccessful execution artifact",
    )
    current = None
    stage_residents = {}
    committed_captures = set()
    committed_sources = {}
    expected_index = 0
    nonflight = communication = 0.0
    counts = Counter()
    report_ids = []
    metrics = {
        task: Counter(
            dict(attempts=0, on_time=0, deadline_misses=0, discarded_outputs=0, lateness_ns=0)
        )
        for task in ("perception", "association", "verification")
    }
    all_skipped = []
    last_finish = 0
    for event in events:
        index, capture = event["observation_id"], ns(event["capture_time_s"])
        require(
            index == expected_index and capture == index * interval,
            "Reused, omitted or busy capture",
        )
        p, selected = event["perception"], event["selected_action_id"]
        action = actions[selected] if selected else None
        charge = p["costs"]
        step = p["step_ns"]
        finish = check_deadline(p, capture, profile["perception_ns"], step)
        require(finish == event["perception_completed_ns"], "Wrong perception availability")
        preparation = ns(costs["encode_s"])
        require(charge["preparation_latency_ns"] == preparation, "Wrong preparation time")
        close(
            charge["preparation_energy_j"],
            costs["encode_s"] * costs["codec_power_w"],
            "Wrong preparation energy",
        )
        require(
            charge["routing_latency_ns"] == ns(mission["router_latency_s"]), "Wrong router latency"
        )
        close(charge["routing_energy_j"], mission["router_energy_j"], "Wrong router energy")
        changed = selected is not None and selected != current
        require(p["reset_slot"] == changed, "Wrong perception reset")
        require(
            charge["load_latency_ns"] == (ns(action["load_latency_s"]) if changed else 0),
            "Wrong load latency",
        )
        close(
            charge["load_energy_j"], action["load_energy_j"] if changed else 0, "Wrong load energy"
        )
        decision = capture + preparation
        dispatch = decision + charge["routing_latency_ns"] + charge["load_latency_ns"]
        require(
            p["decision_ns"] == decision and p["dispatch_ns"] == dispatch,
            "Incorrect task overhead clock",
        )
        expected_battery = max(
            0,
            mission["initial_battery_frac"]
            - (decision / 1e9 * power + nonflight + charge["preparation_energy_j"]) / capacity,
        )
        close(event["battery_at_decision"], expected_battery, "Wrong causal battery state")
        require(event["public_link"]["timestamp_ns"] == decision, "Wrong current link time")
        current_link = next(
            s for s in parent["traces"][item["link"]] if s["start_ns"] <= decision < s["end_ns"]
        )
        require(
            event["public_link"]
            == dict(
                timestamp_ns=decision,
                upload_bps=0 if current_link["outage"] else current_link["upload_bps"],
                download_bps=0 if current_link["outage"] else current_link["download_bps"],
                rtt_ns=current_link["rtt_ns"],
                outage=current_link["outage"],
            ),
            "Current network snapshot differs from frozen trace",
        )
        decision_log = event["router_decision"]
        require(
            "evaluator_target_ids" not in json.dumps(decision_log),
            "Evaluator target labels appear in the policy decision record",
        )
        require(
            decision_log["public_decision_ns"] == decision
            and decision_log["action_id"] == selected,
            "Wrong recorded policy decision",
        )
        require(decision_log["task_deadlines"] == profile, "Policy saw another budget")
        if item["policy"] == "deadline_aware" and selected:
            choice = next(c for c in decision_log["candidates"] if c["action_id"] == selected)
            require(choice["feasible"], "Adaptive policy chose an excluded action")
        phase = next(
            p for p in intent["mission"]["phases"] if ns(p["start_s"]) <= capture < ns(p["end_s"])
        )
        require(decision_log["public_phase_id"] == phase["phase_id"], "Wrong active intent")
        require(
            decision_log["intent_requirements"]
            == {k: phase[k] for k in decision_log["intent_requirements"]},
            "Policy received another intent requirement",
        )
        byte_count = radio = 0.0
        if p["transport"] is not None:
            require(action["placement"] == "remote", "Local action dispatched network")
            byte_count, radio = check_transport(p["transport"], costs)
            require(p["transport"]["start_ns"] == dispatch, "Remote time omitted overhead")
            stop = min(
                dispatch + ns(costs["timeout_s"]),
                p["deadline_ns"],
                ns(intent["limits"]["deadline_s"]),
            )
            require(
                p["transport"]["stop_limit_ns"] == stop and p["transport"]["finish_ns"] <= stop,
                "Transport stop bound violated",
            )
            raw_finish = p["transport"]["finish_ns"] + charge["decode_latency_ns"]
        elif action and action["placement"] == "local":
            raw_finish = dispatch + ns(action["latency_s"])
            close(charge["compute_energy_j"], action["energy_j"], "Local work refunded")
        else:
            raw_finish = dispatch
        require(p["raw_finish_ns"] == raw_finish, "Wrong raw perception finish")
        close(charge["radio_energy_j"], radio, "Wrong actual radio charge")
        close(p["communication_mb"], byte_count, "Wrong actual transmitted bytes")
        outer_deadline = ns(intent["limits"]["deadline_s"])
        task_limited = p["deadline_ns"] <= outer_deadline
        canceled = bool(
            action
            and action["placement"] == "remote"
            and task_limited
            and (
                (p["transport"] is None and dispatch >= min(p["deadline_ns"], outer_deadline))
                or (p["transport"] is not None and p["transport"]["status"] == "deadline")
            )
        )
        require(
            p["deadline_missed"] == (canceled or finish > p["deadline_ns"]),
            "Exact-bound cancellation omitted from deadline misses",
        )
        arrived = bool(
            action
            and (
                action["placement"] == "local"
                or (p["transport"] is not None and p["transport"]["prediction_eligible"])
            )
        )
        require(p["output_arrived"] == arrived, "Undelivered selected output fabricated")
        decode = bool(action and action["placement"] == "remote" and arrived)
        require(
            charge["decode_latency_ns"] == (ns(costs["decode_s"]) if decode else 0),
            "Wrong paid response decode time",
        )
        close(
            charge["decode_energy_j"],
            costs["decode_s"] * costs["codec_power_w"] if decode else 0,
            "Wrong paid response decode energy",
        )
        require(
            p["admitted"]
            == (selected is not None and p["output_arrived"] and p["status"] == "completed"),
            "Wrong output admission",
        )
        require(p["prediction_cache_read"] == p["admitted"], "Discarded cache output read")
        require(
            p["task_deadline_met"] == (p["admitted"] and p["on_time"]),
            "Failed output counted as deadline met",
        )
        if selected:
            metrics["perception"].update(metrics_row(p, True))
            current = selected
        require(event["execution"]["success"] == p["admitted"], "Unavailable mask scored")
        if p["admitted"]:
            cached = manifest["models"][action["source_config_id"]]["predictions"]
            require(
                event["selected_mask_sha256"] == cached[event["source_rgb_key"]]["mask_sha256"],
                "Wrong selected mask",
            )
            counts["admitted_selected_masks"] += 1
        else:
            require(
                event["selected_mask_sha256"] is None
                and event["pipeline"] is None
                and event["emitted_reports"] == []
                and event["execution"]["predicted_pixels"] == 0,
                "Unavailable perception leaked downstream",
            )
            require(event["score"]["matched_target_ids"] == [], "Discarded evidence scored")
        paid = sum(
            v for k, v in charge.items() if k.endswith("energy_j") and k != "charged_energy_j"
        )
        close(charge["charged_energy_j"], paid, "Wrong perception energy sum")
        nonflight += paid
        communication += byte_count
        pipeline = event["pipeline"]
        if pipeline:
            require(p["admitted"], "Pipeline without admitted perception")
            require(
                mission["initial_battery_frac"] * capacity - finish / 1e9 * power - nonflight > 0,
                "New downstream invocation released after battery exhaustion",
            )
            stage_finish, pipeline_paid = finish, 0.0
            predecessor = True
            for stage in pipeline["stages"]:
                task = stage["task"]
                if not stage["dispatched"]:
                    require(
                        not stage["committed"] and stage["costs"]["charged_energy_j"] == 0,
                        "Undispatched downstream work committed or charged",
                    )
                    continue
                require(predecessor, "Downstream released by uncommitted predecessor")
                require(
                    mission["initial_battery_frac"] * capacity
                    - stage_finish / 1e9 * power
                    - nonflight
                    - pipeline_paid
                    > 0,
                    "New stage released after battery exhaustion",
                )
                stage_finish = check_deadline(stage, stage_finish, profile[task + "_ns"], step)
                c = stage["costs"]
                # The frozen matrix has exactly these two authored downstream algorithms.
                descriptor = {
                    "association": ("ASSOC_TERRAIN_WORLD", 30_000_000, 0.45, 50_000_000, 0.75),
                    "verification": ("VERIFY_REPEAT2", 1_000_000, 0.015, 5_000_000, 0.075),
                }[task]
                require(
                    stage["algorithm_id"] == descriptor[0]
                    and c["service_latency_ns"] == descriptor[1],
                    "Wrong selected stage service",
                )
                close(c["service_energy_j"], descriptor[2], "Wrong selected service energy")
                require(
                    stage["raw_finish_ns"]
                    == stage["start_ns"]
                    + c["routing_latency_ns"]
                    + c["load_latency_ns"]
                    + c["service_latency_ns"],
                    "Wrong downstream full service clock",
                )
                changed_stage = stage_residents.get(task) != stage["algorithm_id"]
                require(stage["reset_slot"] == changed_stage, "Wrong downstream residency")
                require(
                    c["load_latency_ns"] == (descriptor[3] if changed_stage else 0),
                    "Wrong stage loading latency",
                )
                close(
                    c["load_energy_j"],
                    descriptor[4] if changed_stage else 0,
                    "Wrong stage loading energy",
                )
                route_ns = plan["downstream_routing_latency_ns"] if task == "association" else 0
                route_j = plan["downstream_routing_energy_j"] if task == "association" else 0
                require(c["routing_latency_ns"] == route_ns, "Wrong downstream routing time")
                close(c["routing_energy_j"], route_j, "Wrong downstream routing cost")
                stage_residents[task] = stage["algorithm_id"]
                stage_paid = c["routing_energy_j"] + c["load_energy_j"] + c["service_energy_j"]
                close(c["charged_energy_j"], stage_paid, "Wrong downstream energy sum")
                pipeline_paid += stage_paid
                metrics[task].update(metrics_row(stage))
                predecessor = stage["committed"]
                if task == "association" and predecessor:
                    committed_captures.add(capture)
                    committed_sources[capture] = action["source_config_id"]
                counts[task + "_receipts"] += 1
            require(pipeline["completed_ns"] == stage_finish, "Wrong final downstream time")
            close(pipeline["charged_energy_j"], pipeline_paid, "Wrong pipeline charge")
            nonflight += pipeline_paid
            finish = stage_finish
            require(event["emitted_reports"] == pipeline["emitted_reports"], "Report mismatch")
            if not pipeline["stages"][0]["committed"]:
                require(
                    pipeline["candidates"] == [] and not pipeline["stages"][1]["dispatched"],
                    "Late association produced a candidate or dispatched verification",
                )
            if not pipeline["stages"][-1]["committed"]:
                require(
                    pipeline["selected_rulecalls"] == 0 and pipeline["emitted_reports"] == [],
                    "Unavailable verifier performed a forward or emitted a report",
                )
            for candidate in pipeline["candidates"]:
                require(
                    candidate["current_capture_ns"] == capture,
                    "Association candidate belongs to another capture",
                )
                require(
                    all(
                        ref["capture_ns"] in committed_captures
                        and ref["source_config_id"] == committed_sources[ref["capture_ns"]]
                        for ref in candidate["evidence"]
                    ),
                    "Late association leaked history",
                )
            if event["emitted_reports"]:
                require(
                    pipeline["stages"][-1]["committed"] and predecessor,
                    "Late or missing verifier emitted report",
                )
            for report in event["emitted_reports"]:
                report_ids.append(report["report_id"])
                require(
                    ns(report["completion_time_s"]) == finish
                    and finish <= ns(min(phase["deadline_s"], intent["limits"]["deadline_s"])),
                    "Report bypassed phase or mission deadline",
                )
                captures = [ns(e["capture_s"]) for e in report["evidence"]]
                require(
                    len(captures) == len(set(captures))
                    and captures == sorted(captures)
                    and set(captures) <= committed_captures
                    and len(captures) >= phase["min_hits"]
                    and captures[-1] == capture
                    and captures[-1] - captures[0] >= ns(phase["min_evidence_span_s"])
                    and all(ns(phase["start_s"]) <= t < ns(phase["end_s"]) for t in captures),
                    "Report bypassed public repeated-evidence intent",
                )
        require(ns(event["completion_time_s"]) == finish, "Wrong public completion time")
        expected_index = max(index + 1, (finish + interval - 1) // interval)
        skipped = list(range(index + 1, min(expected_index, duration // interval)))
        require(event["skipped_observation_ids"] == skipped, "Wrong busy capture skips")
        all_skipped.extend(skipped)
        close(event["cumulative_nonflight_energy_j"], nonflight, "Nonflight ledger drift")
        energy = finish / 1e9 * power + nonflight
        close(event["cumulative_energy_j"], energy, "Flight or task energy omitted")
        close(event["cumulative_communication_mb"], communication, "Byte ledger drift")
        close(
            event["battery_after"],
            max(0, mission["initial_battery_frac"] - energy / capacity),
            "Battery ledger drift",
        )
        counts["events"] += 1
        counts["reports"] += len(event["emitted_reports"])
        counts["perception_deadline_misses"] += p["deadline_missed"]
        last_finish = finish
    require(report_ids == list(range(1, len(report_ids) + 1)), "Report IDs reused or skipped")
    require(result["processed_observations"] == len(events), "Wrong observation count")
    require(result["skipped_observation_ids"] == all_skipped, "Wrong mission skip ledger")
    require(result["final_time_s"] >= last_finish / 1e9, "Mission ends before paid work")
    close(result["cumulative_nonflight_energy_j"], nonflight, "Wrong final nonflight charge")
    close(result["flight_energy_j"], result["final_time_s"] * power, "Wrong final flight charge")
    energy = result["final_time_s"] * power + nonflight
    close(result["cumulative_energy_j"], energy, "Wrong final total energy")
    close(
        result["final_battery_frac"],
        max(0, mission["initial_battery_frac"] - energy / capacity),
        "Wrong final battery",
    )
    close(result["cumulative_communication_mb"], communication, "Wrong final actual bytes")
    require(
        outcome["task_metrics"] == {task: dict(v) for task, v in metrics.items()},
        "Deadline metrics differ from independently counted attempts",
    )
    require(
        outcome["all_dispatched_task_deadlines_met"]
        == all(v["on_time"] == v["attempts"] for v in metrics.values()),
        "Deadline compliance hides undelivered outputs",
    )
    for key, actual in (
        (
            "battery_constraint_success",
            result["final_battery_frac"] >= intent["limits"]["min_final_battery_frac"],
        ),
        (
            "communication_constraint_success",
            communication <= intent["limits"]["communication_budget_mb"],
        ),
        ("deadline_success", result["final_time_s"] <= intent["limits"]["deadline_s"]),
    ):
        require(result["constraints"][key] == actual, "Wrong resource success flag: " + key)
    return dict(counts)


def audit(root):
    root = Path(root).resolve()
    plan, index, completion = (
        read(root / name) for name in ("plan.json", "run_index.json", "completion.json")
    )
    require(completion["complete"] is True, "Finalized matrix required")
    require(
        len(index["cases"]) == len(plan["grid"]) == plan["expected_runs"],
        "Incomplete factorial grid",
    )
    for name, expected in completion["artifacts_sha256"].items():
        require(sha(root / name) == expected, "Changed completed artifact")
    parent = read(Path(plan["parent_root"]) / "plan.json")
    require(
        sha(Path(plan["parent_root"]) / "plan.json") == plan["parent_plan_sha256"],
        "Changed parent input plan",
    )
    for path, expected in plan["implementation_sha256"].items():
        require(sha(path) == expected, "Frozen implementation changed")
    manifests = {}
    counts, bindings = Counter(), {}
    for item, row in zip(plan["grid"], index["cases"], strict=True):
        require(all(row[k] == v for k, v in item.items()), "Grid order or case changed")
        directory = root / row["path"]
        require(directory.resolve().is_relative_to(root), "Run path escapes study")
        for name, expected in row["artifacts_sha256"].items():
            path = directory / name
            require(sha(path) == expected, "Changed run artifact")
            bindings[str(path.relative_to(root))] = expected
        if item["flight"] not in manifests:
            manifest_path = Path(parent["cohort"]) / item["flight"] / "predictions/manifest.json"
            require(
                sha(manifest_path)
                == read(directory / "protocol.json")["prediction_manifest_sha256"],
                "Changed prediction manifest",
            )
            manifests[item["flight"]] = read(manifest_path)
        counts.update(audit_run(directory, item, plan, parent, manifests[item["flight"]]))
    return dict(
        schema_version="task_deadline_independent_receipt_audit_v1",
        passed=True,
        runs=len(index["cases"]),
        expected_runs=plan["expected_runs"],
        counts=dict(counts),
        plan_sha256=sha(root / "plan.json"),
        completion_sha256=sha(root / "completion.json"),
        audit_source_path=str(Path(__file__).resolve()),
        audit_source_sha256=sha(__file__),
        posthoc_validation=True,
        frozen_source_or_outcomes_changed=False,
        scope="Independent receipt arithmetic, output gating, mask-manifest identity, "
        "resource totals, intent report guards and deadline counts. No imports from the "
        "new executor, deadline contract, policy, scheduler or scoring implementation.",
        not_validated="No neural rerun, complete association-state or IoU reimplementation, "
        "future-link simulator reproduction, empirical network/power validation or generalization.",
        audited_artifacts_sha256=bindings,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    report = audit(args.root)
    target = args.root / "independent_deadline_audit.json"
    with target.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({k: report[k] for k in ("passed", "runs", "counts")}, indent=2))


if __name__ == "__main__":
    main()
