"""Independent reconstruction of the prospective virtual-link routing study.

The audited transport algorithm is shared with the simulator and has separate
analytical tests. Policy state, endpoint scoring and paid accounting are rebuilt
from frozen inputs, never by rerunning the mission runner or trusting logged costs.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

from experiments.mission_network_emulation import emulate_job


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_transport(trace, request, logged):
    """Recompute transport, then independently conserve chronological intervals/bits."""
    expected = emulate_job(trace, **request)
    require(expected.to_dict() == logged, "Transport history differs from fixed trace/request")
    cursor, totals = expected.start_ns, {"upload_tx": Fraction(0), "download_tx": Fraction(0)}
    for event in expected.events:
        require(
            event.start_ns == cursor and event.end_ns > cursor,
            "Transport intervals overlap, have gaps, or reverse time",
        )
        cursor = event.end_ns
        if event.phase in totals:
            require(event.bits > 0 and event.rate_bps > 0, "Nonpositive transmission")
            require(
                event.bits <= Fraction(event.rate_bps * (event.end_ns - event.start_ns), 10**9),
                "Transmission exceeds segment capacity",
            )
            totals[event.phase] += event.bits
        else:
            require(event.bits == 0, "Wait/service interval carries payload")
    require(cursor == expected.finish_ns, "Transport finish differs from interval sum")
    require(
        sum(expected.phase_durations_ns.values()) == expected.elapsed_ns,
        "Transport phase durations do not partition elapsed time",
    )
    require(
        totals["upload_tx"] == expected.uploaded_bits
        and totals["download_tx"] == expected.downloaded_bits,
        "Partial payload not conserved",
    )
    require(expected.finish_ns <= expected.stop_limit_ns, "Job continued after cancellation")
    if expected.completed:
        require(
            expected.uploaded_bits == request["upload_bits"]
            and expected.downloaded_bits == request["download_bits"],
            "Incomplete response admitted",
        )
    else:
        require(
            not expected.prediction_eligible and expected.completion_ns is None,
            "Canceled response admitted",
        )
    return expected


def radio_joules(transport, upload_w, download_w, idle_w):
    """Independent phase integration of configured terminal radio power."""
    upload = sum(e.end_ns - e.start_ns for e in transport.events if e.phase == "upload_tx")
    download = sum(e.end_ns - e.start_ns for e in transport.events if e.phase == "download_tx")
    idle = transport.elapsed_ns - upload - download
    return upload / 1e9 * upload_w + download / 1e9 * download_w + idle / 1e9 * idle_w


def check_cancelled_prediction(delivered, reports, tracker_before, tracker_after):
    if not delivered:
        require(not reports, "Canceled output produced a report")
        require(tracker_before == tracker_after, "Canceled output changed tracking state")


def check_public_state(actual, reconstructed):
    require(actual == reconstructed, "Complete current-only public state differs")


def complete_grid(plan, rows):
    keys = ("cell", "flight", "link", "mission", "policy")
    expected = [tuple(r[k] for k in keys) for r in plan["grid"]]
    actual = [tuple(r[k] for k in keys) for r in rows]
    require(len(expected) == len(set(expected)) == plan["expected_replays"], "Invalid planned grid")
    require(
        len(actual) == len(expected) and set(actual) == set(expected),
        "Missing, duplicated or substituted network-study cell",
    )
    require(len({r["path"] for r in rows}) == len(rows), "Repeated execution output path")
    return len(expected)


def local_invariance(entries):
    """Every fixed local selector must reproduce all semantics across three links."""
    grouped = {}
    for entry in entries:
        grouped.setdefault((entry["flight"], entry["policy"]), []).append(entry)
    for key, records in grouped.items():
        require(
            len(records) == 3 and len({r["link"] for r in records}) == 3,
            "Missing local network-invariance control: " + str(key),
        )
        require(
            all(r["semantic"] == records[0]["semantic"] for r in records[1:]),
            "All-local execution changed with network trace: " + str(key),
        )
    return len(grouped)


def same_fields(actual, expected, label):
    for key, value in expected.items():
        require(key in actual and actual[key] == value, f"{label}.{key} differs")


class Accounting:
    """Derive every decision state and paid event from immutable input quantities."""

    def __init__(self, scenario, tracking, costs, trace):
        self.scenario, self.tracking, self.costs, self.trace = scenario, tracking, costs, trace
        self.schedule = scenario.schedule
        self.capacity = scenario.battery_capacity_wh * 3600
        self.initial = scenario.initial_battery_frac * self.capacity
        self.energy = self.communication = self.time = 0.0
        self.index, self.current, self.processed = 0, None, 0
        self.compute = self.radio = self.routing = self.loading = self.load_time = 0.0
        self.preparation = self.tracking_j = self.memory = 0.0
        self.skipped, self.selected = [], []
        self.failures = self.timeouts = 0
        self.last_attempt = None

    def battery(self):
        return max(0.0, (self.initial - self.energy) / self.capacity)

    def advance(self, timestamp):
        new = (timestamp - self.schedule.epoch_ns) / 1e9
        require(new >= self.time, "Reconstructed clock moved backwards")
        self.energy += (new - self.time) * self.scenario.flight_power_w
        self.time = new

    def state(self):
        from missionbench.schemas.runtime_state import EvidenceSummary, RuntimeState
        from experiments.v4_gazebo_pilot.timing import seconds_to_ns

        s, schedule = self.scenario, self.schedule
        capture = schedule.at(self.index)
        require(
            capture < schedule.end_ns and self.battery() > 0,
            "Decision after depleted battery or mission window",
        )
        require(
            self.time <= s.contract.deadline_s
            and (capture - schedule.epoch_ns) / 1e9 <= s.contract.deadline_s,
            "Decision after deadline",
        )
        self.advance(capture)
        require(self.battery() > 0, "Decision after flight exhausted battery")
        self.advance(capture + seconds_to_ns(self.costs.encode_s, allow_zero=True))
        prep = self.costs.encode_s * self.costs.codec_power_w
        self.energy += prep
        self.preparation += prep
        return RuntimeState(
            current_time_s=self.time,
            frame_id=self.index,
            battery_frac=self.battery(),
            power_mode="v4_configured",
            network=s.network.observation_at(self.time),
            current_config_id=self.current,
            remaining_deadline_s=s.contract.deadline_s - self.time,
            cumulative_energy_j=self.energy,
            cumulative_communication_mb=self.communication,
            path_progress=min(1.0, self.time / s.duration_s),
            evidence_summary=EvidenceSummary(0, self.processed, 0.0),
        )

    def pay(self, action, prepared, event):
        from experiments.mission_network_policies import AttemptOutcome
        from experiments.v4_gazebo_pilot.timing import seconds_to_ns

        s, cost, schedule = self.scenario, self.costs, self.schedule
        capture = schedule.at(self.index)
        capture_s = (capture - schedule.epoch_ns) / 1e9
        decision = capture + seconds_to_ns(cost.encode_s, allow_zero=True)
        resource = s.resources.for_config(action.action_id)
        memory = resource.memory_mb + self.tracking.memory_mb
        require(
            s.resources.memory_capacity_mb is None or memory <= s.resources.memory_capacity_mb,
            "Auditor requires the predeclared nonbinding memory pool",
        )
        loading = action.action_id != self.current
        load_s = resource.load_latency_s if loading else 0.0
        load_j = resource.load_energy_j if loading else 0.0
        dispatch = decision + seconds_to_ns(s.router_latency_s + load_s, allow_zero=True)
        delivered, status, transport, radio = True, "completed", None, 0.0
        decode_s = decode_j = communication = 0.0
        latency, compute = action.latency_s, action.energy_j
        if action.placement == "remote":
            request = dict(
                start_ns=dispatch - schedule.epoch_ns,
                upload_bits=len(prepared.payload) * 8,
                download_bits=prepared.response_bytes * 8,
                service_ns=seconds_to_ns(action.latency_s, allow_zero=True),
                timeout_ns=seconds_to_ns(cost.timeout_s),
                deadline_ns=max(dispatch - schedule.epoch_ns, seconds_to_ns(s.contract.deadline_s)),
            )
            job = check_transport(self.trace, request, event["transport"])
            transport, status, delivered = job.to_dict(), job.status, job.prediction_eligible
            require(
                status in ("completed", "timeout", "deadline"),
                "Trace exhausted or unexpected transport failure",
            )
            radio = radio_joules(job, cost.tx_power_w, cost.rx_power_w, cost.wait_power_w)
            communication = float(job.uploaded_bits + job.downloaded_bits) / 8e6
            decode_s = cost.decode_s if delivered else 0.0
            decode_j = decode_s * cost.codec_power_w
            latency, compute = job.elapsed_ns / 1e9 + decode_s, decode_j
        else:
            require(event["transport"] is None, "Local action fabricated transport")
        track_s = self.tracking.latency_s if delivered else 0.0
        track_j = self.tracking.energy_j if delivered else 0.0
        completion, next_index, skipped = schedule.complete(
            self.index,
            seconds_to_ns(
                cost.encode_s + s.router_latency_s + load_s + latency + track_s, allow_zero=True
            ),
        )
        for missed in skipped:
            self.advance(schedule.at(missed))
        self.advance(completion)
        self.energy += compute + radio + s.router_energy_j
        self.energy += load_j + track_j
        self.compute += compute
        self.radio += radio
        self.routing += s.router_energy_j
        self.loading += load_j
        self.load_time += load_s
        self.tracking_j += track_j
        self.memory = max(self.memory, memory)
        self.communication += communication
        self.failures += int(not delivered)
        self.timeouts += int(not delivered and status in ("timeout", "deadline"))
        expected = dict(
            observation_id=self.index,
            capture_time_s=capture_s,
            capture_timestamp_ns=capture,
            decision_time_s=(decision - schedule.epoch_ns) / 1e9,
            decision_timestamp_ns=decision,
            dispatch_timestamp_ns=dispatch,
            dispatch_network_ns=dispatch - schedule.epoch_ns,
            completion_time_s=self.time,
            configured_router_latency_s=s.router_latency_s,
            configured_tracking_latency_s=track_s,
            tracking_energy_j=track_j,
            preparation_latency_s=cost.encode_s,
            preparation_energy_j=cost.encode_s * cost.codec_power_w,
            model_load_latency_s=load_s,
            model_load_energy_j=load_j,
            resident_memory_mb=memory,
            effective_latency_s=(completion - capture) / 1e9,
            attempt_status=status,
            delivered_prediction=delivered,
            source_config_id=action.source_config_id,
            placement=action.placement,
            battery_after=self.battery(),
            cumulative_energy_j=self.energy,
            cumulative_communication_mb=self.communication,
            skipped_observation_ids=list(skipped),
            prepared_payload=prepared.summary(),
            router_error=None,
        )
        same_fields(event, expected, "paid event")
        same_fields(
            event["execution"],
            dict(
                config_id=action.action_id,
                model_strategy_id=action.source_config_id,
                success=delivered,
                mission_latency_s=latency,
                energy_j=compute,
                communication_energy_j=radio,
                communication_mb=communication,
            ),
            "execution",
        )
        diagnostics = event["execution"]["diagnostics"]
        same_fields(
            diagnostics,
            dict(
                source_config_id=action.source_config_id,
                source_signature=action.source_signature,
                placement=action.placement,
                attempt_status=status,
                delivered_prediction=delivered,
                transport=transport,
                decode_latency_s=decode_s,
                decode_energy_j=decode_j,
                no_prediction_loaded_on_failure=not delivered,
            ),
            "execution diagnostics",
        )
        if action.placement == "remote":
            same_fields(diagnostics["radio"], dict(configured_radio_j=radio), "radio accounting")
        else:
            require(diagnostics["radio"] is None, "Local action charged radio")
        self.last_attempt = AttemptOutcome(
            action.action_id, capture_s, self.time, delivered, status
        )
        self.skipped.extend(skipped)
        self.selected.append(action.action_id)
        self.index, self.current = next_index, action.action_id
        self.processed += 1
        return expected, delivered, skipped

    def finish(self):
        """Derive stopping reason including an unpaid next capture and mission tail."""
        from experiments.v4_gazebo_pilot.timing import seconds_to_ns

        s, schedule = self.scenario, self.schedule
        reason = "mission_window_complete"
        if schedule.at(self.index) < schedule.end_ns:
            capture = schedule.at(self.index)
            capture_s = (capture - schedule.epoch_ns) / 1e9
            if self.battery() <= 0:
                reason = "battery_depleted"
            elif self.time > s.contract.deadline_s or capture_s > s.contract.deadline_s:
                if self.time < s.contract.deadline_s:
                    deadline = schedule.epoch_ns + seconds_to_ns(s.contract.deadline_s)
                    self.advance(
                        ((deadline + schedule.step_ns - 1) // schedule.step_ns) * schedule.step_ns
                    )
                reason = "deadline_exceeded"
            else:
                self.advance(capture)
                require(self.battery() <= 0, "Execution silently omitted an eligible decision")
                reason = "battery_depleted"
        elif self.battery() <= 0:
            reason = "battery_depleted"
        elif self.time < s.duration_s:
            self.advance(schedule.end_ns)
            if self.battery() <= 0:
                reason = "battery_depleted"
        return dict(
            status="completed",
            error=None,
            termination_reason=reason,
            mission_completed=reason == "mission_window_complete",
            final_time_s=self.time,
            final_battery_frac=self.battery(),
            cumulative_energy_j=self.energy,
            cumulative_communication_mb=self.communication,
            communication_energy_j=self.radio,
            preparation_energy_j=self.preparation,
            inference_energy_j=self.compute,
            router_energy_j=self.routing,
            model_load_energy_j=self.loading,
            model_load_latency_s=self.load_time,
            tracking_energy_j=self.tracking_j,
            flight_energy_j=self.time * s.flight_power_w,
            peak_configured_memory_mb=self.memory,
            processed_observations=self.processed,
            decision_stride=1,
            decision_period_s=s.observation_interval_s,
            cadence_skipped_observation_ids=[],
            busy_skipped_observation_ids=self.skipped,
            skipped_observation_ids=self.skipped,
            invalid_actions=0,
            privacy_violations=0,
            router_errors=0,
            inference_failures=self.failures,
            expected_transport_timeouts=self.timeouts,
            config_selection_history=self.selected,
            config_switches=sum(
                a != b for a, b in zip(self.selected, self.selected[1:], strict=False)
            ),
        )


def canonical_endpoint(contract, events, reconstructed_result):
    """Apply the original phase/report scorer; independently replace execution admissibility."""
    from experiments.lake_intent_contract import evaluate

    strict = evaluate(contract, events, reconstructed_result)
    value = dict(strict)
    value["constraints"] = dict(strict["constraints"])
    failed = [e for e in events if not e["delivered_prediction"]]
    recoverable = all(
        e["placement"] == "remote" and e["attempt_status"] in ("timeout", "deadline")
        for e in failed
    )
    r = reconstructed_result
    value["constraints"]["execution"] = bool(
        recoverable
        and r["status"] == "completed"
        and r["error"] is None
        and r["mission_completed"]
        and r["termination_reason"] == "mission_window_complete"
        and r["final_time_s"] >= contract.document()["duration_s"]
        and r["processed_observations"] == len(events)
        and r["router_errors"] == r["invalid_actions"] == 0
        and r["inference_failures"] == r["expected_transport_timeouts"] == len(failed)
    )
    value.update(
        schema_version="network_recoverable_intent_result_v1",
        mission_success=all(value["constraints"].values()),
        failed_constraints=[k for k, v in value["constraints"].items() if not v],
        expected_transport_timeouts=len(failed),
        strict_original_endpoint_success=strict["mission_success"],
        strict_original_failed_constraints=strict["failed_constraints"],
        endpoint_deviation="Paid expected transport cancellation permits later recovery; "
        "no prediction, tracker update, report or fallback on cancellation.",
    )
    return value


def reconstruct_summary(account, scenario, events, quality, report_quality, terminal=None):
    """Rebuild terminal resources and all resource flags without trusting the runner."""
    terminal = account.finish() if terminal is None else terminal
    extra, first = scenario.requirements.evaluate(quality, events, report_quality)
    flags = dict(extra)
    flags.update(
        memory_constraint_success=scenario.resources.memory_capacity_mb is None
        or account.memory <= scenario.resources.memory_capacity_mb,
        quality_success=scenario.contract.quality_satisfied(
            quality[scenario.contract.quality_metric]
        ),
        deadline_success=account.time <= scenario.contract.deadline_s,
        battery_constraint_success=account.battery() >= scenario.contract.min_final_battery_frac,
        communication_constraint_success=account.communication
        <= scenario.contract.communication_budget_mb,
        privacy_constraint_success=True,
    )
    terminal.update(
        contract=asdict(scenario.contract),
        quality=quality,
        report_quality=report_quality,
        constraints=flags,
        first_correct_report_s=first,
        mission_success=terminal["mission_completed"] and all(flags.values()),
    )
    return terminal


def run(root):
    """Read every raw recording once, reconstruct every policy execution independently."""
    import copy
    import math

    import numpy as np

    from missionbench.simulator.action_validator import ActionValidator
    from missionbench.v2.evaluation import MissionEvaluator
    from experiments.lake_hindsight_witness import VerifiedMasks
    from experiments.lake_intent_contract import IntentContract, immutable
    from experiments.lake_intent_runner import bind_scenario
    from experiments.lake_tracking import SurfaceProjector, TrackingConfig, TrackingPipeline
    from experiments.lake_tracking_evaluation import RunnerReportEvaluator
    from experiments.lake_tracking_runtime import load_public_terrain
    from experiments.mission_network_policies import FIXED_LOCAL_POLICIES, NetworkFeedbackInput
    from experiments.mission_network_study_v1.executors import prepare_payload
    from experiments.mission_network_study_v1.study import (
        actions_for,
        policy_costs,
        policy_for,
        scenario_for,
        trace_for,
        tracking_cost,
        verify,
    )
    from experiments.mission_routing_v2.feedback import snapshot
    from experiments.mission_routing_v2.train import checked_manifest
    from experiments.mission_runtime_profile import read, save, sha
    from experiments.v4_benchmark.backends import ReplayBackend
    from experiments.v4_benchmark.runner import evaluation_frame, immutable_rgb
    from experiments.v4_benchmark.scenario import digest

    root = Path(root)
    require(not (root / "validation.json").exists(), "Preserve an existing audit")
    plan = verify(root)
    initial_plan_sha = sha(root / "plan.json")
    initial_index_sha = sha(root / "run_index.json")
    index = read(root / "run_index.json")
    require(
        index["complete_execution_grid"] and index["plan_sha256"] == sha(root / "plan.json"),
        "Incomplete or unbound execution index",
    )
    rows = index["cases"]
    executions = complete_grid(plan, rows)
    require(executions == index["expected_replays"], "Index count differs")
    source = Path(plan["cohort"])
    terrain, map_metadata = load_public_terrain(Path(plan["assets"]))
    counts = dict(
        executions=0, events=0, public_decisions_reproduced=0, reports=0, canceled_predictions=0
    )
    hashes, terminals, pose_hashes, local_controls = {}, {}, {}, []
    for flight in sorted({v["source_flight"] for v in plan["cases"].values()}):
        folder = source / flight / "predictions"
        manifest = checked_manifest(folder, source / "protocol.json")
        manifest_sha = sha(folder / "manifest.json")
        require(
            {c: m["signature"] for c, m in manifest["models"].items()} == plan["model_signatures"],
            "Raw prediction-bank model identity differs",
        )
        masks = {c: VerifiedMasks(folder, m) for c, m in manifest["models"].items()}
        contexts, by_capture = [], defaultdict(list)
        for row in rows:
            if row["flight"] != flight:
                continue
            case = plan["cases"][row["cell"]]
            require(
                case["source_flight"] == flight and case["link"] == row["link"],
                "Flight/link label substituted",
            )
            destination = root / row["path"]
            require(destination.resolve().is_relative_to(root.resolve()), "Escaped output path")
            for name, key in (
                ("outcome.json", "outcome_sha256"),
                ("execution/events.json", "events_sha256"),
                ("execution/results.json", "results_sha256"),
                ("protocol.json", "protocol_sha256"),
            ):
                require(sha(destination / name) == row[key], "Changed indexed artifact")
                hashes[str((destination / name).relative_to(root))] = row[key]
            contract = IntentContract(
                read(root / "missions" / row["cell"] / row["mission"] / "intent.json")
            )
            scenario = bind_scenario(scenario_for(root, plan, row["cell"]), contract)
            require(
                manifest["recording_identity"] == scenario.recording_identity,
                "Raw bank belongs to another flight",
            )
            events = read(destination / "execution/events.json")
            result = read(destination / "execution/results.json")
            outcome = read(destination / "outcome.json")
            protocol = read(destination / "protocol.json")
            require(
                events and len({e["observation_id"] for e in events}) == len(events),
                "Empty or duplicate decision sequence",
            )
            require(
                [e["observation_id"] for e in events]
                == sorted(e["observation_id"] for e in events),
                "Decision sequence reordered",
            )
            costs, cost = policy_costs(plan, scenario), tracking_cost(plan)
            trace = trace_for(plan, row["cell"], scenario)
            projector = SurfaceProjector(terrain)
            tracker = TrackingPipeline(TrackingConfig(remember_reports=True), projector)
            tracker.provenance.update(
                map=map_metadata,
                anchor_height_m=projector.anchor_height_m,
                hfov_rad=projector.hfov,
                static_target_assumption=True,
            )
            tracking = dict(configuration=tracker.provenance, cost=asdict(cost))
            same_fields(
                protocol,
                dict(
                    router=row["policy"],
                    gt_aware=False,
                    deployable_policy=True,
                    schema_version="network_tracking_execution_v1",
                    study_plan_sha256=sha(root / "plan.json"),
                    contract_sha256=contract.sha256,
                    public_contract_sha256=contract.public_sha256,
                    scenario_sha256=digest(scenario.source),
                    recording_identity=scenario.recording_identity,
                    public_tracking=tracking,
                    tracking=tracker.provenance,
                    tracking_cost=asdict(cost),
                    network_costs=asdict(costs),
                    network_trace=[asdict(s) for s in trace.segments],
                ),
                "protocol",
            )
            for path, expected in protocol["implementation_sha256"].items():
                require(
                    plan["implementation_sha256"].get(path) == expected and sha(path) == expected,
                    "Execution implementation differs from pre-output source lock",
                )
            same_fields(
                outcome,
                dict(
                    gt_aware=False,
                    deployable_policy=True,
                    public_contract_deliveries=len(events),
                    protocol_sha256=row["protocol_sha256"],
                    events_sha256=row["events_sha256"],
                    execution_sha256=row["results_sha256"],
                ),
                "outcome binding",
            )
            same_fields(
                result,
                dict(
                    scenario_sha256=digest(scenario.source),
                    tracking=tracker.provenance,
                    tracking_cost=asdict(cost),
                    resources=scenario.resources.to_dict(),
                    router=row["policy"],
                    scenario_id=scenario.scenario_id,
                    contract=asdict(scenario.contract),
                    mission_requirements=scenario.requirements.to_dict(),
                    reporting=scenario.reporting.to_dict(),
                ),
                "result binding",
            )
            require(
                result["backend"]["recording_identity"] == scenario.recording_identity,
                "Execution recording identity differs",
            )
            validator = ActionValidator(
                catalog=scenario.catalog,
                allowed_config_ids=scenario.catalog.ids(),
                privacy_level=scenario.contract.privacy_level,
                fallback_config_id=scenario.fallback_config_id,
            )
            context = dict(
                row=row,
                destination=destination,
                scenario=scenario,
                contract=contract,
                events=events,
                result=result,
                outcome=outcome,
                tracker=tracker,
                tracking=tracking,
                policy=policy_for(root, plan, row["policy"], scenario),
                validator=validator,
                account=Accounting(scenario, cost, costs, trace),
                actions={a.action_id: a for a in actions_for(plan, scenario)},
                scorer=MissionEvaluator(
                    scenario.matching_iou_threshold,
                    len(scenario.targets),
                    scenario.observation_interval_s,
                ),
                report_scorer=RunnerReportEvaluator(
                    scenario.matching_iou_threshold, scenario.targets.values(), 2
                ),
                expected_events=[],
                predicted_detections=0,
                skipped=set(),
            )
            contexts.append(context)
            for event in events:
                by_capture[event["observation_id"]].append((context, event))
            counts["executions"] += 1
            counts["events"] += len(events)
        backend = ReplayBackend(scenario, source / flight / "recording")
        poses = []
        try:
            for capture, timestamp in enumerate(backend.frames):
                backend.advance_to(timestamp)
                frame = backend.capture()
                poses.append((timestamp, frame.body_pose))
                # Visibility is evaluator-private and never enters policy reconstruction.
                for c in contexts:
                    if capture in c["skipped"]:
                        truth = evaluation_frame(
                            frame,
                            capture,
                            (timestamp - c["scenario"].schedule.epoch_ns) / 1e9,
                            c["scenario"],
                        )
                        c["scorer"].note_skipped_visibility(truth.visible_target_ids)
                        c["report_scorer"].note_visibility(truth)
                if not by_capture[capture]:
                    continue
                prepared = prepare_payload(frame.rgb)
                predictions = {}
                for c, event in by_capture[capture]:
                    s, account, tracker = c["scenario"], c["account"], c["tracker"]
                    require(account.index == capture, "Busy/cadence capture schedule changed")
                    require(frame.body_pose == event["capture_body_pose"], "Physical pose changed")
                    state = account.state()
                    check_public_state(event["runtime_at_decision"], state.to_dict())
                    link = account.trace.snapshot(
                        s.schedule.at(capture)
                        - s.schedule.epoch_ns
                        + round(account.costs.encode_s * 1e9)
                    )
                    # The timestamp itself is reconstructed before using it below.
                    require(
                        link.timestamp_ns == round(state.current_time_s * 1e9),
                        "Snapshot clock changed",
                    )
                    require(
                        event["public_link"] == link.to_dict(), "Current-only link snapshot differs"
                    )
                    observation = NetworkFeedbackInput(
                        contract=s.contract,
                        intent_contract=immutable(c["contract"].public_document()),
                        public_contract_sha256=c["contract"].public_sha256,
                        state=state,
                        configs=s.catalog,
                        resources=s.resources,
                        tracking=immutable(c["tracking"]),
                        evidence_history=immutable(snapshot(tracker, state.current_time_s)),
                        confirmed_reports_so_far=tracker.report_count,
                        rgb=immutable_rgb(frame.rgb),
                        position_m=tuple(frame.body_pose["position_m"]),
                        quaternion_xyzw=tuple(frame.body_pose["quaternion_xyzw"]),
                        detections_so_far=c["predicted_detections"],
                        requirements=s.requirements,
                        public_link=link,
                        current_payload_bytes=len(prepared.payload),
                        response_bytes=prepared.response_bytes,
                        last_attempt=account.last_attempt,
                    )
                    choice = c["policy"].select(observation)
                    action = c["validator"].validate(choice, current_config_id=account.current)
                    require(
                        action.is_valid and action.to_dict() == event["action"],
                        "Public-only selected action differs",
                    )
                    require(
                        c["policy"].last_decision == event["router_decision"],
                        "Policy decision trace differs",
                    )
                    expected, delivered, skipped = account.pay(
                        c["actions"][choice], prepared, event
                    )
                    c["skipped"].update(skipped)
                    base = c["actions"][choice].source_config_id
                    prediction = np.zeros(frame.rgb.shape[:2], dtype=np.bool_)
                    if delivered:
                        if base not in predictions:
                            predictions[base] = masks[base][prepared.rgb_sha256]
                        prediction = predictions[base]
                    require(
                        prediction.dtype == np.bool_ and prediction.shape == frame.rgb.shape[:2],
                        "Selected raw mask format differs",
                    )
                    same_fields(
                        event["execution"],
                        dict(predicted_pixels=int(prediction.sum())),
                        "mask summary",
                    )
                    require(
                        event["execution"]["diagnostics"]["cache_sha256"] == manifest_sha,
                        "Selected mask cache binding differs",
                    )
                    saved = c["destination"] / "execution" / f"observation_{capture:05d}.npz"
                    with np.load(saved, allow_pickle=False) as archive:
                        require(
                            set(archive.files) == {"rgb", "object_ids", "prediction_mask"}
                            and np.array_equal(archive["rgb"], frame.rgb)
                            and np.array_equal(archive["object_ids"], frame.object_ids)
                            and np.array_equal(archive["prediction_mask"], prediction),
                            "Saved execution frame differs from raw bank or canceled zero mask",
                        )
                    hashes[str(saved.relative_to(root))] = sha(saved)
                    before = copy.deepcopy(vars(tracker.tracker))
                    reports = []
                    if delivered:
                        reports = tracker.update(
                            prediction,
                            expected["capture_time_s"],
                            expected["completion_time_s"],
                            pose=frame.body_pose,
                            camera_pitch_rad=s.world.get("camera_pitch_rad", math.pi / 2),
                        )
                    check_cancelled_prediction(
                        delivered, event["emitted_reports"], before, vars(tracker.tracker)
                    )
                    require(
                        event["emitted_reports"] == reports,
                        "Completed prediction report history differs",
                    )
                    truth = evaluation_frame(frame, capture, expected["capture_time_s"], s)
                    c["report_scorer"].note_visibility(truth)
                    score = c["scorer"].update(truth, prediction)
                    require(score.to_dict() == event["score"], "Raw mask score differs")
                    if delivered:
                        c["report_scorer"].update(truth, prediction, reports)
                    c["predicted_detections"] += score.predicted_components
                    reconstructed = dict(event, **expected)
                    reconstructed.update(
                        score=score.to_dict(),
                        emitted_reports=reports,
                        delivered_prediction=delivered,
                    )
                    c["expected_events"].append(reconstructed)
                    counts["public_decisions_reproduced"] += 1
                    counts["reports"] += len(reports)
                    counts["canceled_predictions"] += int(not delivered)
        finally:
            backend.close()
        pose_hashes[flight] = digest(poses)
        for c in contexts:
            s, account = c["scenario"], c["account"]
            # finish is performed once; summary metrics use the reconstructed final time.
            terminal = account.finish()
            quality = c["scorer"].quality_details(account.time)
            report_quality = c["report_scorer"].summary(account.time)
            terminal = reconstruct_summary(
                account, s, c["expected_events"], quality, report_quality, terminal
            )
            same_fields(c["result"], terminal, "reconstructed result")
            canonical = canonical_endpoint(c["contract"], c["expected_events"], terminal)
            same_fields(c["outcome"], canonical, "canonical endpoint")
            row = c["row"]
            same_fields(
                row,
                dict(
                    full_success=canonical["mission_success"],
                    calls=len(c["events"]),
                    phases_succeeded=sum(p["success"] for p in canonical["phases"]),
                    report_recall=canonical["reporting"]["recall_by_deadline"],
                    failed_constraints=canonical["failed_constraints"],
                    expected_transport_timeouts=account.timeouts,
                    communication_mb=account.communication,
                    final_battery=account.battery(),
                    selections=dict(Counter(account.selected)),
                ),
                "index summary",
            )
            terminals[row["path"]] = dict(terminal, canonical_outcome=canonical)
            if row["policy"] in FIXED_LOCAL_POLICIES:
                require(
                    all(e["placement"] == "local" for e in c["events"]),
                    "Fixed-local policy offloaded",
                )
                semantic_events = [
                    {
                        k: e[k]
                        for k in (
                            "observation_id",
                            "action",
                            "capture_timestamp_ns",
                            "completion_time_s",
                            "capture_body_pose",
                            "battery_after",
                            "cumulative_energy_j",
                            "score",
                            "emitted_reports",
                            "skipped_observation_ids",
                        )
                    }
                    for e in c["expected_events"]
                ]
                local_controls.append(
                    dict(
                        flight=flight,
                        policy=row["policy"],
                        link=row["link"],
                        semantic=dict(
                            events=semantic_events, terminal=terminal, endpoint=canonical
                        ),
                    )
                )
        print(
            f"Reconstructed {flight}: {counts['public_decisions_reproduced']} decisions",
            flush=True,
        )
    require(
        counts["executions"] == executions
        and counts["events"] == counts["public_decisions_reproduced"],
        "Incomplete independent reconstruction",
    )
    groups = local_invariance(local_controls)
    require(
        groups == len(plan["flights"]) * len(FIXED_LOCAL_POLICIES), "Missing fixed-local controls"
    )
    verify(root)
    require(
        sha(root / "plan.json") == initial_plan_sha
        and sha(root / "run_index.json") == initial_index_sha,
        "Plan/index changed during audit",
    )
    # Reject writes racing the audit, including any saved mask change.
    for relative, expected in hashes.items():
        require(sha(root / relative) == expected, "Artifact changed during audit")
    validation = dict(
        complete=True,
        schema_version="mission_network_audit_v1",
        plan_sha256=sha(root / "plan.json"),
        run_index_sha256=sha(root / "run_index.json"),
        validator_sha256=sha(Path(__file__)),
        source_output_sha256=hashes,
        terminal_reconstructions=terminals,
        recording_pose_sha256=pose_hashes,
        fixed_local_invariance_groups=groups,
        canonical_outcomes_verified=True,
        public_states_and_resource_flags_reconstructed=True,
        raw_selected_mask_scores_reconstructed=True,
        report_identity_support_reconstructed=True,
        per_event_cost_clock_partial_bytes_verified=True,
        transport_algorithm_shared_with_simulator=True,
        transport_analytical_tests="tests/test_mission_network_emulation.py",
        no_mission_runner_called=True,
        fresh_evaluation=False,
        **counts,
    )
    save(root / "validation.json", validation)
    return validation


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checked = run(args.output)
    print(
        {k: checked[k] for k in ("complete", "executions", "events", "public_decisions_reproduced")}
    )
