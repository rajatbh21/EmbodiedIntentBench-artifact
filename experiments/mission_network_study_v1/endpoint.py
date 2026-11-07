"""Network endpoint: paid expected transport cancellations are recoverable."""

import copy

from experiments.lake_intent_contract import evaluate


def valid_expected_timeout(event):
    execution = event["execution"]
    diagnostic = execution.get("diagnostics") or {}
    transport = diagnostic.get("transport") or {}
    return (
        event.get("attempt_status") in ("timeout", "deadline")
        and event.get("placement") == "remote"
        and diagnostic.get("attempt_status") == event["attempt_status"]
        and transport.get("status") == event["attempt_status"]
        and transport.get("prediction_eligible") is False
        and event.get("delivered_prediction") is False
        and diagnostic.get("delivered_prediction") is False
        and diagnostic.get("no_prediction_loaded_on_failure") is True
        and execution["success"] is False
        and execution["predicted_pixels"] == 0
        and event["emitted_reports"] == []
        and event["configured_tracking_latency_s"] == 0
        and event["tracking_energy_j"] == 0
        and diagnostic.get("decode_latency_s") == 0
        and diagnostic.get("decode_energy_j") == 0
    )


def evaluate_network(contract, events, result):
    strict = evaluate(contract, events, result)
    outcome = copy.deepcopy(strict)
    recoverable = sum(valid_expected_timeout(e) for e in events)
    accepted = all(e["execution"]["success"] is True or valid_expected_timeout(e) for e in events)
    execution_valid = (
        result["status"] == "completed"
        and result["error"] is None
        and result["mission_completed"] is True
        and result["termination_reason"] == "mission_window_complete"
        and result["final_time_s"] >= contract.document()["duration_s"]
        and result["processed_observations"] == len(events)
        and result["router_errors"] == result["invalid_actions"] == 0
        and result["inference_failures"] == recoverable
        and result["expected_transport_timeouts"] == recoverable
        and accepted
    )
    outcome["constraints"]["execution"] = execution_valid
    outcome.update(
        schema_version="network_recoverable_intent_result_v1",
        mission_success=all(outcome["constraints"].values()),
        failed_constraints=[k for k, v in outcome["constraints"].items() if not v],
        expected_transport_timeouts=recoverable,
        strict_original_endpoint_success=strict["mission_success"],
        strict_original_failed_constraints=strict["failed_constraints"],
        endpoint_deviation="Paid expected transport cancellation permits later recovery; "
        "no prediction, tracker update, report or fallback on cancellation.",
    )
    return outcome
