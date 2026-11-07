"""A public evaluation envelope; hidden evaluator identities never enter it.

The original policies already receive the intent and deadline profile separately.
This envelope documents and validates those inputs without claiming that old
executions used a new policy API or a new success definition.
"""

from experiments.lake_intent_contract import IntentContract
from experiments.mission_task_deadlines.contract import DeadlineProfile

SCHEMA = "perception_evaluation_v1"
FIXED_DOWNSTREAM = {
    "association_id": "ASSOC_TERRAIN_WORLD",
    "verification_id": "VERIFY_REPEAT2",
}


def public_specification(intent: dict, protocol: dict) -> dict:
    contract = IntentContract(intent)
    profile = DeadlineProfile.from_dict(protocol["deadline_profile"])
    if protocol["contract_sha256"] != contract.sha256:
        raise ValueError("Protocol does not bind this intent")
    if protocol["public_contract_sha256"] != contract.public_sha256:
        raise ValueError("Protocol does not bind this public intent")
    if protocol["downstream_selection"] != FIXED_DOWNSTREAM:
        raise ValueError("Perception-only comparison requires fixed downstream algorithms")
    if (profile.association_ns, profile.verification_ns) != (150_000_000, 30_000_000):
        raise ValueError("Perception-only comparison requires fixed downstream budgets")
    return {
        "schema_version": SCHEMA,
        "task": "perception",
        "decision": "perception configuration and local/remote placement, or paid abstention",
        "intent": contract.public_document(),
        "public_intent_sha256": contract.public_sha256,
        "task_deadlines": profile.to_dict(),
        "deadline_origin": "input capture availability, before preparation and routing",
        "deadline_rule": "inclusive usable finish; late outputs discarded and costs retained",
        "fixed_downstream": dict(FIXED_DOWNSTREAM),
        "recording_identity": protocol["recording"]["recording_identity"],
        "online_observations": [
            "current RGB",
            "current link snapshot",
            "current battery fraction",
            "cumulative communication",
            "resident action",
            "current capture time",
            "payload and response sizes",
        ],
        "private_to_evaluator": [
            "target identities and masks",
            "future link changes",
            "unselected model outputs",
            "future observations",
        ],
        "primary_endpoint": "unchanged task_deadline_recoverable_intent_result_v1 mission_success",
        "phase_availability": "finish of the fixed downstream pipeline",
        "role": "public documentation envelope for existing inputs; no policy API migration",
    }
