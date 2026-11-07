"""Bind the rejected sensor-only acquisition and unchanged method implementation."""

import ast
from pathlib import Path

from experiments.lake_feasibility_inference import read
from experiments.lake_intent_contract import IntentContract
from experiments.lake_missions.assets import sha
from experiments.v4_benchmark.scenario import digest

REJECTION_SHA256 = "8b0f33ce1c6e7942891c60d4bab704914021c39cbcf07c9e8d745b57913708df"
EVIDENCE = (
    "protocol.json",
    "study/protocol.json",
    "sensor_gate.json",
    "physical_route_validation.json",
    "acquisition_completion.json",
)
METHOD_FIELDS = (
    "policies",
    "batteries",
    "missions",
    "active_cpu_power_sensitivity_w",
    "primary_policy",
    "primary_comparisons",
    "primary_endpoint",
    "training",
    "risk_training",
    "temporal_training",
    "coverage_training",
    "profile_sha256",
)
INTERVENTION_FIELDS = (
    "mission",
    "executor_configs",
    "public_profiles",
    "network_trace",
    "resources",
    "reporting",
    "mission_requirements",
)


def rejected_acquisition(root, *, expected_sha256=REJECTION_SHA256):
    """Validate the specific saved rejection and return an exact content binding."""
    root = Path(root)
    rejection = root / "ADMISSION_REJECTED.json"
    if sha(rejection) != expected_sha256:
        raise ValueError("Changed sensor-only acquisition rejection")
    note = read(rejection)
    if (
        note["schema_version"] != "pre_inference_acquisition_rejection_v1"
        or note["acquisition_complete"] is not True
        or any(
            note[k] is not False
            for k in (
                "admitted_to_model_evaluation",
                "neural_predictions_started",
                "policy_replays_started",
            )
        )
        or set(note["evidence_sha256"]) != set(EVIDENCE)
    ):
        raise ValueError("Rejection must precede all model inference and policy evaluation")
    for name, expected in note["evidence_sha256"].items():
        if sha(root / name) != expected:
            raise ValueError("Changed rejected acquisition evidence: " + name)
    cohort = read(root / "protocol.json")
    study = read(root / "study/protocol.json")
    sensor = read(root / "sensor_gate.json")
    route = read(root / "physical_route_validation.json")
    acquisition = read(root / "acquisition_completion.json")
    if (
        sensor["passed"] is not False
        or sensor["model_predictions_used"] is not False
        or sensor["rule_defined_before_capture"] is not True
        or route["all_passed"] is not False
        or route["model_predictions_used"] is not False
        or acquisition["complete"] is not True
        or sensor["protocol_sha256"] != sha(root / "protocol.json")
        or sensor["physical_route_validation_sha256"]
        != sha(root / "physical_route_validation.json")
        or acquisition["sensor_gate_sha256"] != sha(root / "sensor_gate.json")
        or study["cohort_protocol_sha256"] != sha(root / "protocol.json")
        or Path(study["cohort"]).resolve() != root.resolve()
        or set(note["recording_sha256"]) != set(cohort["cases"])
    ):
        raise ValueError("Rejected sensor/route/recording provenance differs")
    for flight, expected in note["recording_sha256"].items():
        if sha(root / flight / "recording/recording.json") != expected:
            raise ValueError("Changed rejected physical recording")
        if (root / flight / "predictions").exists():
            raise ValueError("Rejected acquisition must have no neural prediction attempt")
    if (root / "study/runs").exists():
        raise ValueError("Rejected acquisition must have no policy replay attempt")
    return dict(
        cohort=str(root.resolve()),
        rejection_sha256=expected_sha256,
        evidence_sha256=note["evidence_sha256"],
        recording_sha256=note["recording_sha256"],
        raw_rgb_and_geometry_gt_inspected=True,
        model_outputs_inspected=False,
        policy_outcomes_inspected=False,
        sensor_admission_passed=False,
        excluded_as_policy_trial=True,
    )


def method_parity():
    """Check factory/cost-binding AST parity and byte-identical integrity checks."""
    previous = Path(__file__).parents[1] / "mission_transfer_comparison"
    current = Path(__file__).parent

    def tree(path, name):
        module = ast.parse(path.read_text())
        matches = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == name]
        if len(matches) != 1:
            raise ValueError("Missing unique method factory")
        return ast.dump(matches[0], include_attributes=False)

    identities = {}
    for name in ("make_policy", "scenario_for"):
        a, b = tree(previous / "study.py", name), tree(current / "study.py", name)
        if a != b:
            raise ValueError("Changed policy factory or configured-cost binding: " + name)
        identities[name + "_ast_sha256"] = digest(a)
    if sha(previous / "integrity.py") != sha(current / "integrity.py"):
        raise ValueError("Changed inherited execution integrity checker")
    return dict(
        inherited_comparison_package=str(previous.resolve()),
        inherited_study_sha256=sha(previous / "study.py"),
        inherited_integrity_sha256=sha(previous / "integrity.py"),
        factory_and_cost_binding_identical=True,
        **identities,
    )


def unchanged_design(protocol, predecessor):
    for key in METHOD_FIELDS:
        if protocol[key] != predecessor[key]:
            raise ValueError("Changed inherited comparison method/design: " + key)


def unchanged_interventions(cohort, rejected):
    """Geometry engineering cannot silently change difficulty thresholds or costs."""
    current, previous = read(cohort / "protocol.json"), read(rejected / "protocol.json")
    if (
        set(current["cases"]) != set(previous["cases"])
        or current["missions"] != previous["missions"]
    ):
        raise ValueError("Geometry revision must retain all acquisition conditions and intents")
    fingerprints = {}
    for flight, case in current["cases"].items():
        if any(case[k] != previous["cases"][flight][k] for k in ("layout_group", "lighting")):
            raise ValueError("Changed layout/lighting condition assignment")
        for mission in current["missions"]:
            a, b = cohort / flight / mission, rejected / flight / mission
            new_scenario, old_scenario = read(a / "scenario.json"), read(b / "scenario.json")
            values = {}
            for key in INTERVENTION_FIELDS:
                if new_scenario.get(key) != old_scenario.get(key):
                    raise ValueError("Changed inherited scenario cost/model/timing field: " + key)
                values[key] = new_scenario.get(key)
            if read(a / "contract.json") != read(b / "contract.json"):
                raise ValueError("Changed inherited contract limits")
            new_intent = IntentContract(read(a / "intent.json")).public_document()
            old_intent = IntentContract(read(b / "intent.json")).public_document()
            if new_intent != old_intent:
                raise ValueError("Changed inherited public mission intent/threshold")
            values.update(contract=read(a / "contract.json"), public_intent=new_intent)
            fingerprints[f"{flight}/{mission}"] = digest(values)
    return dict(
        costs_models_timing_contracts_unchanged=True,
        permitted_revision=(
            "Physical geometry/collision implementation, not costs or intent difficulty"
        ),
        intervention_sha256=fingerprints,
    )
