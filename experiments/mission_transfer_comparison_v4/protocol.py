"""Preserve pre-capture methods while amending admission scope before model outputs.

The original exact cross-lighting pose gate remains failed. This protocol reuses
all recorded banks for exact within-recording router, intent and battery contrasts;
lighting is an observational stratum, not an isolated causal intervention.
"""

import argparse
import ast
import copy
import shutil
from pathlib import Path

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.mission_coverage_routing.protocol import profile_paths
from experiments.mission_transfer_comparison_v3.protocol import verify as verify_previous
from experiments.mission_transfer_within_recording.admission import verify as verify_admission
from experiments.v4_benchmark.scenario import digest

PROVENANCE = dict(
    fresh_test=False,
    prospective_policy_evaluation=True,
    new_capture_run=False,
    recordings_reused_without_edit=True,
    strict_unmodified_prospective_test=False,
    fully_prospective_protocol=False,
    fully_sensor_unseen_environment=False,
    map_unseen_by_development_training=True,
    geometry_revised_after_rejected_acquisition=True,
    raw_rgb_and_geometry_gt_seen_before_revision=True,
    model_outputs_seen_before_freeze=False,
    policy_outcomes_seen_before_freeze=False,
    outcome_driven_method_change=False,
    methods_frozen_before_capture=True,
    evaluation_scope_frozen_before_capture=False,
    admission_amended_after_capture=True,
    admission_amended_before_model_outputs=True,
    original_strict_sensor_admission_passed=False,
    within_recording_policy_pairing_exact=True,
    intent_and_battery_pairing_exact=True,
    cross_lighting_pose_pairing_exact=False,
    lighting_causal_effect_identified=False,
)
CHANGED_FIELDS = {
    "schema_version",
    "implementation_sha256",
    "scope",
    "claims",
    "inherited_method_parity",
    "new_capture_run",
}


def method_parity():
    previous = Path(__file__).parents[1] / "mission_transfer_comparison_v3"
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


def original_method_freeze(cohort, previous):
    """Bind the saved pre-capture acquisition order, not filesystem timestamps."""
    order_path = cohort / "acquisition_order.json"
    order = read(order_path)
    if (
        order["study_protocol_sha256"] != sha(previous / "protocol.json")
        or order["cohort_protocol_sha256"] != sha(cohort / "protocol.json")
        or order["methods_frozen_before_this_capture_run"] is not True
        or order["neural_inference_started"] is not False
    ):
        raise ValueError("Missing original method freeze before physical capture")
    protocol = read(previous / "protocol.json")
    if order["flights"] != list(read(cohort / "protocol.json")["cases"]):
        raise ValueError("Changed original acquisition population")
    if protocol["methods_frozen_before_capture"] is not True:
        raise ValueError("Original methods were not frozen before capture")
    return dict(
        acquisition_order_sha256=sha(order_path),
        original_study_protocol_sha256=sha(previous / "protocol.json"),
        methods_frozen_before_capture=True,
        amendment_frozen_before_capture=False,
    )


def no_outputs(cohort, previous, output):
    cases = read(cohort / "protocol.json")["cases"]
    if any((cohort / f / "predictions").exists() for f in cases):
        raise ValueError("Scope amendment must precede every neural prediction attempt")
    if any(
        (root / "runs").exists() or (root / "witnesses").exists() for root in (previous, output)
    ):
        raise ValueError("Scope amendment must precede every policy or witness replay attempt")


def source_paths(previous, admission):
    repo = Path(__file__).resolve().parents[2]
    inherited = {**previous["implementation_sha256"], **admission["implementation_sha256"]}
    own = [
        p for p in Path(__file__).parent.glob("*.py") if p.name not in ("report.py", "pipeline.py")
    ]
    own += [
        repo / "tests/test_mission_transfer_comparison_v4.py",
        repo / "docs/mission_transfer_comparison_v4.md",
    ]
    if any(not p.is_file() for p in own):
        raise ValueError("Complete v4 source, tests and method documentation before freezing")
    return inherited, own


def inherited_design(protocol, previous):
    for key, value in previous.items():
        if key not in CHANGED_FIELDS and protocol.get(key) != value:
            raise ValueError("Changed inherited method, cost, intent or grid: " + key)


def check_prediction_admission(manifest, protocol):
    if manifest.get("routing_admission_sha256") != protocol[
        "routing_admission_sha256"
    ] or manifest.get("routing_admission_root") != str(Path(protocol["admission"]).resolve()):
        raise ValueError("Prediction bank belongs to a different routing admission")


def prepare(cohort, admission, output):
    cohort, admission, output = (p.resolve() for p in (cohort, admission, output))
    previous_root = cohort / "study"
    previous = verify_previous(previous_root)
    certificate = verify_admission(admission)
    if Path(certificate["cohort"]).resolve() != cohort:
        raise ValueError("Admission certificate belongs to a different cohort")
    original = original_method_freeze(cohort, previous_root)
    no_outputs(cohort, previous_root, output)
    output.mkdir(parents=True, exist_ok=False)
    for name, expected in previous["frozen_files"].items():
        source, target = previous_root / name, output / name
        if sha(source) != expected:
            raise ValueError("Changed original condition")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    sources, own = source_paths(previous, certificate)
    sources.update({str(path.resolve()): sha(path) for path in own})
    protocol = copy.deepcopy(previous)
    protocol.update(
        schema_version="mission_map_transfer_comparison_v4",
        pre_amendment_study=str(previous_root),
        pre_amendment_protocol_sha256=sha(previous_root / "protocol.json"),
        admission=str(admission),
        routing_admission_sha256=sha(admission / "routing_admission_v1.json"),
        original_method_freeze=original,
        inherited_method_parity=method_parity(),
        implementation_sha256=sources,
        scope="within_recording_policy_transfer_after_disclosed_postcapture_scope_amendment",
        claims=(
            "Descriptive comparison of unchanged routers on the same recorded inputs; "
            "intent and battery reuse exact RGB/poses; lighting is an observational stratum. "
            "Scope admission was amended after capture and before all model outputs."
        ),
        **PROVENANCE,
    )
    inherited_design(protocol, previous)
    save(output / "protocol.json", protocol)
    return verify(output)


def verify(root):
    from experiments.mission_transfer_comparison_v4.study import POLICIES

    protocol = read(root / "protocol.json")
    if protocol["schema_version"] != "mission_map_transfer_comparison_v4" or protocol[
        "policies"
    ] != list(POLICIES):
        raise ValueError("Changed comparison version or policy grid")
    cohort, previous_root, admission = (
        Path(protocol[k]) for k in ("cohort", "pre_amendment_study", "admission")
    )
    if previous_root.resolve() != (cohort / "study").resolve():
        raise ValueError("Wrong original pre-capture method freeze")
    previous = verify_previous(previous_root)
    certificate = verify_admission(admission)
    if (
        Path(certificate["cohort"]).resolve() != cohort.resolve()
        or sha(admission / "routing_admission_v1.json") != protocol["routing_admission_sha256"]
        or sha(previous_root / "protocol.json") != protocol["pre_amendment_protocol_sha256"]
        or original_method_freeze(cohort, previous_root) != protocol["original_method_freeze"]
        or method_parity() != protocol["inherited_method_parity"]
    ):
        raise ValueError("Changed admission, original method freeze or method parity")
    inherited_design(protocol, previous)
    if any(protocol.get(k) is not v for k, v in PROVENANCE.items()):
        raise ValueError("Changed capture/amendment provenance or paired-comparison scope")
    inherited, own = source_paths(previous, certificate)
    expected = {**inherited, **{str(path.resolve()): sha(path) for path in own}}
    if protocol["implementation_sha256"] != expected:
        raise ValueError("Changed or incomplete frozen source inventory")
    for name, expected_hash in protocol["implementation_sha256"].items():
        if sha(Path(name)) != expected_hash:
            raise ValueError("Changed frozen source: " + name)
    for name, expected_hash in protocol["frozen_files"].items():
        if sha(root / name) != expected_hash:
            raise ValueError("Changed exact intent/battery condition copy")
    for name, path in profile_paths(protocol).items():
        if sha(path) != protocol["profile_sha256"][name]:
            raise ValueError("Changed development profile")
    return protocol


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.cohort, args.admission, args.output)
