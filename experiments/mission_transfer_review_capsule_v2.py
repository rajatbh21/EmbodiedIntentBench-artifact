"""Export and verify a portable endpoint capsule, retaining failed acquisition history.

The isolated check uses only Python's standard library. It recomputes canonical
endpoints from audited events; it does not repeat inference, physics, routing,
raw-image matching or the evaluator-only witness search.
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True

SCHEMA = "mission_transfer_endpoint_capsule_v2"
SCORER_FILES = (
    "experiments/__init__.py",
    "experiments/lake_intent_contract.py",
    "experiments/lake_missions/__init__.py",
    "experiments/lake_missions/objectives.py",
)
PROVENANCE = {
    "fresh_test": False,
    "prospective_policy_evaluation": True,
    "fully_sensor_unseen_environment": False,
    "map_unseen_by_development_training": True,
    "geometry_revised_after_rejected_acquisition": True,
    "raw_rgb_and_geometry_gt_seen_before_revision": True,
    "model_outputs_seen_before_freeze": False,
    "policy_outcomes_seen_before_freeze": False,
    "fully_prospective_protocol": False,
    "methods_frozen_before_capture": True,
    "evaluation_scope_frozen_before_capture": False,
    "admission_amended_after_capture": True,
    "admission_amended_before_model_outputs": True,
    "original_strict_sensor_admission_passed": False,
    "within_recording_policy_pairing_exact": True,
    "intent_and_battery_pairing_exact": True,
    "cross_lighting_pose_pairing_exact": False,
    "lighting_causal_effect_identified": False,
}


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative_path(value):
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or path == Path("."):
        raise ValueError("Unsafe relative artifact path")
    return path


def exact_inventory(root, manifest):
    """Reject unlisted or non-source executable material before scorer imports."""
    files = manifest["files_sha256"]
    if "manifest.json" in files or "README.txt" not in files:
        raise ValueError("Only manifest.json may be outside the hash inventory")
    expected = set(files) | {"manifest.json"}
    for name in expected:
        path = relative_path(name)
        if path.as_posix() != name:
            raise ValueError("Noncanonical inventory path")
        if "__pycache__" in path.parts or path.suffix.lower() in (".pyc", ".pyo"):
            raise ValueError("Forbidden bytecode in capsule inventory")
    actual = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if path.is_symlink():
            raise ValueError("Symlink is forbidden in capsule")
        if "__pycache__" in relative.parts or path.suffix.lower() in (".pyc", ".pyo"):
            raise ValueError("Forbidden bytecode or cache directory in capsule")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("Nonregular file is forbidden in capsule")
        actual.add(relative.as_posix())
    if actual != expected:
        raise ValueError("Unlisted or missing capsule files: " + str(sorted(actual ^ expected)))
    for name, expected_hash in files.items():
        if sha(root / name) != expected_hash:
            raise ValueError("Changed capsule artifact: " + name)
    return actual


def grid(protocol, rows):
    expected = {
        (f, m, p)
        for f in protocol["cases"]
        for m in protocol["missions"]
        for p in protocol["policies"]
    }
    actual = [(r["flight"], r["mission"], r["policy"]) for r in rows]
    if not expected or len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Incomplete policy grid")
    if len(set(protocol["policies"])) != len(protocol["policies"]):
        raise ValueError("Repeated policy registry entry")
    return {(f, m) for f in protocol["cases"] for m in protocol["missions"]}


def witness_grid(cells, rows):
    actual = [(r["flight"], r["mission"]) for r in rows]
    if len(actual) != len(cells) or set(actual) != cells:
        raise ValueError("Incomplete witness-search grid")
    for row in rows:
        if type(row["full_contract_witness"]) is not bool:
            raise ValueError("Explicit witness status required")
        if "path" in row:
            if (
                type(row["full_success"]) is not bool
                or row["full_success"] != row["full_contract_witness"]
            ):
                raise ValueError("Paid replay and witness status differ")
        elif row["full_contract_witness"]:
            raise ValueError("A successful witness requires paid replay")


def preacquisition_history(root, copy):
    """Export a superseded design and its original source bytes, never a flight."""
    revision = read(root / "PRE_ACQUISITION_REVISION.json")
    inventory = read(root / "source_snapshot_manifest.json")
    protocol = read(root / "protocol.json")
    if (
        revision["status"] != "superseded_before_capture"
        or revision["physical_captures"] != 0
        or revision["model_predictions"] != 0
        or revision["source_snapshot_preserved"] is not True
        or inventory["verified"] is not True
        or inventory["protocol_sha256"] != sha(root / "protocol.json")
    ):
        raise ValueError("Changed superseded pre-acquisition history")
    sources = {r["original_path"]: r["sha256"] for r in inventory["files"]}
    if len(sources) != len(inventory["files"]) or sources != protocol["implementation_sha256"]:
        raise ValueError("Incomplete superseded source snapshot inventory")
    for name in ("PRE_ACQUISITION_REVISION.json", "source_snapshot_manifest.json", "protocol.json"):
        copy(root / name, "preacquisition_v1/" + name)
    for entry in inventory["files"]:
        relative = relative_path(entry["snapshot"])
        copy(root / relative, "preacquisition_v1/" + relative.as_posix(), entry["sha256"])
    return dict(
        root=str(root.resolve()),
        status="superseded_before_capture",
        physical_captures=0,
        model_predictions=0,
        source_files=len(sources),
        protocol_sha256=sha(root / "protocol.json"),
        snapshot_manifest_sha256=sha(root / "source_snapshot_manifest.json"),
        revision_note_sha256=sha(root / "PRE_ACQUISITION_REVISION.json"),
    )


def attribution_evidence(study, copy):
    """Preserve captured attribution documents without redistributing model weights."""
    inventory = study / "attribution_evidence.json"
    sidecar = study / "ATTRIBUTION.md"
    if not inventory.exists() and not sidecar.exists():
        return None
    if not inventory.is_file() or not sidecar.is_file():
        raise ValueError("Incomplete attribution sidecar")
    document = read(inventory)
    if (
        document["schema_version"] != "mission_transfer_attribution_v1"
        or document["posthoc"] is not True
        or document["sidecar"] != "ATTRIBUTION.md"
        or Path(document["study"]).resolve() != study.resolve()
    ):
        raise ValueError("Wrong attribution evidence scope")
    copy(sidecar, "ATTRIBUTION.md", document["sidecar_sha256"])
    copy(inventory, "attribution_evidence.json")
    destinations = set()
    for entry in document["files"]:
        destination = relative_path(entry["capsule_path"])
        if destination.parts[0] != "attribution_sources" or destination in destinations:
            raise ValueError("Invalid or duplicate attribution destination")
        destinations.add(destination)
        source = Path(entry["path"])
        if not source.is_absolute() or source.stat().st_size != entry["bytes"]:
            raise ValueError("Changed attribution source size or path")
        copy(source, destination.as_posix(), entry["sha256"])
    return dict(
        inventory_sha256=sha(inventory),
        sidecar_sha256=document["sidecar_sha256"],
        source_documents=len(destinations),
        posthoc=True,
    )


def build(study, output, preacquisition_root=None):
    """Verify workspace bindings, then copy immutable endpoint evidence."""
    from experiments.mission_transfer_comparison_v4.protocol import profile_paths, verify

    study, output = study.resolve(), output.resolve()
    protocol = verify(study)
    completion = read(study / "completion.json")
    if not completion["complete"]:
        raise ValueError("Study is incomplete")
    analysis = read(study / "analysis.json")
    validation = read(study / "validation.json")
    witnesses = read(study / "witness_validation.json")
    cells = grid(protocol, analysis["policy_cases"])
    witness_grid(cells, witnesses["cases"])
    if any(not x["complete"] for x in (analysis, validation, witnesses)):
        raise ValueError("Complete study and witness validation required")
    if any(
        x["protocol_sha256"] != sha(study / "protocol.json")
        for x in (analysis, validation, witnesses)
    ):
        raise ValueError("Report protocol binding differs")
    output.mkdir(parents=True, exist_ok=False)
    files = {}

    def copy(source, name, expected=None):
        relative = relative_path(name)
        digest = sha(source)
        if expected is not None and digest != expected:
            raise ValueError("Changed audited input: " + str(source))
        if name in files and files[name] != digest:
            raise ValueError("Conflicting export destination: " + name)
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if sha(destination) != digest:
            raise ValueError("Export byte mismatch")
        files[name] = digest

    repo = Path(__file__).resolve().parents[1]
    copy(Path(__file__), "review.py")
    for name in SCORER_FILES:
        copy(repo / name, name, protocol["implementation_sha256"][str((repo / name).resolve())])
    copy(study / "completion.json", "study/completion.json")
    for name, expected in completion["artifacts_sha256"].items():
        copy(study / relative_path(name), "study/" + name, expected)
    copy(study / "witness_validation.json", "study/witness_validation.json")
    optional_reports = (
        "interpretation.json",
        "INTERPRETATION.md",
        "recall_transfer.json",
        "RECALL_TRANSFER.md",
        "recall_transfer_figure.json",
        "host_timing.json",
        "HOST_TIMING.md",
        "paper_release.json",
        "PAPER_READINESS.md",
        "energy_robustness.json",
        "ENERGY_ROBUSTNESS.md",
        "verification_before_inference.json",
        "verification_posthoc_helpers.json",
        "verification_posthoc_final.json",
        "verification_capsule_repair_v2.json",
        "capsule_repair_history_v2.json",
    )
    included_reports = []
    for name in optional_reports:
        if (study / name).is_file():
            copy(study / name, "study/" + name)
            included_reports.append(name)
    if "capsule_repair_history_v2.json" in included_reports:
        repair = read(study / "capsule_repair_history_v2.json")
        if (
            repair["schema_version"] != "mission_transfer_capsule_inventory_repair_v2"
            or repair["main_results_unchanged"] is not True
            or repair["study_protocol_sha256"] != sha(study / "protocol.json")
        ):
            raise ValueError("Wrong release repair history")
        for entry in repair["superseded_artifacts"]:
            source = Path(entry["path"])
            if sha(source) != entry["sha256"] or source.stat().st_size != entry["bytes"]:
                raise ValueError("Superseded release evidence changed")
            if entry.get("export_path"):
                destination = relative_path(entry["export_path"])
                if destination.parts[0] != "release_history":
                    raise ValueError("Wrong release-history destination")
                copy(source, destination.as_posix(), entry["sha256"])
    attribution = attribution_evidence(study, copy)
    if "recall_transfer_figure.json" in included_reports:
        figure = read(study / "recall_transfer_figure.json")
        if (
            figure["complete"] is not True
            or figure["posthoc"] is not True
            or figure["evaluation_refitting"] is not False
            or figure["protocol_sha256"] != sha(study / "protocol.json")
            or figure["diagnostic_sha256"] != sha(study / "recall_transfer.json")
            or set(figure["artifacts_sha256"]) != {"recall_transfer.png", "recall_transfer.pdf"}
        ):
            raise ValueError("Incomplete or unbound recall figure")
        for name, expected in figure["artifacts_sha256"].items():
            copy(study / name, "study/" + name, expected)
    cohort = Path(protocol["cohort"])
    cohort_protocol = read(cohort / "protocol.json")
    for name in (
        "protocol.json",
        "sensor_gate.json",
        "physical_route_validation.json",
        "acquisition_completion.json",
        "feasibility_diagnostics.json",
        "FEASIBILITY.md",
    ):
        copy(cohort / name, "cohort/" + name)
    for name in (
        "environment.json",
        "runtime_inventory.json",
        "powercap_probe.json",
        "runtime_reproduction_inventory.json",
        "verification_before_model_evaluation.json",
    ):
        if (cohort / name).is_file():
            copy(cohort / name, "cohort/" + name)
    for name, expected in cohort_protocol["frozen_files"].items():
        copy(cohort / relative_path(name), "cohort/" + name, expected)
    for name, expected in protocol["frozen_files"].items():
        copy(study / relative_path(name), "study/" + name, expected)
    diagnostic = read(cohort / "feasibility_diagnostics.json")
    for flight, data in diagnostic["flights"].items():
        copy(
            cohort / flight / "diagnostic_outcomes.json",
            f"cohort/{flight}/diagnostic_outcomes.json",
            data["outcomes_sha256"],
        )
        copy(
            cohort / flight / "predictions/manifest.json",
            f"cohort/{flight}/predictions/manifest.json",
            data["source_manifest_sha256"],
        )
        for name in ("recording/recording.json", "predictions/inference.json"):
            copy(cohort / flight / name, f"cohort/{flight}/{name}")
    assets = Path(protocol["assets"])
    copy(assets / "manifest.json", "assets/manifest.json", cohort_protocol["asset_manifest_sha256"])
    copy(
        assets / "loader_audit/validation.json",
        "assets/loader_audit/validation.json",
        cohort_protocol["installed_collision_loader_audit_sha256"],
    )
    history = protocol["rejected_acquisition"]
    rejected = Path(history["cohort"])
    if preacquisition_root is None and "_v2_" in rejected.name:
        candidate = rejected.with_name(rejected.name.replace("_v2_", "_v1_", 1))
        if (candidate / "PRE_ACQUISITION_REVISION.json").exists():
            preacquisition_root = candidate
    preacquisition = (
        preacquisition_history(preacquisition_root, copy)
        if preacquisition_root is not None
        else None
    )
    copy(
        rejected / "ADMISSION_REJECTED.json",
        "rejected_v2/ADMISSION_REJECTED.json",
        history["rejection_sha256"],
    )
    for name, expected in history["evidence_sha256"].items():
        copy(rejected / relative_path(name), "rejected_v2/" + name, expected)
    for flight, expected in history["recording_sha256"].items():
        copy(
            rejected / flight / "recording/recording.json",
            f"rejected_v2/{flight}/recording/recording.json",
            expected,
        )
    admission = Path(protocol["admission"])
    copy(
        admission / "routing_admission_v1.json",
        "admission/routing_admission_v1.json",
        protocol["routing_admission_sha256"],
    )
    certificate = read(admission / "routing_admission_v1.json")
    v3_history = certificate["original_evidence"]
    copy(
        cohort / "ADMISSION_REJECTED.json",
        "rejected_v3/ADMISSION_REJECTED.json",
        v3_history["rejection_sha256"],
    )
    for name, expected in v3_history["evidence_sha256"].items():
        copy(cohort / relative_path(name), "rejected_v3/" + name, expected)
    for flight, expected in v3_history["recording_sha256"].items():
        copy(
            cohort / flight / "recording/recording.json",
            f"rejected_v3/{flight}/recording/recording.json",
            expected,
        )
    source_inventory = {}

    def source_snapshot(source, expected):
        source = source.resolve()
        relative = "source_snapshot/" + source.relative_to(repo).as_posix()
        copy(source, relative, expected)
        source_inventory[str(source)] = dict(relative=relative, sha256=expected)

    for name, expected in protocol["implementation_sha256"].items():
        source_snapshot(Path(name), expected)
    frozen_source_inventory = dict(source_inventory)
    for name, path in profile_paths(protocol).items():
        copy(path, "profiles/" + name + ".json", protocol["profile_sha256"][name])
    audited_policy_rows, audited_witness_rows = [], []
    for flight in protocol["cases"]:
        name = f"validation_{flight}.json"
        copy(study / name, "study/" + name, validation["condition_audits_sha256"][name])
        audit = read(study / name)
        for name, expected in audit["source_summaries_sha256"].items():
            copy(study / relative_path(name), "study/" + name, expected)
            audited_policy_rows.extend(read(study / name)["cases"])
        directory = study / "witnesses" / flight
        copy(
            directory / "validation.json",
            f"study/witnesses/{flight}/validation.json",
            witnesses["independent_audits_sha256"][flight],
        )
        audit = read(directory / "validation.json")
        copy(
            directory / "plan.json",
            f"study/witnesses/{flight}/plan.json",
            audit["search_plan_sha256"],
        )
        plan = read(directory / "plan.json")
        for name, expected in plan["helper_sha256"].items():
            source_snapshot(Path(name), expected)
        for name, expected in audit["source_summaries_sha256"].items():
            copy(study / relative_path(name), "study/" + name, expected)
            audited_witness_rows.extend(read(study / name)["cases"])
        for row in [r for r in witnesses["cases"] if r["flight"] == flight]:
            name = f"witnesses/{flight}/{row['mission']}/search.json"
            copy(study / name, "study/" + name, row["search_sha256"])
            search_result = read(study / name)
            if search_result["conclusion"] != row["search_conclusion"] or (
                search_result["schedule"] is not None
            ) != ("path" in row):
                raise ValueError("Witness search and paid replay status differ")
    grid(protocol, audited_policy_rows)
    witness_grid(cells, audited_witness_rows)
    indexed = {(r["flight"], r["mission"], r["policy"]): r for r in analysis["policy_cases"]}
    for row in audited_policy_rows:
        published = indexed[row["flight"], row["mission"], row["policy"]]
        if any(published.get(k) != v for k, v in row.items()):
            raise ValueError("Published policy row differs from independent audit")
    if sorted(audited_witness_rows, key=lambda r: (r["flight"], r["mission"])) != sorted(
        witnesses["cases"], key=lambda r: (r["flight"], r["mission"])
    ):
        raise ValueError("Witness search denominator differs from independent audits")
    exported = []
    for role, rows in (("policy", analysis["policy_cases"]), ("witness", witnesses["cases"])):
        for row in rows:
            if "path" not in row:
                continue  # The full failed-search row remains in witness_searches below.
            relative = f"{role}/{row['flight']}/{row['mission']}/{row['policy'].replace(':', '_')}"
            source = study / relative_path(row["path"])
            for name, key in (
                ("outcome.json", "outcome_sha256"),
                ("execution/events.json", "events_sha256"),
                ("execution/results.json", "results_sha256"),
            ):
                copy(source / name, relative + "/" + name, row[key])
            intent = study / row["flight"] / row["mission"] / "intent.json"
            if read(source / "intent.json") != read(intent):
                raise ValueError("Replay intent differs from frozen condition")
            copy(source / "intent.json", relative + "/intent.json")
            exported.append(
                dict(
                    role=role,
                    flight=row["flight"],
                    mission=row["mission"],
                    policy=row["policy"],
                    relative=relative,
                    success=row["full_success"],
                )
            )
    posthoc_source_reports = {
        study / "completion.json": "experiments/mission_transfer_comparison_v4/report.py",
        study / "interpretation.json": "experiments/mission_transfer_interpretation.py",
        study / "recall_transfer.json": "experiments/mission_transfer_recall_check.py",
        study / "recall_transfer_figure.json": "experiments/mission_transfer_recall_plot.py",
        study / "host_timing.json": "experiments/mission_transfer_host_timing.py",
        study / "energy_robustness.json": "experiments/mission_transfer_energy.py",
    }
    posthoc_source_reports[cohort / "runtime_reproduction_inventory.json"] = (
        "experiments/mission_transfer_runtime_inventory.py"
    )
    for report, module in posthoc_source_reports.items():
        if report.exists():
            report_document = read(report)
            source_snapshot(repo / module, report_document["source_sha256"])
            for helper, expected in report_document.get("helper_sha256", {}).items():
                source_snapshot(Path(helper), expected)
    # Auxiliary test evidence is inspectable but does not expand the execution freeze.
    for relative in list(files):
        name = Path(relative)
        if (
            name.parent.as_posix() in ("study", "cohort")
            and name.name.startswith("verification_")
            and name.suffix == ".json"
        ):
            for test, expected in read(output / name).get("tests_sha256", {}).items():
                source = Path(test)
                if not source.is_absolute():
                    source = repo / relative_path(test)
                source_snapshot(source, expected)
    source_snapshot(Path(__file__), sha(Path(__file__)))
    manifest = dict(
        schema_version=SCHEMA,
        files_sha256=files,
        policy_count=len(protocol["policies"]),
        condition_count=len(cells),
        execution_count=len(analysis["policy_cases"]),
        witness_search_count=len(witnesses["cases"]),
        paid_witness_count=sum("path" in r for r in witnesses["cases"]),
        full_contract_witnesses=sum(r["full_contract_witness"] for r in witnesses["cases"]),
        independent_layouts=protocol["independent_evaluation_layouts"],
        maps=1,
        rows=exported,
        witness_searches=witnesses["cases"],
        frozen_source_inventory=frozen_source_inventory,
        source_inventory=source_inventory,
        protocol_sha256=sha(study / "protocol.json"),
        completion_sha256=sha(study / "completion.json"),
        rejected_acquisition=history,
        preacquisition_revision=preacquisition,
        routing_admission_sha256=protocol["routing_admission_sha256"],
        rejected_strict_pairing_acquisition=v3_history,
        included_posthoc_reports=included_reports,
        attribution_evidence=attribution,
        provenance={k: protocol[k] for k in PROVENANCE},
        scope=(
            "Canonical endpoints from audited exported logs; not complete experiment reproduction"
        ),
    )
    (output / "README.txt").write_text(
        "MISSION TRANSFER ENDPOINT REVIEW CAPSULE\n\n"
        "Python 3.11 or later; no installed packages required:\n"
        "    python3 -I -S -B review.py check\n\n"
        "Release v2 rejects all unlisted files, symlinks, nonregular files, and Python "
        "bytecode/cache directories before importing the scorer. README.txt is hashed; "
        "only manifest.json is outside its own hash inventory. The checker also disables "
        "bytecode writes. This repairs the prior endpoint release's untracked-cache "
        "exposure; the frozen experiment and endpoint definitions are unchanged.\n"
        f"The frozen grid has {manifest['policy_count']} policies, {len(cells)} conditions, "
        f"and {manifest['execution_count']} executions. Failed runs remain in the denominator.\n"
        f"All {len(cells)} evaluator-only witness searches are retained, including failed "
        "searches and failed paid replays. Failure to find a witness is inconclusive.\n"
        "The capsule verifies hashes and recomputes canonical mission endpoints. It does "
        "not repeat raw-image matching, physics, inference, policy selection or witness search.\n"
        "Raw RGB and geometry GT from the rejected v2 acquisition informed the v3 geometry "
        "repair. No model or policy outputs existed then; policies and costs stayed fixed. "
        "The v3 acquisition then failed strict cross-lighting pose equality. Its scope was "
        "amended after capture and before model outputs to exact within-recording policy "
        "contrasts. Intent and battery share exact inputs; lighting is observational. "
        "This is neither a wholly prospective protocol nor a sensor-unseen holdout.\n"
        "Both rejected acquisitions' notes, gates and recording manifests are included, "
        "along with the new admission certificate. Full frozen and posthoc source snapshots "
        "are included; model weights, "
        "raw sensor arrays and simulator assets remain outside this endpoint-only capsule.\n"
        + (
            "The v1 source snapshots are included as a superseded pre-acquisition design: "
            "zero physical captures and zero model predictions, not an additional failed flight.\n"
            if preacquisition
            else ""
        )
    )
    files["README.txt"] = sha(output / "README.txt")
    manifest["inventory_exceptions"] = ["manifest.json"]
    save(output / "manifest.json", manifest)
    exact_inventory(output, manifest)
    print(
        json.dumps(
            dict(
                complete=True,
                files=len(files),
                executions=len(analysis["policy_cases"]),
                witness_searches=len(witnesses["cases"]),
                paid_witnesses=manifest["paid_witness_count"],
            )
        ),
        flush=True,
    )
    return manifest


def check(root):
    """Recompute endpoints with no workspace paths or third-party dependencies."""
    root = root.resolve()
    manifest = read(root / "manifest.json")
    if manifest["schema_version"] != SCHEMA:
        raise ValueError("Wrong capsule schema")
    if manifest.get("inventory_exceptions") != ["manifest.json"]:
        raise ValueError("Wrong inventory exception policy")
    actual_files = exact_inventory(root, manifest)
    files = manifest["files_sha256"]

    def bound(name, expected):
        relative_path(name)
        if files.get(name) != expected:
            raise ValueError("Missing or changed evidence binding: " + name)

    for name in (
        *SCORER_FILES,
        "review.py",
        "study/protocol.json",
        "study/completion.json",
        "study/analysis.json",
        "study/validation.json",
        "study/witness_validation.json",
    ):
        if name not in files:
            raise ValueError("Unpinned required capsule artifact: " + name)
    protocol = read(root / "study/protocol.json")
    if protocol["schema_version"] != "mission_map_transfer_comparison_v4":
        raise ValueError("Wrong comparison protocol")
    if any(protocol[k] is not v for k, v in PROVENANCE.items()):
        raise ValueError("Changed sensor-informed provenance")
    if manifest["provenance"] != {k: protocol[k] for k in PROVENANCE}:
        raise ValueError("Capsule provenance differs from protocol")
    bound("study/protocol.json", manifest["protocol_sha256"])
    bound("study/completion.json", manifest["completion_sha256"])
    completion = read(root / "study/completion.json")
    if not completion["complete"]:
        raise ValueError("Incomplete study")
    for name, expected in completion["artifacts_sha256"].items():
        bound("study/" + name, expected)
    for name, expected in protocol["frozen_files"].items():
        bound("study/" + name, expected)
    snapshots = manifest["frozen_source_inventory"]
    if set(snapshots) != set(protocol["implementation_sha256"]):
        raise ValueError("Incomplete frozen source inventory")
    for original, expected in protocol["implementation_sha256"].items():
        entry = snapshots[original]
        if entry["sha256"] != expected or manifest["source_inventory"].get(original) != entry:
            raise ValueError("Changed frozen source inventory")
        bound(entry["relative"], expected)
    for entry in manifest["source_inventory"].values():
        bound(entry["relative"], entry["sha256"])
    attribution = manifest.get("attribution_evidence")
    if attribution is not None:
        bound("attribution_evidence.json", attribution["inventory_sha256"])
        bound("ATTRIBUTION.md", attribution["sidecar_sha256"])
        inventory = read(root / "attribution_evidence.json")
        if (
            inventory["schema_version"] != "mission_transfer_attribution_v1"
            or inventory["posthoc"] is not True
            or inventory["sidecar"] != "ATTRIBUTION.md"
            or inventory["sidecar_sha256"] != attribution["sidecar_sha256"]
            or len(inventory["files"]) != attribution["source_documents"]
        ):
            raise ValueError("Changed attribution inventory")
        destinations = set()
        for entry in inventory["files"]:
            destination = relative_path(entry["capsule_path"])
            if destination.parts[0] != "attribution_sources" or destination in destinations:
                raise ValueError("Invalid or duplicate attribution destination")
            destinations.add(destination)
            bound(destination.as_posix(), entry["sha256"])
            if (root / destination).stat().st_size != entry["bytes"]:
                raise ValueError("Changed attribution document size")
    for name in SCORER_FILES:
        matching = [v for k, v in snapshots.items() if k.endswith("/" + name)]
        if len(matching) != 1:
            raise ValueError("Canonical scorer absent from frozen inventory")
        bound(name, matching[0]["sha256"])
    for name, expected in protocol["profile_sha256"].items():
        bound("profiles/" + name + ".json", expected)
    preacquisition = manifest.get("preacquisition_revision")
    if preacquisition is not None:
        bound("preacquisition_v1/protocol.json", preacquisition["protocol_sha256"])
        bound(
            "preacquisition_v1/source_snapshot_manifest.json",
            preacquisition["snapshot_manifest_sha256"],
        )
        bound(
            "preacquisition_v1/PRE_ACQUISITION_REVISION.json",
            preacquisition["revision_note_sha256"],
        )
        revision = read(root / "preacquisition_v1/PRE_ACQUISITION_REVISION.json")
        inventory = read(root / "preacquisition_v1/source_snapshot_manifest.json")
        old_protocol = read(root / "preacquisition_v1/protocol.json")
        if (
            revision["status"] != "superseded_before_capture"
            or revision["physical_captures"] != 0
            or revision["model_predictions"] != 0
            or inventory["verified"] is not True
            or inventory["protocol_sha256"] != preacquisition["protocol_sha256"]
            or len(inventory["files"]) != preacquisition["source_files"]
            or {r["original_path"]: r["sha256"] for r in inventory["files"]}
            != old_protocol["implementation_sha256"]
        ):
            raise ValueError("Changed pre-acquisition revision history")
        for entry in inventory["files"]:
            bound("preacquisition_v1/" + entry["snapshot"], entry["sha256"])
    bound("cohort/protocol.json", protocol["cohort_protocol_sha256"])
    cohort = read(root / "cohort/protocol.json")
    for name, expected in cohort["frozen_files"].items():
        bound("cohort/" + name, expected)
    bound("assets/manifest.json", cohort["asset_manifest_sha256"])
    bound("assets/loader_audit/validation.json", cohort["installed_collision_loader_audit_sha256"])
    sensor = read(root / "cohort/sensor_gate.json")
    route = read(root / "cohort/physical_route_validation.json")
    acquisition = read(root / "cohort/acquisition_completion.json")
    if (
        sensor["passed"] is not False
        or route["all_passed"] is not True
        or acquisition["acquisition_complete"] is not True
        or acquisition["admitted_to_model_evaluation"] is not False
        or acquisition["complete"] is not False
        or sensor["paired_poses_identical"] is not False
        or sensor["model_predictions_used"] is not False
        or route["model_predictions_used"] is not False
    ):
        raise ValueError("Original strict sensor rejection was not preserved")
    bound("cohort/protocol.json", sensor["protocol_sha256"])
    bound("cohort/physical_route_validation.json", sensor["physical_route_validation_sha256"])
    bound("cohort/sensor_gate.json", acquisition["sensor_gate_sha256"])
    bound("admission/routing_admission_v1.json", protocol["routing_admission_sha256"])
    if manifest["routing_admission_sha256"] != protocol["routing_admission_sha256"]:
        raise ValueError("Capsule routing admission differs")
    admission = read(root / "admission/routing_admission_v1.json")
    scope = admission["scope"]
    if (
        admission["admitted"] is not True
        or scope["original_strict_gate_passed"] is not False
        or scope["fresh_full_prospective_test"] is not False
        or scope["cross_lighting_pose_equality_required"] is not False
        or scope["cross_lighting_numeric_tolerance_used"] is not False
        or scope["pure_lighting_causal_effect_identified"] is not False
        or scope["same_available_rgb_pose_timestamp_bank_across_policies"] is not True
        or admission["pre_model_attestation"]
        != {
            "neural_prediction_attempts": 0,
            "policy_replay_attempts": 0,
            "rgb_and_ground_truth_inspected": True,
            "model_and_policy_outcomes_inspected": False,
        }
    ):
        raise ValueError("Changed within-recording scope amendment")
    v3_history = admission["original_evidence"]
    if manifest["rejected_strict_pairing_acquisition"] != v3_history:
        raise ValueError("Changed v3 strict rejection history")
    bound("rejected_v3/ADMISSION_REJECTED.json", v3_history["rejection_sha256"])
    for name, expected in v3_history["evidence_sha256"].items():
        bound("rejected_v3/" + name, expected)
    for flight, expected in v3_history["recording_sha256"].items():
        bound(f"rejected_v3/{flight}/recording/recording.json", expected)
        bound(f"cohort/{flight}/recording/recording.json", expected)
    bound("rejected_v3/study/protocol.json", protocol["pre_amendment_protocol_sha256"])
    bound(
        "rejected_v3/acquisition_order.json",
        protocol["original_method_freeze"]["acquisition_order_sha256"],
    )
    original_order = read(root / "rejected_v3/acquisition_order.json")
    if (
        original_order["methods_frozen_before_this_capture_run"] is not True
        or original_order["study_protocol_sha256"] != protocol["pre_amendment_protocol_sha256"]
    ):
        raise ValueError("Original pre-capture method freeze differs")
    v3_note = read(root / "rejected_v3/ADMISSION_REJECTED.json")
    if (
        v3_note["original_admission_passed"] is not False
        or v3_note["neural_predictions_started"] is not False
        or v3_note["policy_replays_started"] is not False
    ):
        raise ValueError("Original strict rejection is not a pre-model decision")
    history = protocol["rejected_acquisition"]
    if manifest["rejected_acquisition"] != history:
        raise ValueError("Changed rejected-acquisition history")
    bound("rejected_v2/ADMISSION_REJECTED.json", history["rejection_sha256"])
    note = read(root / "rejected_v2/ADMISSION_REJECTED.json")
    if (
        note["acquisition_complete"] is not True
        or any(
            note[k] is not False
            for k in (
                "admitted_to_model_evaluation",
                "neural_predictions_started",
                "policy_replays_started",
            )
        )
        or note["evidence_sha256"] != history["evidence_sha256"]
        or note["recording_sha256"] != history["recording_sha256"]
    ):
        raise ValueError("Rejected acquisition was not a sensor-only failure")
    for name, expected in history["evidence_sha256"].items():
        bound("rejected_v2/" + name, expected)
    for flight, expected in history["recording_sha256"].items():
        bound(f"rejected_v2/{flight}/recording/recording.json", expected)
    failed_sensor = read(root / "rejected_v2/sensor_gate.json")
    failed_route = read(root / "rejected_v2/physical_route_validation.json")
    if (
        failed_sensor["passed"] is not False
        or failed_route["all_passed"] is not False
        or failed_sensor["model_predictions_used"] is not False
        or failed_route["model_predictions_used"] is not False
    ):
        raise ValueError("Rejected acquisition gate history differs")
    analysis = read(root / "study/analysis.json")
    witnesses = read(root / "study/witness_validation.json")
    validation = read(root / "study/validation.json")
    for report in (analysis, witnesses, validation):
        if not report["complete"] or report["protocol_sha256"] != manifest["protocol_sha256"]:
            raise ValueError("Incomplete or unbound result report")
    rows = manifest["rows"]
    policies = [r for r in rows if r["role"] == "policy"]
    paid = [r for r in rows if r["role"] == "witness"]
    if len(rows) != len(policies) + len(paid):
        raise ValueError("Unknown exported execution role")
    cells = grid(protocol, policies)
    grid(protocol, analysis["policy_cases"])
    searches = manifest["witness_searches"]
    witness_grid(cells, searches)
    if searches != witnesses["cases"]:
        raise ValueError("Changed witness-search outcomes or denominator")
    for row in searches:
        name = f"study/witnesses/{row['flight']}/{row['mission']}/search.json"
        bound(name, row["search_sha256"])
        search_result = read(root / name)
        if search_result["conclusion"] != row["search_conclusion"] or (
            search_result["schedule"] is not None
        ) != ("path" in row):
            raise ValueError("Witness search and paid replay status differ")
    expected_paid = {(r["flight"], r["mission"]) for r in searches if "path" in r}
    if (
        len(paid) != len(expected_paid)
        or {(r["flight"], r["mission"]) for r in paid} != expected_paid
    ):
        raise ValueError("Incomplete paid witness export")
    expected_counts = dict(
        policy_count=len(protocol["policies"]),
        condition_count=len(cells),
        execution_count=len(policies),
        witness_search_count=len(searches),
        paid_witness_count=len(paid),
        full_contract_witnesses=sum(r["full_contract_witness"] for r in searches),
    )
    if any(manifest[k] != v for k, v in expected_counts.items()):
        raise ValueError("Incorrect capsule denominator")
    if witnesses["full_contract_witnesses"] != expected_counts["full_contract_witnesses"]:
        raise ValueError("Witness success summary differs")
    # Verify before importing the capsule's scorer; cwd and site packages are unnecessary.
    sys.path.insert(0, str(root))
    from experiments.lake_intent_contract import IntentContract, evaluate

    scorer_path = Path(sys.modules[IntentContract.__module__].__file__).resolve()
    if not scorer_path.is_relative_to(root):
        raise ValueError("Run review.py in a fresh isolated Python process")
    summary = {name: 0 for name in protocol["policies"]}
    witness_successes = 0
    policy_lookup = {(r["flight"], r["mission"], r["policy"]): r for r in analysis["policy_cases"]}
    witness_lookup = {(r["flight"], r["mission"]): r for r in searches}
    for row in rows:
        directory = root / relative_path(row["relative"])
        original = (
            policy_lookup[row["flight"], row["mission"], row["policy"]]
            if row["role"] == "policy"
            else witness_lookup[row["flight"], row["mission"]]
        )
        for name, key in (
            ("outcome.json", "outcome_sha256"),
            ("execution/events.json", "events_sha256"),
            ("execution/results.json", "results_sha256"),
        ):
            bound(row["relative"] + "/" + name, original[key])
        if row["policy"] != original["policy"] or row["success"] != original["full_success"]:
            raise ValueError("Exported execution status differs")
        intent = read(directory / "intent.json")
        frozen_intent = read(root / "study" / row["flight"] / row["mission"] / "intent.json")
        if intent != frozen_intent:
            raise ValueError("Exported execution intent differs")
        events = read(directory / "execution/events.json")
        runtime = read(directory / "execution/results.json")
        outcome = read(directory / "outcome.json")
        recomputed = evaluate(IntentContract(intent), events, runtime)
        for key in ("mission_success", "constraints", "failed_constraints", "phases", "reporting"):
            if recomputed[key] != outcome[key]:
                raise ValueError("Canonical endpoint mismatch: " + row["relative"])
        if recomputed["mission_success"] != row["success"]:
            raise ValueError("Exported success mismatch")
        if row["role"] == "policy":
            summary[row["policy"]] += recomputed["mission_success"]
        else:
            witness_successes += recomputed["mission_success"]
    expected = {r["policy"]: r["successes"] for r in analysis["policy_summary"]}
    if summary != expected:
        raise ValueError("Published summary mismatch")
    if witness_successes != expected_counts["full_contract_witnesses"]:
        raise ValueError("Paid witness endpoint total differs")
    result = dict(
        complete=True,
        portable_endpoint_reproduction=True,
        exact_inventory_verified=True,
        regular_files_verified=len(actual_files),
        inventory_exceptions=["manifest.json"],
        bytecode_forbidden=True,
        bytecode_writes_disabled=sys.dont_write_bytecode,
        policy_executions=len(policies),
        conditions=len(cells),
        policies=len(summary),
        witness_searches=len(searches),
        paid_witnesses=len(paid),
        witness_successes=witness_successes,
        searches_without_paid_replay=len(searches) - len(paid),
        unresolved_searches=sum(r["search_conclusion"] == "search_inconclusive" for r in searches),
        certified_resource_infeasible_searches=sum(
            r["search_conclusion"] == "certified_resource_infeasible" for r in searches
        ),
        failed_paid_witnesses=len(paid) - witness_successes,
        failed_search_is_inconclusive=True,
        policy_successes=summary,
        rejected_acquisition_history_verified=True,
        rejected_strict_pairing_history_verified=True,
        postcapture_scope_amendment_verified=True,
        superseded_preacquisition_history_verified=preacquisition is not None,
        lighting_causal_effect_identified=False,
        frozen_source_files=len(snapshots),
        raw_image_reconstruction=False,
        neural_inference_repeated=False,
        physics_repeated=False,
        policy_decisions_repeated=False,
        witness_search_repeated=False,
        isolated_python=bool(sys.flags.isolated),
        site_packages_disabled=bool(sys.flags.no_site),
        scorer_loaded_from=str(scorer_path),
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


def package(capsule, archive, report):
    """Relocate an archive and prove isolated scoring plus a hash negative control."""
    import subprocess
    import tarfile
    import tempfile

    capsule, archive, report = (p.resolve() for p in (capsule, archive, report))
    if archive.exists() or report.exists():
        raise FileExistsError("Preserve the prior archive and portability report")
    manifest = read(capsule / "manifest.json")
    if manifest["schema_version"] != SCHEMA:
        raise ValueError("Wrong capsule schema")
    actual_files = exact_inventory(capsule, manifest)
    archive.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "x:gz", compresslevel=6) as bundle:
        for name in sorted(actual_files):
            bundle.add(capsule / name, arcname=capsule.name + "/" + name, recursive=False)
    scratch = Path(tempfile.mkdtemp(prefix="mission_transfer_endpoint_"))
    with tarfile.open(archive, "r:gz") as bundle:
        bundle.extractall(scratch, filter="data")
    extracted = scratch / capsule.name
    command = [sys.executable, "-I", "-S", "-B", str(extracted / "review.py"), "check"]

    def run(label):
        proc = subprocess.run(command, cwd=scratch, capture_output=True, text=True, check=False)
        log = report.with_name(report.stem + "_" + label + ".log")
        log.write_text(proc.stdout + proc.stderr)
        return proc, log

    baseline, baseline_log = run("clean")
    if baseline.returncode:
        raise RuntimeError("Relocated endpoint check failed: " + str(baseline_log))
    result = json.loads(baseline.stdout)
    if not (
        result["complete"]
        and result["isolated_python"]
        and result["site_packages_disabled"]
        and result["exact_inventory_verified"]
        and result["bytecode_forbidden"]
        and result["bytecode_writes_disabled"]
        and result["regular_files_verified"] == len(actual_files)
        and result["policy_executions"] == manifest["execution_count"]
        and result["conditions"] == manifest["condition_count"]
        and result["witness_searches"] == manifest["witness_search_count"]
        and result["paid_witnesses"] == manifest["paid_witness_count"]
        and result["witness_successes"] == manifest["full_contract_witnesses"]
        and Path(result["scorer_loaded_from"]).is_relative_to(extracted)
    ):
        raise ValueError("Incomplete isolated endpoint reproduction")
    name = manifest["rows"][0]["relative"] + "/outcome.json"
    target = extracted / relative_path(name)
    original = target.read_bytes()
    try:
        target.write_bytes(original + b"\n")
        tampered, tampered_log = run("tampered")
    finally:
        target.write_bytes(original)
    if tampered.returncode == 0 or "Changed capsule artifact" not in tampered.stderr:
        raise ValueError("Changed-artifact negative control was not rejected")
    if sha(target) != manifest["files_sha256"][name]:
        raise ValueError("Temporary copy restoration failed")
    if sha(extracted / "manifest.json") != sha(capsule / "manifest.json"):
        raise ValueError("Archive manifest differs from source")
    cache = extracted / "experiments" / "__pycache__"
    cache.mkdir()
    unlisted = cache / "lake_intent_contract.injected.pyc"
    try:
        unlisted.write_bytes(b"unlisted bytecode negative control")
        injected, injected_log = run("unlisted_bytecode")
    finally:
        unlisted.unlink()
        cache.rmdir()
    if injected.returncode == 0 or "Forbidden bytecode" not in injected.stderr:
        raise ValueError("Unlisted-bytecode negative control was not rejected")
    exact_inventory(extracted, manifest)
    audit = dict(
        complete=True,
        schema_version="mission_transfer_endpoint_portability_v2",
        scope="Relocated canonical scoring; not complete experiment reproduction",
        archive=str(archive),
        archive_sha256=sha(archive),
        archive_bytes=archive.stat().st_size,
        capsule_manifest_sha256=sha(capsule / "manifest.json"),
        files_verified=len(manifest["files_sha256"]),
        archive_regular_files=len(actual_files),
        inventory_exceptions=["manifest.json"],
        relocated_root=str(extracted),
        command=command,
        python_version=sys.version,
        endpoint_result=result,
        negative_control=dict(
            artifact=name,
            modification="Append newline to disposable outcome",
            rejected=True,
            exit_code=tampered.returncode,
            temporary_copy_restored=True,
        ),
        unlisted_bytecode_negative_control=dict(
            artifact=str(unlisted.relative_to(extracted)),
            rejected=True,
            exit_code=injected.returncode,
            removed_after_test=True,
        ),
        logs_sha256={str(p): sha(p) for p in (baseline_log, tampered_log, injected_log)},
        source_sha256=sha(Path(__file__)),
    )
    save(report, audit)
    print(
        json.dumps(
            {k: audit[k] for k in ("complete", "archive", "archive_bytes", "files_verified")}
        )
    )
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("build", "check", "package"))
    parser.add_argument("--study", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--capsule", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--preacquisition-history", type=Path)
    args = parser.parse_args()
    if args.stage == "build":
        build(args.study, args.output, args.preacquisition_history)
    elif args.stage == "package":
        package(args.capsule, args.archive, args.report)
    else:
        check(args.output or Path(__file__).resolve().parent)
