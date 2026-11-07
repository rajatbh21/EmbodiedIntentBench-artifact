"""Inventory-only preparation repair; all virtual-network runtime methods are unchanged.

The original preflight stopped before writing a plan or executing any policy.
This standalone preparation copy preserves its source snapshots and exact spec.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import shutil
import time
from dataclasses import asdict
from pathlib import Path

from experiments.mission_network_study_v1 import study as original_study
from experiments.mission_network_study_v1.study import (
    CALIBRATION,
    CONFIGS,
    DEVELOPMENT,
    FLIGHTS,
    LINKS,
    MISSION,
    POLICIES,
    REPO,
    ActionDescriptor,
    authored_traces,
    calibrated_arm,
    cell_name,
    derivative_documents,
    freeze_specification,
    implementation_inventory,
    imports,
    policy_for,
    read,
    require,
    save,
    scenario_for,
    sha,
    specification,
    verify,
)

OUTPUT = REPO / "results/mission_network_study_v2_20261027"
RETAINED = REPO / "results/mission_network_study_v1_20261027"


def runtime_inventory(distribution_name, module_name):
    try:
        distribution_version = importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        distribution_version = None
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        return dict(
            distribution_version=distribution_version,
            module_available=False,
            module_version=None,
            module_file=None,
            unavailable_reason="Optional module is not installed in the active interpreter",
        )
    return dict(
        distribution_version=distribution_version,
        module_available=True,
        module_version=getattr(module, "__version__", None),
        module_file=getattr(module, "__file__", None),
        unavailable_reason=None,
    )


def prepare(root, calibration=CALIBRATION):
    from experiments import mission_network_audit
    from experiments.mission_execution_profile import verify as verify_calibration

    root, calibration = Path(root), Path(calibration)
    require(
        sha(root / "specification.json") == sha(RETAINED / "specification.json"),
        "Inventory repair must preserve the original specification bytes",
    )
    require(
        not (RETAINED / "plan.json").exists() and not (RETAINED / "runs").exists(),
        "This inventory repair applies only to the zero-execution preflight",
    )
    require(read(root / "specification.json") == specification(), "Changed preregistration")
    require(
        not (root / "plan.json").exists() and not (root / "runs").exists(), "Preserve prior plan"
    )
    api = imports()
    protocol = api["development_verify"](DEVELOPMENT)
    profile = verify_calibration(calibration)
    base = api["scenario_for"](DEVELOPMENT, protocol, FLIGHTS[0], MISSION, 15)
    signatures = {s.config_id: api["model_signature"](s) for s in base.executors}
    arm = calibrated_arm(profile, signatures)
    require(arm["tracking_calibration_used"], "Verified tracking calibration required")
    actions = []
    for spec in base.executors:
        require(spec.config_id in CONFIGS, "Unexpected neural pool")
        resource = base.resources.for_config(spec.config_id)
        latency = arm["latency_s"][spec.config_id]
        actions.extend(
            [
                ActionDescriptor(
                    spec.config_id,
                    spec.config_id,
                    "local",
                    signatures[spec.config_id],
                    latency,
                    latency * 15,
                    resource.load_latency_s,
                    resource.load_energy_j,
                    resource.memory_mb,
                ),
                ActionDescriptor(
                    "REMOTE__" + spec.config_id,
                    spec.config_id,
                    "remote",
                    signatures[spec.config_id],
                    latency * 0.25,
                    0,
                    0,
                    0,
                    64.0,
                ),
            ]
        )
    frozen, copied = {}, {}

    def pin(path):
        frozen[str(Path(path).resolve())] = sha(path)

    def copy_file(path, relative):
        pin(path)
        dest = root / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        require(sha(dest) == sha(path), "Copied input differs")
        copied[str(relative)] = sha(dest)

    for name in ("plan.json", "profile.json", "completion.json"):
        pin(calibration / name)
    for name in ("specification.json", "specification_chronology.json"):
        pin(root / name)
    pin(DEVELOPMENT / "protocol.json")
    # Known controls justify the broad fixed-local reference set, never tuning link outcomes.
    for name in ("summary.json", "completion.json"):
        pin(REPO / "results/mission_measured_cost_study_v2_20261027" / name)
    source = Path(protocol["cohort"])
    pin(source / "protocol.json")
    profile_sources = dict(
        recall=REPO
        / "results/mission_policy_joint_training_v1_20261018/all_development/recall.json",
        risk=REPO / "results/mission_routing_v4_risk_training_20261017/risk_profile.json",
    )
    for key, path in profile_sources.items():
        copy_file(path, Path("profiles") / (key + ".json"))
    cases = {}
    for flight in FLIGHTS:
        source_case = protocol["cases"][flight]
        require(source_case["role"] == "development", "Evaluation recording forbidden")
        manifest = api["checked_manifest"](
            source / flight / "predictions", source / "protocol.json"
        )
        for name in (
            "recording/recording.json",
            "predictions/manifest.json",
            "predictions/completion.json",
            "predictions/inference.json",
        ):
            pin(source / flight / name)
        original = {}
        for name in ("scenario", "intent", "contract"):
            path = DEVELOPMENT / flight / MISSION / (name + ".json")
            original[name] = read(path)
            copy_file(path, Path("originals") / flight / (name + ".json"))
        for link in LINKS:
            cell = cell_name(flight, link)
            cases[cell] = dict(
                source_flight=flight,
                link=link,
                role="development",
                environment=source_case["environment"],
                lineage=source_case["lineage"],
                layout_group=source_case["layout_group"],
            )
            values = derivative_documents(
                original["scenario"], original["intent"], original["contract"], arm["latency_s"]
            )
            for name, value in zip(("scenario", "intent", "contract"), values, strict=True):
                dest = root / "missions" / cell / MISSION / (name + ".json")
                save(dest, value)
                copied[str(dest.relative_to(root))] = sha(dest)
            draft = dict(
                cases=cases,
                assets=protocol["assets"],
                traces=authored_traces(),
                actions=[asdict(a) for a in actions],
                model_signatures=signatures,
            )
            loaded = scenario_for(root, draft, cell)
            require(loaded.recording_identity == manifest["recording_identity"], "Changed flight")
            require(
                {c: manifest["models"][c]["signature"] for c in CONFIGS} == signatures,
                "Changed neural prediction identity",
            )
    _, terrain_metadata = api["load_public_terrain"](Path(protocol["assets"]))
    pin(terrain_metadata["raster_path"])
    pin(Path(protocol["assets"]) / "harmonic_terrain/model.sdf")
    for path in RETAINED.rglob("*"):
        if path.is_file():
            pin(path)
    implementations = implementation_inventory()
    for path in [
        *Path(original_study.__file__).parent.glob("*.py"),
        Path(__file__),
        REPO / "tests/test_mission_network_prepare_v2.py",
        Path(mission_network_audit.__file__),
        REPO / "tests/test_mission_network_study.py",
        REPO / "tests/test_mission_network_audit.py",
        REPO / "tests/test_mission_network_policies.py",
        REPO / "tests/test_mission_network_emulation.py",
    ]:
        implementations[str(path.resolve())] = sha(path)
    snapshots = {}
    for path, expected in implementations.items():
        relative = Path("source_snapshot") / Path(path).relative_to(REPO)
        dest = root / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        require(sha(dest) == expected, "Source snapshot differs")
        copied[str(relative)] = expected
        snapshots[path] = str(relative)
    plan = specification()
    plan.update(
        schema_version="mission_network_study_v1",
        cohort=str(source),
        assets=protocol["assets"],
        cases=cases,
        actions=[asdict(a) for a in actions],
        tracking_cost=arm["tracking_cost"],
        local_latency_s=arm["latency_s"],
        model_signatures=signatures,
        calibration=str(calibration.resolve()),
        calibration_sha256=sha(calibration / "profile.json"),
        profile_paths={k: "profiles/" + k + ".json" for k in profile_sources},
        input_sha256=frozen,
        copied_input_sha256=copied,
        implementation_sha256=implementations,
        source_snapshot_paths=snapshots,
        runtime_versions={
            name: runtime_inventory(name, module)
            for name, module in (("numpy", "numpy"), ("Pillow", "PIL"), ("scipy", "scipy"))
        },
        preflight_revision=dict(
            retained_attempt=str(RETAINED),
            rejection_sha256=sha(RETAINED / "PREFLIGHT_REJECTED.json"),
            original_specification_sha256=sha(RETAINED / "specification.json"),
            change="Optional inventory records actual module metadata or absence; "
            "no runtime, policy, cost, contract, link or endpoint change",
            prior_network_executions=0,
        ),
        prepared_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    # Instantiate every policy before capturing any new selection/outcome.
    scenario = scenario_for(root, plan, next(iter(cases)))
    for name in POLICIES:
        policy_for(root, plan, name, scenario)
    save(root / "plan.json", plan)
    verify(root)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("specify", "prepare"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.command == "specify":
        freeze_specification(args.output)
    else:
        prepare(args.output)


if __name__ == "__main__":
    main()
