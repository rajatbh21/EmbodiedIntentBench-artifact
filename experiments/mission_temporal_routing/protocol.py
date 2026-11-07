"""Freeze policy definitions and all 84 paired executions before evaluation."""

from pathlib import Path

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.mission_routing_v2.flights import split_check
from experiments.mission_strategy_routing.protocol import verify as verify_parent
from experiments.mission_temporal_routing.train import verify as verify_training

CONDITIONS = (
    "cohort",
    "training",
    "risk_training",
    "assets",
    "cases",
    "missions",
    "batteries",
    "active_cpu_power_sensitivity_w",
)


def checked_parent(parent):
    p = verify_parent(parent)
    audit = read(parent / "release_audit.json")
    if not audit["complete"] or audit["fresh_test"]:
        raise ValueError("A completed exploratory strategy study is required")
    for name, expected in audit["artifacts_sha256"].items():
        if sha(parent / name) != expected:
            raise ValueError("Changed strategy reference")
    return p


def prepare(parent, training, output):
    from experiments.mission_temporal_routing.study import POLICIES

    if output.exists():
        raise FileExistsError("Preserve prior temporal study")
    old, profile = checked_parent(parent), verify_training(training)
    for case in old["cases"].values():
        split_check(profile["training_lineage"], case["lineage"])
    output.mkdir(parents=True)
    files = {}
    for case in old["cases"]:
        for mission in old["missions"]:
            folder = output / case / mission
            folder.mkdir(parents=True)
            for name in ("scenario.json", "contract.json", "intent.json"):
                path = Path(case) / mission / name
                (output / path).write_bytes((parent / path).read_bytes())
                files[str(path)] = sha(output / path)
    repo = Path(__file__).resolve().parents[2]
    sources = [
        p for tree in ("experiments", "missionbench") for p in (repo / tree).rglob("*.py")
    ]
    sources += [
        repo / "tests/test_mission_temporal_routing.py",
        repo / "docs/mission_temporal_routing.md",
    ]
    save(
        output / "protocol.json",
        dict(
            schema_version="mission_temporal_study_v1",
            parent=str(parent.resolve()),
            temporal_training=str(training.resolve()),
            **{k: old[k] for k in CONDITIONS},
            policies=list(POLICIES),
            frozen_files=files,
            source_sha256={str(path): sha(path) for path in sorted(sources)},
            parent_sha256={
                n: sha(parent / n)
                for n in ("protocol.json", "analysis.json", "validation.json", "release_audit.json")
            },
            temporal_training_sha256=sha(training / "completion.json"),
            scope="exploratory_temporal_policy_and_matched_ablation_on_inspected_v6",
            primary_endpoint="Full contract success over all twelve paired conditions",
            primary_comparisons=[
                ["temporal_mission", name]
                for name in (
                    "survival_reference",
                    "temporal_only",
                    "temporal_no_repetition",
                    "temporal_no_image",
                    "temporal_no_history",
                    "temporal_no_battery",
                )
            ],
            secondary_endpoints=[
                "phase_success",
                "report_recall",
                "false_components",
                "reserve_margin_j",
            ],
            fresh_test=False,
            fitted_on_evaluation=False,
            parameter_search=False,
            new_physical_flights=0,
            new_neural_calls=0,
            independent_evaluation_layouts=1,
            evaluation_recordings=2,
            statistics="Paired descriptive contrasts; no frame-level significance tests",
            reproduction_control=(
                "survival_reference must exactly reproduce twelve mission_survival schedules"
            ),
        ),
    )
    return verify(output)


def verify(output):
    from experiments.mission_temporal_routing.study import POLICIES

    p = read(output / "protocol.json")
    if p["schema_version"] != "mission_temporal_study_v1" or p["policies"] != list(POLICIES):
        raise ValueError("Wrong study or policy grid")
    parent = Path(p["parent"])
    old = checked_parent(parent)
    profile = verify_training(Path(p["temporal_training"]))
    if sha(Path(p["temporal_training"]) / "completion.json") != p["temporal_training_sha256"]:
        raise ValueError("Changed temporal training")
    for case in p["cases"].values():
        split_check(profile["training_lineage"], case["lineage"])
    for key in CONDITIONS:
        if p[key] != old[key]:
            raise ValueError("Changed experimental condition")
    for path, expected in p["source_sha256"].items():
        if sha(Path(path)) != expected:
            raise ValueError("Changed frozen source: " + path)
    for name, expected in p["parent_sha256"].items():
        if sha(parent / name) != expected:
            raise ValueError("Changed reference artifacts")
    for path, expected in p["frozen_files"].items():
        if sha(output / path) != expected or sha(parent / path) != expected:
            raise ValueError("Changed flight or mission contract")
    return p
