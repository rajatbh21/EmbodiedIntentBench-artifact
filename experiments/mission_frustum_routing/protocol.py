"""Development layout splits and an immutable geometry-feedback experiment."""

from pathlib import Path

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.mission_frustum_routing.geometry import HORIZONTAL_MARGIN_M, VERTICAL_MARGIN_M
from experiments.mission_policy_selection.protocol import verify as verify_parent

MISSIONS = ("rapid_survey", "verified_inspection")


def checked_parent(parent):
    p = verify_parent(parent)
    audit = read(parent / "release_audit.json")
    if not audit["complete"]:
        raise ValueError("Complete development selection required")
    for name, expected in audit["artifacts_sha256"].items():
        if sha(parent / name) != expected:
            raise ValueError("Changed development selection artifact")
    return p


def prepare(parent, output):
    from experiments.mission_frustum_routing.study import POLICIES

    if output.exists():
        raise FileExistsError("Preserve prior frustum study")
    old = checked_parent(parent)
    output.mkdir(parents=True)
    frozen = {}
    for flight in old["cases"]:
        for mission in MISSIONS:
            (output / flight / mission).mkdir(parents=True)
            for name in ("scenario.json", "contract.json", "intent.json"):
                path = Path(flight) / mission / name
                (output / path).write_bytes((parent / path).read_bytes())
                frozen[str(path)] = sha(output / path)
    repo = Path(__file__).resolve().parents[2]
    save(
        output / "protocol.json",
        dict(
            schema_version="mission_frustum_development_v1",
            parent=str(parent.resolve()),
            **{
                k: old[k]
                for k in (
                    "cohort",
                    "training",
                    "original_risk",
                    "assets",
                    "cases",
                    "active_cpu_power_sensitivity_w",
                )
            },
            missions=list(MISSIONS),
            policies=list(POLICIES),
            scope="development_only_leave_complete_layout_out",
            primary_comparison="frustum_full versus provenance_mean across all eight conditions",
            horizontal_margin_m=HORIZONTAL_MARGIN_M,
            vertical_margin_m=VERTICAL_MARGIN_M,
            margins_are_calibrated_bounds=False,
            estimates_occlusion=False,
            gate_trials=(
                "Only pre-action inside-frustum opportunities count as reassociation trials"
            ),
            gate_current="Omit currently outside-frustum pending tracks; retain discovery",
            router_geometry_cost="Included in configured 50 ms router charge for every policy",
            frozen_files=frozen,
            source_sha256={
                str(x): sha(x)
                for tree in ("experiments", "missionbench")
                for x in sorted((repo / tree).rglob("*.py"))
            },
            parent_sha256={
                name: sha(parent / name)
                for name in (
                    "protocol.json",
                    "analysis.json",
                    "validation.json",
                    "release_audit.json",
                )
            },
            fresh_test=False,
            new_physical_flights=0,
            new_neural_calls=0,
            parameter_search=False,
            fitted_profiles_changed=False,
            caveats=[
                "Two previously inspected development layouts on one map, not new scenes",
                "Both lighting mates excluded from each recall/risk training fold",
                "All four strict-inspection cases are sensor-infeasible and remain included",
                "Frustum inclusion cannot distinguish occlusion from a detector miss",
                "Predicted static positions, ideal odometry and public terrain prior",
                "Configured margins and costs, not measured deployment robustness",
            ],
        ),
    )
    return verify(output)


def verify(output):
    from experiments.mission_frustum_routing.study import POLICIES

    p = read(output / "protocol.json")
    if (
        p["schema_version"] != "mission_frustum_development_v1"
        or p["missions"] != list(MISSIONS)
        or p["policies"] != list(POLICIES)
    ):
        raise ValueError("Wrong frustum development grid")
    old = checked_parent(Path(p["parent"]))
    for key in (
        "cases",
        "training",
        "original_risk",
        "cohort",
        "assets",
        "active_cpu_power_sensitivity_w",
    ):
        if p[key] != old[key]:
            raise ValueError("Changed development split or profile")
    for path, expected in p["source_sha256"].items():
        if sha(Path(path)) != expected:
            raise ValueError("Changed frozen source: " + path)
    for path, expected in p["frozen_files"].items():
        if sha(output / path) != expected or sha(Path(p["parent"]) / path) != expected:
            raise ValueError("Changed fixed-observation mission")
    for name, expected in p["parent_sha256"].items():
        if sha(Path(p["parent"]) / name) != expected:
            raise ValueError("Changed parent evidence")
    return p
