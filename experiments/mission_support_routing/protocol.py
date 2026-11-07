"""Freeze a two-factor support-policy study on an explicitly inspected cohort."""

from pathlib import Path

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.resource_routing_study.protocol import verify as verify_parent


def checked_parent(parent):
    protocol = verify_parent(parent)
    audit = read(parent / "release_audit.json")
    if not audit["complete"] or audit["fresh_test"]:
        raise ValueError("A complete exploratory resource study is required")
    for name, expected in audit["artifacts_sha256"].items():
        if sha(parent / name) != expected:
            raise ValueError("Changed parent artifact: " + name)
    return protocol


def prepare(parent, output):
    from experiments.mission_support_routing.evidence import PRIOR_MASS
    from experiments.mission_support_routing.study import POLICIES

    if output.exists():
        raise FileExistsError("Preserve prior support study")
    old = checked_parent(parent)
    output.mkdir(parents=True)
    frozen = {}
    for case in old["cases"]:
        for mission in old["missions"]:
            folder = output / case / mission
            folder.mkdir(parents=True)
            for name in ("scenario.json", "contract.json", "intent.json"):
                path = Path(case) / mission / name
                (output / path).write_bytes((parent / path).read_bytes())
                frozen[str(path)] = sha(output / path)
    repo = Path(__file__).resolve().parents[2]
    sources = {
        str(path): sha(path)
        for tree in ("experiments", "missionbench")
        for path in sorted((repo / tree).rglob("*.py"))
    }
    save(
        output / "protocol.json",
        dict(
            schema_version="mission_support_study_v1",
            parent=str(parent.resolve()),
            scope="exploratory_support_policy_development_on_inspected_v6",
            **{
                k: old[k]
                for k in (
                    "cohort",
                    "training",
                    "risk_training",
                    "assets",
                    "cases",
                    "missions",
                    "batteries",
                    "active_cpu_power_sensitivity_w",
                )
            },
            policies=list(POLICIES),
            primary_comparison="provenance_weakest versus baseline across all twelve cells",
            factors=["prediction model provenance", "weakest versus mean pending support"],
            prior_mass=PRIOR_MASS,
            fitted_profiles_changed=False,
            parameter_search=False,
            fresh_test=False,
            new_physical_flights=0,
            new_neural_calls=0,
            source_sha256=sources,
            frozen_files=frozen,
            parent_sha256={
                name: sha(parent / name)
                for name in (
                    "protocol.json",
                    "analysis.json",
                    "validation.json",
                    "release_audit.json",
                )
            },
            reference_fixed_controls=(
                "Reuse all nine validated parent controls; no new control executions"
            ),
            feasibility=(
                "Reuse unchanged parent contracts and ten paid evaluator witnesses; two unresolved"
            ),
            caveats=[
                "Prediction reassociation is not correctness or calibrated recall",
                "No evidence for never-detected targets or identity across expired tracks",
                "Phase-local trials; old reports never substitute for revisit evidence",
                "Configured overhead, latency and energy; no measured deployment claim",
                "All outcomes retained; no policy revision after this protocol is frozen",
            ],
        ),
    )
    return verify(output)


def verify(output):
    from experiments.mission_support_routing.evidence import PRIOR_MASS
    from experiments.mission_support_routing.study import POLICIES

    p = read(output / "protocol.json")
    if p["schema_version"] != "mission_support_study_v1" or p["policies"] != list(POLICIES):
        raise ValueError("Wrong support study or policy grid")
    old = checked_parent(Path(p["parent"]))
    for key in ("cases", "missions", "batteries", "active_cpu_power_sensitivity_w"):
        if p[key] != old[key]:
            raise ValueError("Changed support-study conditions")
    if p["prior_mass"] != PRIOR_MASS:
        raise ValueError("Changed support prior")
    for name, expected in p["parent_sha256"].items():
        if sha(Path(p["parent"]) / name) != expected:
            raise ValueError("Changed parent study")
    for path, expected in p["source_sha256"].items():
        if sha(Path(path)) != expected:
            raise ValueError("Changed frozen source: " + path)
    for path, expected in p["frozen_files"].items():
        if sha(output / path) != expected or sha(Path(p["parent"]) / path) != expected:
            raise ValueError("Support intervention changed a contract or scenario")
    return p
