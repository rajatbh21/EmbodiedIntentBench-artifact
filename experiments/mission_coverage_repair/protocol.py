"""Preserve the pre-acquisition design and disclose a normalization-only repair."""

import copy
from pathlib import Path

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.mission_coverage_routing.protocol import profile_paths
from experiments.mission_coverage_routing.protocol import verify as original_verify


def prepare(original, root):
    p = copy.deepcopy(original_verify(original))
    root.mkdir(parents=True, exist_ok=False)
    for name, expected in p["frozen_files"].items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((original / name).read_bytes())
        if sha(path) != expected:
            raise ValueError("Changed intervention")
    failure_paths = sorted(original.glob("runs/*/*/*/*/execution/results.json"))
    failures = []
    for path in failure_paths:
        r = read(path)
        events = read(path.with_name("events.json"))
        if not r["router_errors"]:
            raise ValueError("Only interrupted erroneous original runs expected")
        failures.append(
            dict(
                path=str(path.resolve()),
                results_sha256=sha(path),
                events_sha256=sha(path.with_name("events.json")),
                router_errors=r["router_errors"],
                first_error=next(e["router_error"] for e in events if e["router_error"]),
            )
        )
    if len(failures) != 4:
        raise ValueError("Expected four preserved failed executions")
    repo = Path(__file__).resolve().parents[2]
    sources = [*Path(__file__).parent.glob("*.py"), repo / "tests/test_mission_coverage_repair.py"]
    p.update(
        schema_version="mission_coverage_numerical_repair_v1",
        original_study=str(original.resolve()),
        original_protocol_sha256=sha(original / "protocol.json"),
        repair_source_sha256={str(s.resolve()): sha(s) for s in sources},
        failed_executions=failures,
        pre_acquisition_design=True,
        fresh_test=False,
        strict_unmodified_prospective_test=False,
        scope="fresh_physics_predeclared_design_with_disclosed_numerical_repair",
        repair=(
            "Normalize each count distribution after Bernoulli update; "
            "objectives and coefficients unchanged"
        ),
        outcome_driven_parameter_change=False,
        implementation_repaired_after_failed_replays=True,
    )
    save(root / "protocol.json", p)
    return verify(root)


def verify(root):
    from experiments.mission_coverage_repair.study import POLICIES

    p = read(root / "protocol.json")
    original = Path(p["original_study"])
    old = original_verify(original)
    if p["schema_version"] != "mission_coverage_numerical_repair_v1" or p["policies"] != list(
        POLICIES
    ):
        raise ValueError("Changed repaired grid")
    if sha(original / "protocol.json") != p["original_protocol_sha256"]:
        raise ValueError("Changed original")
    for key in (
        "cases",
        "missions",
        "policies",
        "batteries",
        "profile_sha256",
        "cohort_protocol_sha256",
        "primary_policy",
        "primary_comparisons",
        "frozen_files",
    ):
        if p[key] != old[key]:
            raise ValueError("Changed design beyond numerical repair: " + key)
    for name, expected in p["frozen_files"].items():
        if sha(root / name) != expected:
            raise ValueError("Changed mission")
    for name, expected in p["repair_source_sha256"].items():
        if sha(Path(name)) != expected:
            raise ValueError("Changed repair source")
    for f in p["failed_executions"]:
        if (
            sha(Path(f["path"])) != f["results_sha256"]
            or sha(Path(f["path"]).with_name("events.json")) != f["events_sha256"]
        ):
            raise ValueError("Changed preserved failure")
    # Re-exported for the unchanged profile-loading interface.
    if not profile_paths(p):
        raise ValueError("Missing profiles")
    return p
