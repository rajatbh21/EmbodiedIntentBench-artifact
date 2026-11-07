"""Require the full prospective grid and combine independent condition audits."""

from pathlib import Path

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.mission_coverage_routing.protocol import verify
from experiments.mission_coverage_routing.sensor import verify_gate


def complete_grid(rows, p):
    expected = {(f, m, c) for f in p["cases"] for m in p["missions"] for c in p["policies"]}
    actual = [(r["flight"], r["mission"], r["policy"]) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Missing, extra or repeated evaluation cell")


def run(root):
    p = verify(root)
    cohort = Path(p["cohort"])
    verify_gate(cohort)
    rows = []
    audits = {}
    hashes = {}
    groups = {}
    sources = {}
    for f, c in p["cases"].items():
        path = root / f"validation_{f}.json"
        a = read(path)
        if (
            not a["complete"]
            or a["condition"] != f
            or a["protocol_sha256"] != sha(root / "protocol.json")
            or a["validator_sha256"] != sha(Path(__file__).with_name("validate.py"))
        ):
            raise ValueError("Changed condition audit")
        for n, expected in a["source_summaries_sha256"].items():
            if sha(root / n) != expected:
                raise ValueError("Changed audited summary")
            rows.extend(read(root / n)["cases"])
        hashes[path.name] = sha(path)
        audits[f] = a
        groups.setdefault(c["layout_group"], []).append(a["recording_pose_sha256"])
        key = c["source_flight"]
        triple = (a["actual_neural_calls"], a["frame_config_pairs"], a["recording_pose_sha256"])
        if key in sources and sources[key] != triple:
            raise ValueError("Battery intervention changed source")
        sources[key] = triple
    complete_grid(rows, p)
    if len(groups) != 3 or any(len(v) != 4 or len(set(v)) != 1 for v in groups.values()):
        raise ValueError("Lighting or battery changed paired trajectory")
    for row in rows:
        for name, key in (
            ("outcome.json", "outcome_sha256"),
            ("execution/events.json", "events_sha256"),
            ("execution/results.json", "results_sha256"),
        ):
            if sha(root / row["path"] / name) != row[key]:
                raise ValueError("Changed audited execution")
    result = dict(
        complete=True,
        fresh_test=True,
        protocol_sha256=sha(root / "protocol.json"),
        policy_cases=len(rows),
        independent_layouts=len(groups),
        physical_recordings=len(sources),
        actual_neural_calls=sum(v[0] for v in sources.values()),
        frame_config_pairs=sum(v[1] for v in sources.values()),
        public_decisions_reproduced=sum(a["public_decisions_reproduced"] for a in audits.values()),
        condition_audits_sha256=hashes,
        paired_lighting_and_battery_poses_identical=True,
        primary_policy=p["primary_policy"],
        primary_policy_selected_before_acquisition=True,
        source_sha256=sha(Path(__file__)),
    )
    if (
        result["policy_cases"] != 504
        or result["actual_neural_calls"] != 6120
        or result["frame_config_pairs"] != 3240
    ):
        raise ValueError("Wrong experiment size")
    save(root / "validation.json", result)
    print(result, flush=True)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
