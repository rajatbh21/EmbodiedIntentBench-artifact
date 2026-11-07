"""Require every declared condition and independently reconstructed result."""

from pathlib import Path

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.mission_transfer_comparison_v4.protocol import verify
from experiments.mission_transfer_within_recording.admission import verify as verify_admission


def complete_grid(rows, p):
    expected = {(f, m, c) for f in p["cases"] for m in p["missions"] for c in p["policies"]}
    actual = [(r["flight"], r["mission"], r["policy"]) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Missing, extra or repeated evaluation cell")


def paired_sources(audits, protocol):
    """Require exact shared banks for battery mates, not for lighting strata."""
    if set(audits) != set(protocol["cases"]):
        raise ValueError("Incomplete audited condition population")
    sources, layouts = {}, set()
    for flight, case in protocol["cases"].items():
        audit = audits[flight]
        layouts.add(case["layout_group"])
        identity = (
            audit["actual_neural_calls"],
            audit["frame_config_pairs"],
            audit["recording_pose_sha256"],
        )
        source = case["source_flight"]
        if source in sources and sources[source] != identity:
            raise ValueError("Battery intervention changed source")
        sources[source] = identity
    if len(layouts) != protocol["independent_evaluation_layouts"]:
        raise ValueError("Incomplete layout population")
    if len(sources) != protocol["physical_recordings"]:
        raise ValueError("Missing source recording")
    return sources


def run(root):
    p = verify(root)
    verify_admission(Path(p["admission"]))
    rows, hashes, audits = [], {}, {}
    for f in p["cases"]:
        path = root / f"validation_{f}.json"
        a = read(path)
        if (
            not a["complete"]
            or a["condition"] != f
            or a["protocol_sha256"] != sha(root / "protocol.json")
            or a["validator_sha256"] != sha(Path(__file__).with_name("validate.py"))
            or a["routing_admission_sha256"] != p["routing_admission_sha256"]
        ):
            raise ValueError("Changed condition audit")
        for n, expected in a["source_summaries_sha256"].items():
            if sha(root / n) != expected:
                raise ValueError("Changed audited summary")
            rows.extend(read(root / n)["cases"])
        hashes[path.name] = sha(path)
        audits[f] = a
    complete_grid(rows, p)
    sources = paired_sources(audits, p)
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
        fresh_test=False,
        prospective_policy_evaluation=True,
        geometry_revised_after_rejected_acquisition=True,
        methods_frozen_before_capture=True,
        protocol_sha256=sha(root / "protocol.json"),
        policy_cases=len(rows),
        independent_layouts=p["independent_evaluation_layouts"],
        physical_recordings=len(sources),
        actual_neural_calls=sum(v[0] for v in sources.values()),
        frame_config_pairs=sum(v[1] for v in sources.values()),
        public_decisions_reproduced=sum(a["public_decisions_reproduced"] for a in audits.values()),
        condition_audits_sha256=hashes,
        within_recording_policy_inputs_identical=True,
        paired_intent_and_battery_inputs_identical=True,
        cross_lighting_poses_identical=False,
        lighting_causal_effect_identified=False,
        admission_amended_after_capture=True,
        routing_admission_sha256=p["routing_admission_sha256"],
        primary_policy=p["primary_policy"],
        baseline="approxdet_eq5_adapter",
        source_sha256=sha(Path(__file__)),
        scope=p["scope"],
    )
    save(root / "validation.json", result)
    print(result, flush=True)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
