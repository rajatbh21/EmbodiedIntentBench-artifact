"""Describe adjacent visible-target errors; never feed evaluator identities to a router."""

import argparse
from pathlib import Path

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.mission_coverage_routing.cohort import verify


def transition_counts(rows, config):
    selected = {r["capture_id"]: r for r in rows if r["config_id"] == config}
    if len(selected) != 60:
        raise ValueError("Complete fixed-rate flight required")
    counts = [[0, 0], [0, 0]]
    for i in range(1, 60):
        a, b = selected[i - 1], selected[i]
        visible = set(a["visible_target_ids"]) & set(b["visible_target_ids"])
        for identity in visible:
            previous = int(identity in a["matched_target_ids"])
            current = int(identity in b["matched_target_ids"])
            counts[previous][current] += 1
    return counts


def describe(counts):
    missed, hit = sum(counts[0]), sum(counts[1])
    return dict(
        pairs=missed + hit,
        transitions=counts,
        hit_after_miss=counts[0][1] / missed if missed else None,
        hit_after_hit=counts[1][1] / hit if hit else None,
    )


def run(root):
    p = verify(root)
    diagnostic = read(root / "feasibility_diagnostics.json")
    if diagnostic["protocol_sha256"] != sha(root / "protocol.json"):
        raise ValueError("Wrong source diagnostics")
    flights, totals, sources = [], {}, {}
    for flight in p["cases"]:
        path = root / flight / "diagnostic_outcomes.json"
        expected = diagnostic["flights"][flight]["outcomes_sha256"]
        if sha(path) != expected:
            raise ValueError("Changed raw-scored outcomes")
        rows = read(path)
        sources[flight] = expected
        for config in sorted({r["config_id"] for r in rows}):
            counts = transition_counts(rows, config)
            flights.append(dict(flight=flight, config=config, **describe(counts)))
            total = totals.setdefault(config, [[0, 0], [0, 0]])
            for previous in (0, 1):
                for current in (0, 1):
                    total[previous][current] += counts[previous][current]
    summary = [dict(config=config, **describe(counts)) for config, counts in totals.items()]
    save(
        root / "error_sequences.json",
        dict(
            complete=True,
            posthoc=True,
            evaluator_only=True,
            summary=summary,
            flights=flights,
            source_sha256=sha(Path(__file__)),
            outcomes_sha256=sources,
            protocol_sha256=sha(root / "protocol.json"),
            gt_used_by_deployable_policy=False,
            parameters_changed=False,
            independence_test=False,
            statistical_inference=False,
            caveat="Pooled transitions mix time, target difficulty, identity and visibility. "
            "They are not a calibrated temporal model or proof of conditional dependence.",
        ),
    )
    lines = [
        "# Adjacent-observation diagnostic",
        "",
        "These descriptive counts use evaluator identities on consecutive captures "
        "two seconds apart, only when the same target is visible in both. They expose "
        "the persistence of hits and misses in this fixed corpus. Pooled differences "
        "can reflect target difficulty and changing context as well as temporal "
        "dependence; they do not test independence conditional on the policy's "
        "features. No probability is fitted, policy is changed or significance is claimed.",
        "",
        "| Configuration | Visible adjacent pairs | P(hit after miss) | P(hit after hit) |",
        "|---|---:|---:|---:|",
    ]
    for r in summary:
        miss = "NA" if r["hit_after_miss"] is None else f"{r['hit_after_miss']:.3f}"
        hit = "NA" if r["hit_after_hit"] is None else f"{r['hit_after_hit']:.3f}"
        lines.append(f"| {r['config']} | {r['pairs']} | {miss} | {hit} |")
    lines += [
        "",
        "Anonymous coverage planning assumes independent future Bernoulli evidence "
        "within each prototype while extrapolating the current image. This assumption "
        "remains unvalidated; the transition table is a failure-analysis diagnostic, "
        "not a replacement for prospective temporal calibration on development data.",
        "",
    ]
    (root / "ERROR_SEQUENCES.md").write_text("\n".join(lines))
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
