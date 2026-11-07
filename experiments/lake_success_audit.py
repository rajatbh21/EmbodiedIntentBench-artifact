"""Expose phase evidence versus inherited report success without changing either contract."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha


def audit(output):
    summary = read(output / "summary.json")
    cases = [
        (r["environment"], r["mission_id"], r["policy"], False, Path(r["output"]))
        for r in summary["comparisons"]
    ]
    for path in sorted((output / "hindsight").glob("*/*/verified.json")):
        cases.append(
            (
                path.parent.parent.name,
                path.parent.name,
                "GT_AWARE_OFFLINE_WITNESS",
                True,
                path.parent / "execution",
            )
        )
    rows = []
    for env, mission, policy, hindsight, path in cases:
        result, phases = read(path / "results.json"), read(path / "objectives.json")
        rows.append(
            dict(
                environment=env,
                mission_id=mission,
                policy=policy,
                gt_aware=hindsight,
                phase_evidence_success=phases["mission_success"],
                inherited_runner_success=result["mission_success"],
                joint_contract_success=phases["mission_success"] and result["mission_success"],
                inherited_report_recall=result["report_quality"]["report_recall"],
                required_report_recall=result["mission_requirements"]["min_report_recall"],
                failed_inherited_constraints=[
                    k for k, v in result["constraints"].items() if v is False
                ],
                results_sha256=sha(path / "results.json"),
                objectives_sha256=sha(path / "objectives.json"),
            )
        )
    groups = {}
    for label, gt in (("predefined_policies", False), ("offline_witnesses", True)):
        selected = [r for r in rows if r["gt_aware"] == gt]
        groups[label] = dict(
            runs=len(selected),
            **{
                key: sum(r[key] for r in selected)
                for key in (
                    "phase_evidence_success",
                    "inherited_runner_success",
                    "joint_contract_success",
                )
            },
        )
    save(
        output / "success_audit.json",
        dict(
            schema_version="1.0",
            implementation_sha256=sha(Path(__file__)),
            summary_sha256=sha(output / "summary.json"),
            groups=groups,
            rows=rows,
            scope="Post-hoc semantic audit; original thresholds and scores unchanged",
            joint_rule="Both contracts must pass; this does not replace the frozen primary score",
        ),
    )
    lines = [
        "# Success-definition audit",
        "",
        "The lake phase scorer and inherited runner expose distinct success contracts. "
        "Keep their outcomes separate. This audit changes no score or requirement; the "
        "joint column conservatively requires both and is not a renamed frozen metric.",
        "",
        "| Cohort | Runs | Phase evidence passes | Inherited runner passes | Both pass |",
        "|---|---:|---:|---:|---:|",
    ]
    for group, values in groups.items():
        lines.append(
            f"| {group} | {values['runs']} | {values['phase_evidence_success']} | "
            f"{values['inherited_runner_success']} | {values['joint_contract_success']} |"
        )
    lines += [
        "",
        "The two offline rescue witnesses meet all four phase evidence stages and "
        "resource constraints, but each produces only two correctly tracked reports out "
        "of nine targets. Report recall 0.222 is below the inherited 0.5 requirement. "
        "Their runner mission_success remains false. They establish phase-evidence "
        "feasibility only; no end-to-end reporting or joint-contract success is shown.",
        "",
        "Before headline policy ranking, version and declare the authoritative intent "
        "contract. If reporting is required, include it consistently in optimization and "
        "evaluation and test predicted tracking. If the benchmark measures perception "
        "evidence only, label that scope explicitly and expose reporting as a separate "
        "metric. Do not silently drop the inherited requirement or retrospectively "
        "relabel these witnesses as end-to-end successes.",
        "",
    ]
    (output / "SUCCESS_AUDIT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    audit(parser.parse_args().output)
