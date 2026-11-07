"""Leave-one-declared-scene-family-out mission diagnostics using frozen predictions.

All scenarios are development diagnostics. Repeated contracts/seeds are not new
independent scenes; this tool does not publish statistical superiority claims.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from missionbench.schemas.loading import DocumentReader, read_json_object
from experiments.v4_benchmark.backends import ReplayBackend, save_json
from experiments.v4_benchmark.calibrated_router import fit, merge_profiles
from experiments.v4_benchmark.probe_cache import build_cached, checksum
from experiments.v4_benchmark.routers import build_router
from experiments.v4_benchmark.runner import run_mission
from experiments.v4_benchmark.scenario import exported_source, load_scenario


def mission_diagnostics(result, events):
    """Evaluator-only timing and joint-quality diagnostics; never router inputs."""
    completion = [
        event["completion_time_s"] for event in events if event["score"]["matched_components"]
    ]
    decisions = [event["router_decision"] for event in events if event.get("router_decision")]
    supported = [d for d in decisions if "context_supported" in d]
    return dict(
        first_correct_report_s=min(completion) if completion else None,
        joint_quality_diagnostic=(
            result["quality"]["target_recall"] >= 0.5
            and result["quality"]["detection_precision"] >= 0.95
            and result["quality"]["total_predictions"] > 0
        ),
        unsupported_context_fraction=(
            sum(not d["context_supported"] for d in supported) / len(supported)
            if supported
            else None
        ),
    )


def static_comparisons(rows):
    pairs = []
    for row in rows:
        if row["router"].startswith("static:"):
            continue
        successful = [
            r
            for r in rows
            if r["case_id"] == row["case_id"]
            and r["metric"] == row["metric"]
            and r["router"].startswith("static:")
            and r["mission_success"]
        ]
        best = min(successful, key=lambda r: r["energy_j"]) if successful else None
        pairs.append(
            dict(
                case_id=row["case_id"],
                metric=row["metric"],
                router=row["router"],
                hindsight_reference=best["router"] if best else None,
                missed_mission_despite_successful_static=bool(best and not row["mission_success"]),
                energy_delta_j=row["energy_j"] - best["energy_j"]
                if best and row["mission_success"]
                else None,
            )
        )
    return pairs


def run_study(source: Path, output: Path):
    reader = DocumentReader(
        read_json_object(source),
        context=str(source),
        allowed_fields=("study_schema_version", "cost_profile", "entries"),
    )
    if reader.get_str("study_schema_version") != "0.1":
        raise ValueError("unsupported study schema")
    entries = reader.get_object_list(
        "entries",
        allowed_fields=("case_id", "family_id", "scenario", "recording", "probe", "cache"),
    )
    cases = []
    for item in entries:
        case = dict(case_id=item.get_str("case_id"), family_id=item.get_str("family_id"))
        for key in ("scenario", "recording", "probe", "cache"):
            case[key] = (source.parent / item.get_str(key)).resolve()
        case["loaded"] = load_scenario(case["scenario"])
        cases.append(case)
    if len({c["case_id"] for c in cases}) != len(cases) or len({c["family_id"] for c in cases}) < 2:
        raise ValueError("need unique case IDs and at least two declared families")
    worlds = {}
    for case in cases:
        from experiments.v4_benchmark.scenario import digest

        key = digest(case["loaded"].world)
        if key in worlds and worlds[key] != case["family_id"]:
            raise ValueError("identical world assigned to different families")
        worlds[key] = case["family_id"]
    cost_path = (source.parent / reader.get_str("cost_profile")).resolve()
    costs = read_json_object(cost_path)
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "study.json", read_json_object(source))
    frozen = []
    for index, case in enumerate(cases):
        case["profile"] = output / f"profile_{index:03d}.json"
        fit(case["scenario"], case["recording"], case["probe"], case["profile"], use_altitude=True)
        # Verify masks and signatures before scoring any router.
        build_cached(case["loaded"].executors, case["cache"], case["loaded"].recording_identity)
        frozen.append(
            dict(
                case_id=case["case_id"],
                family_id=case["family_id"],
                scenario_sha256=checksum(case["scenario"]),
                cache_sha256=checksum(case["cache"] / "manifest.json"),
                profile_sha256=checksum(case["profile"]),
            )
        )
    save_json(
        output / "inputs.json",
        dict(schema_version="0.1", cost_profile_sha256=checksum(cost_path), entries=frozen),
    )
    rows = []
    for index, case in enumerate(cases):
        training = [c["profile"] for c in cases if c["family_id"] != case["family_id"]]
        artifact = output / f"fold_{index:03d}.json"
        merge_profiles(training, artifact)
        for metric, threshold in [("target_recall", 0.5), ("detection_precision", 0.95)]:
            folder = output / f"case_{index:03d}_{metric}"
            folder.mkdir()
            data = exported_source(case["loaded"])
            for spec in data["executor_configs"]:
                from experiments.v4_benchmark.calibrated_router import model_signature

                original = next(
                    s for s in case["loaded"].executors if s.config_id == spec["config_id"]
                )
                cost = costs["models"][spec["config_id"]]
                if cost["signature"] != model_signature(original):
                    raise ValueError("cost/model signature mismatch")
                latency = cost["measured_latency_p95_s"]
                if (
                    not isinstance(latency, (int, float))
                    or not math.isfinite(latency)
                    or latency <= 0
                ):
                    raise ValueError("invalid cost latency")
                spec["mission_latency_s"] = math.ceil(latency / 0.01) * 0.01
            data["public_profiles"] = [
                dict(
                    config_id=s["config_id"],
                    expected_latency_ms=s["mission_latency_s"] * 1000,
                    expected_upload_mb=0,
                    quality_tier=s["quality_tier"],
                )
                for s in data["executor_configs"]
            ]
            data["mission"]["router_latency_s"] = 0.01
            data["mission"]["router_energy_j"] = 0.05
            contract = json.loads(
                (case["scenario"].parent / case["loaded"].source["contract_path"]).read_text()
            )
            contract.update(quality_metric=metric, quality_threshold=threshold)
            save_json(folder / "contract.json", contract)
            save_json(folder / "scenario.json", data)
            scenario = load_scenario(folder / "scenario.json")
            ids = [s.config_id for s in scenario.executors]
            policies = [(f"static:{key}", f"static:{key}", {}) for key in ids]
            policies += [
                ("intent_visual", "intent_visual", {}),
                ("calibrated_context", "calibrated", dict(artifact_path=str(artifact.resolve()))),
                (
                    "calibrated_global",
                    "calibrated",
                    dict(artifact_path=str(artifact.resolve()), use_context=False),
                ),
                ("periodic", "periodic", dict(config_ids=ids, period_s=2)),
            ]
            policies += [
                (f"random_{seed}", "seeded_random", dict(config_ids=ids, seed=seed))
                for seed in range(3)
            ]
            for ordinal, (label, name, params) in enumerate(policies):
                backend = ReplayBackend(scenario, case["recording"])
                try:
                    result = run_mission(
                        scenario,
                        backend,
                        build_router(
                            name, scenario.profiles, scenario.observation_interval_s, params
                        ),
                        build_cached(
                            scenario.executors, case["cache"], scenario.recording_identity
                        ),
                        router_name=label,
                        output=folder / f"router_{ordinal:02d}",
                    )
                finally:
                    backend.close()
                events = json.loads((folder / f"router_{ordinal:02d}" / "events.json").read_text())
                decisions = [e.get("router_decision") for e in events]
                rows.append(
                    dict(
                        **mission_diagnostics(result, events),
                        case_id=case["case_id"],
                        family_id=case["family_id"],
                        metric=metric,
                        router=label,
                        mission_success=result["mission_success"],
                        status=result["status"],
                        error=result["error"],
                        recall=result["quality"]["target_recall"],
                        precision=result["quality"]["detection_precision"],
                        energy_j=result["cumulative_energy_j"],
                        processed=result["processed_observations"],
                        skipped=len(result["skipped_observation_ids"]),
                        switches=result["config_switches"],
                        choices=dict(Counter(result["config_selection_history"])),
                        decision_count=sum(d is not None for d in decisions),
                    )
                )
                save_json(output / "rows.json", rows)
    # Freeze an all-development profile for FUTURE scenes, never score it on these sources.
    merge_profiles([c["profile"] for c in cases], output / "all_development_profile.json")
    report = dict(
        study_schema_version="0.1",
        scope="cross-family development diagnostics, NOT held-out benchmark performance",
        execution_mode="precomputed_real_model_predictions",
        scene_count=len(cases),
        family_count=len({c["family_id"] for c in cases}),
        independence_verified=False,
        random_seeds_are_not_independent_scenes=True,
        rows=rows,
        static_comparisons=static_comparisons(rows),
        reference_scope="Hindsight best successful static; analysis only, not a deployable policy",
    )
    save_json(output / "report.json", report)
    lines = [
        "# Cross-family development diagnostics",
        "",
        "Frozen predictions; configured mission costs. No population generalization claim.",
        "",
        "| Scene | Contract | Router | Success | Recall | Precision | Calls | Energy J |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['case_id']} | {r['metric']} | {r['router']} | {r['mission_success']} "
            f"| {r['recall']:.3f} | {r['precision']:.3f} | {r['processed']} | {r['energy_j']:.2f} |"
        )
    (output / "comparison.md").write_text("\n".join(lines) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_study(args.study, args.output)
    print(f"Completed {len(report['rows'])} development mission comparisons")
    return 2 if any(row["status"] == "error" for row in report["rows"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
