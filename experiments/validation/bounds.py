"""Physical-opportunity and model-union feasibility bounds (evaluator-only, GT-aware).

Responsibility: tabulate, per eligible capture of each recording, which focal
targets are (a) visible at all in ground truth, (b) visible with at least the
evaluator's minimum component area, and (c) matched by each candidate model's
stored mask under the unchanged evaluator; then decide, per contract, whether
the contract's evidence and report-recall requirements could be met at all.

Each bound assumes zero latency and that every capture is processed, so it
dominates every legal schedule: failing a bound is a sound infeasibility
certificate for that recording (sensor/detectable) or that model pool
(model_union). Passing a bound never proves feasibility. Never a policy input.
"""

from __future__ import annotations

import numpy as np

from experiments.lake_missions.objectives import enough_evidence


def capture_table(ws, flight):
    """Per-capture GT visibility and per-model matches for one recording."""
    from experiments.lake_hindsight_witness import VerifiedMasks
    from experiments.v4_benchmark.runner import evaluation_frame
    from experiments.wide_area_routing.runner import validate_bank
    from missionbench.v2.evaluation import MissionEvaluator

    plan = ws.plan
    item = next(i for i in plan["grid"] if i["flight"] == flight)
    case = ws.load_case(item)
    scenario = case["scenario"]
    bank, manifest = ws.recording(flight)
    indices = validate_bank(scenario, bank)
    focal = plan["cases"][flight]["evaluator_target_ids"]
    by_name = {name: oid for oid, name in scenario.targets.items()}
    models = sorted({a.source_config_id for a in case["actions"]})
    masks = {
        m: VerifiedMasks(ws.root / "predictions" / flight, manifest["models"][m]) for m in models
    }
    min_px = MissionEvaluator(
        scenario.matching_iou_threshold, len(scenario.targets), scenario.observation_interval_s
    ).min_component_px
    rows = []
    for index in indices:
        frame, prepared = bank["frames"][index], bank["payloads"][index]
        t = (scenario.schedule.at(index) - scenario.schedule.epoch_ns) / 1e9
        truth = evaluation_frame(frame, index, t, scenario)
        counts = np.bincount(frame.object_ids.ravel().astype(np.int64))
        pixels = {
            name: int(counts[by_name[name]]) if by_name[name] < len(counts) else 0 for name in focal
        }
        matched = {}
        for m in models:
            evaluator = MissionEvaluator(
                scenario.matching_iou_threshold,
                len(scenario.targets),
                scenario.observation_interval_s,
            )
            score = evaluator.update(truth, masks[m][prepared.rgb_sha256])
            matched[m] = sorted(set(score.matched_target_ids) & set(focal))
        rows.append(
            dict(
                index=index,
                t=t,
                release_ns=scenario.schedule.at(index) - scenario.schedule.epoch_ns,
                pixels=pixels,
                visible=sorted(n for n in focal if pixels[n] > 0),
                detectable=sorted(n for n in focal if pixels[n] >= min_px),
                matched=matched,
            )
        )
    return dict(flight=flight, focal=focal, min_component_px=min_px, models=models, captures=rows)


def bound_result(table, contract_doc, kind):
    """Evaluate one bound for one contract. ``kind``: sensor | detectable | model_union."""
    focal = table["focal"]
    observed = []
    for row in table["captures"]:
        if kind == "sensor":
            ids = row["visible"]
        elif kind == "detectable":
            ids = row["detectable"]
        elif kind == "model_union":
            ids = sorted({t for m in row["matched"].values() for t in m})
        else:
            raise ValueError("Unknown bound")
        observed.append((row["t"], set(ids)))
    phases = []
    reportable = set()
    for phase in contract_doc["mission"]["phases"]:
        times = {
            t: [
                c
                for c, ids in observed
                if t in ids and phase["start_s"] <= c < phase["end_s"] and c <= phase["deadline_s"]
            ]
            for t in focal
        }
        supported = sorted(t for t, ts in times.items() if ts and enough_evidence(ts, phase))
        reportable |= set(supported)
        phases.append(
            dict(
                phase_id=phase["phase_id"],
                supported=supported,
                passed=len(supported) / len(focal) >= phase["required_recall"],
                observation_counts={t: len(ts) for t, ts in times.items()},
            )
        )
    reports_ok = len(reportable) / len(focal) >= contract_doc["reporting"]["min_report_recall"]
    return dict(
        kind=kind,
        passed=all(p["passed"] for p in phases) and reports_ok,
        phases=phases,
        reportable=sorted(reportable),
    )
