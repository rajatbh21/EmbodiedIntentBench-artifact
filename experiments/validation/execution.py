"""Parallel execution of validation jobs over the instrumented runtime.

Responsibility: run declared jobs (one grid cell + protocol + latency multiplier
+ contract variant + optional scripted policy/observer) in worker processes and
return compact, JSON-serialisable summaries. Optionally rescore each in-memory
history under every declared contract variant and relaxation (evaluation only).

Boundaries: no policy or scoring logic lives here; it dispatches to
``runtime``, ``contracts`` and ``leakage``.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_WS = None


def _init(root):
    global _WS
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from experiments.validation import runtime

    runtime.install_memoization()
    _WS = runtime.Workspace(root)


class ScriptedPolicy:
    """Hindsight schedule: a fixed action per capture release time (GT-aware, offline only)."""

    def __init__(self, script):
        self.script = {int(k): v for k, v in script.items()}

    def select(self, **inputs):
        return dict(
            policy="hindsight_script",
            action_id=self.script.get(inputs["release_ns"]),
            reason="offline hindsight schedule; not a deployable policy",
        )


def summarize(outcome):
    window = outcome["windows"][0]
    return dict(
        success=outcome["dynamic_mission_success"],
        failed=outcome["failed_constraints"],
        window_checks=window["checks"],
        focal_correct_reports=window["focal_correct_reports"],
        supported_targets=[sorted(p["supported_targets"]) for p in outcome["phases"]],
        evidence_recall=[p["evidence_recall"] for p in outcome["phases"]],
        false_components=[p["false_components"] for p in outcome["phases"]],
        world_false_reports=window["world_false_reports"],
        world_duplicate_reports=window["world_duplicate_reports"],
        first_valid_report_s=window["first_valid_report_s"],
    )


def compact_events(events):
    return [
        dict(
            i=e["observation_id"],
            t=e["capture_time_s"],
            done=e["completion_time_s"],
            action=e["selected_action_id"],
            admitted=e["perception"]["admitted"],
            status=e["perception"]["status"],
            late=e["perception"]["deadline_missed"],
            matched=e["score"]["matched_target_ids"],
            fp=e["score"]["false_positive_components"],
            visible=e["score"].get("visible_target_ids"),
            reports=len(e["emitted_reports"]),
            skipped=e["skipped_observation_ids"],
            energy=e["cumulative_nonflight_energy_j"],
        )
        for e in events
    ]


def compact_reports(result):
    return [
        dict(
            target=r["target_id"],
            done=r["completion_time_s"],
            evidence=sorted(e["capture_s"] for e in r["evidence"]),
        )
        for r in result["report_quality"]["reports"]
    ]


def _contract(item, name):
    from experiments.validation.contracts import contract_for

    case = _WS.plan["cases"][item["flight"]]
    return contract_for(case, item["battery"], name)


def run_job(job):
    from experiments.intent_mission_v2.runtime import outcome as frozen_outcome
    from experiments.validation import contracts, runtime

    item = job["item"]
    name = job.get("contract", contracts.REFERENCE)
    contract = None if name == contracts.REFERENCE else _contract(item, name)
    outcome_fn = None if name == contracts.REFERENCE else contracts.outcome
    policy = ScriptedPolicy(job["script"]) if job.get("script") is not None else None
    observer = None
    if job.get("observe"):
        from experiments.validation.leakage import LeakageObserver

        bank, _ = _WS.recording(item["flight"])
        observer = LeakageObserver(
            _WS.plan["cases"][item["flight"]]["evaluator_target_ids"],
            bank["frames"],
            item["intent_schedule"],
        )
    res = runtime.run_case(
        _WS,
        item,
        protocol=job.get("protocol", "causal"),
        latency_multiplier=job.get("multiplier", 1),
        contract=contract,
        policy=policy,
        outcome_fn=outcome_fn,
        observer=observer,
        policy_label=job.get("label"),
    )
    events, result, outcome = res["events"], res["result"], res["outcome"]
    out = dict(
        job_id=job["job_id"],
        row=res["row"],
        summary=summarize(outcome),
        pipeline_failures=res["pipeline_failures"],
    )
    if job.get("keep_events", True):
        out["events"] = compact_events(events)
        out["reports"] = compact_reports(result)
    if observer is not None:
        out["leakage"] = observer.summary()
    if job.get("outcome_parity"):
        ours = contracts.outcome(_contract(item, contracts.REFERENCE), events, result)
        frozen = frozen_outcome(_contract(item, contracts.REFERENCE), events, result)
        keys = ("dynamic_mission_success", "checks", "failed_constraints", "window_successes")
        diff = [k for k in keys if ours[k] != frozen[k]]
        diff += [
            k
            for k in (
                "checks",
                "focal_correct_reports",
                "evidence_recall",
                "world_false_reports",
                "world_duplicate_reports",
                "first_valid_report_s",
                "deadline_misses",
            )
            if ours["windows"][0][k] != frozen["windows"][0][k]
        ]
        out["outcome_parity_differences"] = diff
    if job.get("sweep"):
        base = _contract(item, contracts.REFERENCE)
        doc = base.document()
        out["sweep"] = "".join(
            "1"
            if contracts.outcome(base, events, result, document=contracts.sweep_document(doc, s))[
                "dynamic_mission_success"
            ]
            else "0"
            for s in contracts.sweep_settings()
        )
    if job.get("rescore"):
        out["rescore"] = {
            v: summarize(contracts.outcome(_contract(item, v), events, result))
            for v in contracts.VARIANTS
        }
        ref = _contract(item, contracts.REFERENCE)
        out["relaxed"] = {
            r: summarize(contracts.outcome(ref, events, result, relax=[r]))
            for r in contracts.RELAXATIONS
        }
    return out


def _run_chunk(jobs):
    return [run_job(j) for j in jobs]


def execute(root, jobs, workers=8, chunk=12, progress=None):
    """Run jobs grouped by recording so each worker loads one bank per chunk."""
    by_flight = defaultdict(list)
    for j in jobs:
        by_flight[j["item"]["flight"]].append(j)
    chunks = [
        fl[i : i + chunk] for _, fl in sorted(by_flight.items()) for i in range(0, len(fl), chunk)
    ]
    results = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(str(root),)) as pool:
        for done in pool.map(_run_chunk, chunks):
            results.extend(done)
            if progress:
                progress(len(results), len(jobs))
    results.sort(key=lambda r: r["job_id"])
    return results


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, allow_nan=False) + "\n")


def read_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f]
