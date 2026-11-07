"""Execution-constrained hindsight witness search (evaluator-only; never a policy).

Responsibility: for one condition (recording x urgency schedule x link x
battery) and one contract, search per-capture action scripts and execute each
candidate through the unchanged causal runtime with that contract as public
input. A successful script is a *demonstrated feasible schedule* under every
declared cost, deadline and resource rule.

Method: first-improvement local search seeded with every released policy's
executed schedule and every constant schedule; one action changed per move;
lexicographic objective (success, passed checks, phase-supported targets,
timely focal reports, fewer false/duplicate reports, battery). It uses hidden
ground truth through the objective and future information through the script,
so it is an analysis bound only. It is not optimal: failure is inconclusive.
"""

from __future__ import annotations

ABSTAIN = None


def objective(summary, row):
    checks = summary["window_checks"]
    passed = sum(bool(v) for v in checks.values()) + (0 if summary["failed"] else 1)
    return (
        int(summary["success"]),
        passed,
        sum(len(s) for s in summary["supported_targets"]),
        summary["focal_correct_reports"],
        -(summary["world_false_reports"] + summary["world_duplicate_reports"]),
        round(row["final_battery_frac"], 9),
    )


def script_from_events(events, release_ns_by_index):
    """Hold each processed capture's action over the captures its execution skipped."""
    script, last = {}, ABSTAIN
    processed = {e["i"]: e["action"] for e in events}
    for index in sorted(release_ns_by_index):
        if index in processed:
            last = processed[index]
        script[release_ns_by_index[index]] = last
    return script


def search(job):
    """Runs inside a worker initialised by ``execution._init``."""
    from experiments.validation import execution

    carrier, contract = job["item"], job["contract"]
    actions = list(job["actions"]) + ([ABSTAIN] if job["include_abstain"] else [])
    order = sorted(job["release_ns_by_index"].items())
    budget, evaluations, history = job["max_evaluations"], 0, []
    cache = {}

    def evaluate(script):
        nonlocal evaluations
        key = tuple(script[r] for _, r in order)
        if key in cache:
            return cache[key]
        evaluations += 1
        out = execution.run_job(
            dict(
                job_id=job["job_id"],
                item=carrier,
                contract=contract,
                script={str(k): v for k, v in script.items()},
                label="hindsight_witness",
                keep_events=True,
            )
        )
        value = (objective(out["summary"], out["row"]), out)
        cache[key] = value
        return value

    best = None
    for start in job["starts"]:
        script = {int(k): v for k, v in start["script"].items()}
        score, out = evaluate(script)
        if best is None or score > best[0]:
            best = (score, script, out, start["source"])
    sweeps = 0
    while not best[0][0] and evaluations < budget and sweeps < job["max_sweeps"]:
        sweeps += 1
        improved = False
        for _, release in order:
            for action in actions:
                if action == best[1][release] or evaluations >= budget:
                    continue
                candidate = dict(best[1])
                candidate[release] = action
                score, out = evaluate(candidate)
                if score > best[0]:
                    best = (score, candidate, out, "local_search")
                    improved = True
                    history.append(dict(evaluations=evaluations, objective=list(score)))
                    if score[0]:
                        break
            if best[0][0] or evaluations >= budget:
                break
        if not improved:
            break
    score, script, out, source = best
    return dict(
        job_id=job["job_id"],
        condition=job["condition"],
        contract=contract,
        success=bool(score[0]),
        objective=list(score),
        source=source,
        evaluations=evaluations,
        sweeps=sweeps,
        exhausted_budget=evaluations >= budget,
        script={str(k): v for k, v in sorted(script.items())},
        summary=out["summary"],
        row=out["row"],
        events=out["events"],
        reports=out["reports"],
        improvement_history=history,
    )


def _search_chunk(jobs):
    return [search(j) for j in jobs]
