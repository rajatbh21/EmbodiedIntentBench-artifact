"""Tables and summary statistics for the four validation experiments.

Responsibility: read the raw stage outputs under ``results/validation`` and
write compact, committed artifacts (CSV + ``summary.json``) under the report
directory. No execution or rescoring happens here; every number is derived
from stage receipts.

Units: a *condition* is recording x urgency schedule x link x battery (72);
a *run* is condition x policy (648 per contract/protocol/multiplier). The
independent physical unit is the *recording* (6; 3 route families x 2
sectors, one map). Uncertainty is therefore reported per recording and as a
recording-cluster bootstrap, never by treating replay conditions as
independent missions.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from experiments.validation import contracts, execution

POLICIES = (
    "fixed_local_yolo",
    "fixed_local_cabinet",
    "fixed_local_mask",
    "fixed_remote_mask",
    "fastest_current",
    "quality_initial",
    "quality_current",
    "risk_initial",
    "risk_current",
)
LABELS = {
    "fixed_local_yolo": "Fixed local YOLO11s (fast)",
    "fixed_local_cabinet": "Fixed local CABiNet (fast)",
    "fixed_local_mask": "Fixed local Mask R-CNN (accurate, slow)",
    "fixed_remote_mask": "Fixed remote Mask R-CNN (accurate)",
    "fastest_current": "Adaptive: fastest feasible",
    "quality_initial": "Adaptive: quality, ignores urgency",
    "quality_current": "Adaptive: quality, urgency-aware",
    "risk_initial": "Adaptive: quality+risk, ignores urgency",
    "risk_current": "Adaptive: quality+risk, urgency-aware",
}
CONTRACTS = ("survey", "confirm", "confirm_strict", "revisit", "time_critical")
SEED = 20261103
BOOT = 10000


# ------------------------------------------------------------------ helpers
def key_of(item):
    return "/".join(
        (item["flight"], item["intent_schedule"], item["link"], item["energy_condition"])
    )


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in fields})


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, (list, tuple, set, frozenset)):
        return ";".join(map(str, sorted(v) if isinstance(v, (set, frozenset)) else v))
    return v


def cluster_ci(successes_by_recording, trials_by_recording):
    """Percentile bootstrap over recordings (fixed seed). Descriptive only."""
    recs = sorted(trials_by_recording)
    if not recs or sum(trials_by_recording.values()) == 0:
        return None, None
    rng = random.Random(SEED)
    stats = []
    for _ in range(BOOT):
        sample = [rng.choice(recs) for _ in recs]
        n = sum(trials_by_recording[r] for r in sample)
        stats.append(sum(successes_by_recording[r] for r in sample) / n if n else 0.0)
    stats.sort()
    return stats[int(0.025 * BOOT)], stats[int(0.975 * BOOT) - 1]


def evidence_times(events, deadline_s=44.0):
    times = defaultdict(list)
    for e in events:
        if e["admitted"] and e["done"] <= deadline_s:
            for t in e["matched"]:
                times[t].append(e["t"])
    return times


def supported_count(summary):
    """Targets supported in every phase (for single-phase contracts: supported targets)."""
    sets = [set(s) for s in summary["supported_targets"]]
    return len(set.intersection(*sets)) if sets else 0


# ------------------------------------------------------------------ loading
def load(out):
    out = Path(out)
    data = dict(
        reference=execution.read_jsonl(out / "reference.jsonl"),
        reruns=execution.read_jsonl(out / "reruns.jsonl"),
        causal=execution.read_jsonl(out / "causal.jsonl"),
        hindsight=execution.read_jsonl(out / "hindsight.jsonl"),
        skipped=json.loads((out / "hindsight_skipped.json").read_text()),
        bounds=json.loads((out / "bounds.json").read_text()),
        parity=json.loads((out / "parity.json").read_text()),
    )
    data["plan_grid"] = [r["row"] for r in data["reference"]]
    return data


def histories(data):
    """(analysis, contract, run_id) -> summary/events. analysis in rescore|rerun."""
    h = {}
    for r in data["reference"]:
        run = r["job_id"]
        for c in CONTRACTS:
            h[("rescore", c, run)] = dict(summary=r["rescore"][c], events=r["events"], row=r["row"])
        h[("rerun", contracts.REFERENCE, run)] = dict(
            summary=r["summary"], events=r["events"], row=r["row"]
        )
    for r in data["reruns"]:
        c, run = r["job_id"].split("|", 1)
        h[("rerun", c, run)] = dict(summary=r["summary"], events=r["events"], row=r["row"])
    return h


def feasibility_tables(data):
    b = {(x["flight"], x["contract"], x["kind"]): x for x in data["bounds"]["bounds"]}
    return b


# ------------------------------------------------------------------ Exp 1
def _recording_successes(H, items, analysis, policy, contract, recording):
    return sum(
        H[(analysis, contract, i["run_id"])]["summary"]["success"]
        for i in items
        if i["policy"] == policy and i["flight"] == recording
    )


def experiment1(data, H, B):
    items = data["plan_grid"]
    recordings = sorted({i["flight"] for i in items})
    out = dict(
        success={}, evidence=[], reversals=[], decision_changes=[], best={}, per_recording=[]
    )
    for analysis in ("rescore", "rerun"):
        table = {}
        for c in CONTRACTS:
            if analysis == "rerun" and not contracts.VARIANTS[c].get("rerun_supported", True):
                continue
            feasible = [
                f
                for f in recordings
                if B[(f, c, "sensor")]["passed"] and B[(f, c, "model_union")]["passed"]
            ]
            for p in POLICIES:
                runs = [i for i in items if i["policy"] == p]
                succ_rec, trial_rec = Counter(), Counter()
                total_s = 0
                conf = rep = 0
                for i in runs:
                    s = H[(analysis, c, i["run_id"])]["summary"]
                    total_s += s["success"]
                    conf += supported_count(s)
                    rep += s["focal_correct_reports"]
                    if i["flight"] in feasible:
                        succ_rec[i["flight"]] += s["success"]
                        trial_rec[i["flight"]] += 1
                n = sum(trial_rec.values())
                k = sum(succ_rec.values())
                lo, hi = cluster_ci(succ_rec, trial_rec)
                table[(p, c)] = dict(
                    analysis=analysis,
                    policy=p,
                    label=LABELS[p],
                    contract=c,
                    successes_feasible=k,
                    conditions_feasible=n,
                    rate_feasible=k / n if n else None,
                    ci_lo=lo,
                    ci_hi=hi,
                    successes_all=total_s,
                    conditions_all=len(runs),
                    feasible_recordings=len(feasible),
                    excluded_recordings=sorted(set(recordings) - set(feasible)),
                    mean_confirmed_targets=conf / len(runs),
                    mean_timely_focal_reports=rep / len(runs),
                    recordings_with_success=sum(1 for f in feasible if succ_rec[f] > 0),
                )
                for f in recordings:
                    out["per_recording"].append(
                        dict(
                            analysis=analysis,
                            contract=c,
                            policy=p,
                            recording=f,
                            feasible=f in feasible,
                            successes=sum(
                                H[(analysis, c, i["run_id"])]["summary"]["success"]
                                for i in runs
                                if i["flight"] == f
                            ),
                            conditions=sum(1 for i in runs if i["flight"] == f),
                        )
                    )
        out["success"][analysis] = list(table.values())
        present = sorted({c for (_, c) in table}, key=CONTRACTS.index)
        for c in present:
            best = max(table[(p, c)]["successes_feasible"] for p in POLICIES)
            out["best"].setdefault(analysis, {})[c] = dict(
                successes=best,
                conditions=table[(POLICIES[0], c)]["conditions_feasible"],
                policies=[p for p in POLICIES if table[(p, c)]["successes_feasible"] == best],
            )
        # Pairwise ranking reversals on feasible conditions.
        for (a, b2), (x, y) in itertools.product(
            itertools.combinations(POLICIES, 2), itertools.combinations(present, 2)
        ):
            dx = table[(a, x)]["successes_feasible"] - table[(b2, x)]["successes_feasible"]
            dy = table[(a, y)]["successes_feasible"] - table[(b2, y)]["successes_feasible"]
            if dx * dy < 0:
                agree = 0
                for f in recordings:
                    fx = _recording_successes(H, items, analysis, a, x, f) - _recording_successes(
                        H, items, analysis, b2, x, f
                    )
                    fy = _recording_successes(H, items, analysis, a, y, f) - _recording_successes(
                        H, items, analysis, b2, y, f
                    )
                    agree += (fx * dx > 0) and (fy * dy > 0)
                out["reversals"].append(
                    dict(
                        analysis=analysis,
                        policy_a=a,
                        policy_b=b2,
                        contract_x=x,
                        contract_y=y,
                        a_minus_b_x=dx,
                        a_minus_b_y=dy,
                        min_abs_margin=min(abs(dx), abs(dy)),
                        recordings_reversing_same_direction=agree,
                    )
                )
        for c in present:
            for p in POLICIES:
                out["evidence"].append(
                    dict(
                        analysis=analysis,
                        contract=c,
                        policy=p,
                        mean_confirmed_targets=table[(p, c)]["mean_confirmed_targets"],
                        mean_timely_focal_reports=table[(p, c)]["mean_timely_focal_reports"],
                    )
                )
    # Decision changes when the contract is a public input (reruns vs reference histories).
    for c in CONTRACTS:
        if c == contracts.REFERENCE or not contracts.VARIANTS[c].get("rerun_supported", True):
            continue
        for p in POLICIES:
            runs = [i for i in items if i["policy"] == p]
            changed = outcome_changed = 0
            for i in runs:
                ref = [
                    (e["i"], e["action"])
                    for e in H[("rerun", contracts.REFERENCE, i["run_id"])]["events"]
                ]
                new = [(e["i"], e["action"]) for e in H[("rerun", c, i["run_id"])]["events"]]
                changed += ref != new
                outcome_changed += (
                    H[("rescore", c, i["run_id"])]["summary"]["success"]
                    != H[("rerun", c, i["run_id"])]["summary"]["success"]
                )
            out["decision_changes"].append(
                dict(
                    contract=c,
                    policy=p,
                    runs=len(runs),
                    runs_with_changed_actions=changed,
                    runs_where_rerun_and_rescore_disagree=outcome_changed,
                )
            )
    return out


# ------------------------------------------------------------------ Exp 2
def experiment2(data, H, B):
    items = data["plan_grid"]
    by_cond = defaultdict(list)
    for i in items:
        by_cond[key_of(i)].append(i)
    witnesses = {(h["contract"], h["condition"]): h for h in data["hindsight"]}
    rows, gap = [], []
    bounds_rows = []
    recordings = sorted({i["flight"] for i in items})
    for f in recordings:
        for c in CONTRACTS:
            r = dict(recording=f, contract=c)
            for kind in ("sensor", "detectable", "model_union"):
                x = B[(f, c, kind)]
                r[kind + "_passed"] = x["passed"]
                r[kind + "_supported_targets"] = len(
                    set.intersection(*[set(p["supported"]) for p in x["phases"]])
                )
            bounds_rows.append(r)
    classes = {}
    for c in CONTRACTS:
        analysis = "rerun" if contracts.VARIANTS[c].get("rerun_supported", True) else "rescore"
        for key, conds in sorted(by_cond.items()):
            f = conds[0]["flight"]
            succ = [
                i["policy"] for i in conds if H[(analysis, c, i["run_id"])]["summary"]["success"]
            ]
            w = witnesses.get((c, key))
            if not B[(f, c, "sensor")]["passed"]:
                cls = "A_physically_infeasible"
            elif not B[(f, c, "model_union")]["passed"]:
                cls = "B_model_pool_infeasible"
            elif succ:
                cls = "C_released_policy" if analysis == "rerun" else "C_rescored_released_history"
            elif w is not None and w["success"]:
                cls = "C_hindsight_witness"
            else:
                cls = "B_unresolved"
            classes[(c, key)] = cls
            rows.append(
                dict(
                    contract=c,
                    condition=key,
                    recording=f,
                    split=conds[0]["split"],
                    feasibility_class=cls,
                    successful_policies=succ,
                    witness_searched=w is not None,
                    witness_found=bool(w and w["success"]),
                    witness_source=w["source"] if w else None,
                    witness_evaluations=w["evaluations"] if w else None,
                    witness_budget_exhausted=w["exhausted_budget"] if w else None,
                )
            )
        for p in POLICIES:
            counts = Counter()
            for key, conds in by_cond.items():
                i = next(x for x in conds if x["policy"] == p)
                counts[
                    (classes[(c, key)], H[(analysis, c, i["run_id"])]["summary"]["success"])
                ] += 1
            in_c = {k: n for k, n in counts.items() if k[0].startswith("C_")}
            demonstrated = sum(in_c.values())
            wins = sum(n for (cls, s), n in in_c.items() if s)
            gap.append(
                dict(
                    contract=c,
                    policy=p,
                    demonstrated_feasible_conditions=demonstrated,
                    successes_in_feasible=wins,
                    policy_attributable_failures=demonstrated - wins,
                    failures_in_certified_infeasible=sum(
                        n
                        for (cls, s), n in counts.items()
                        if cls[0] in "AB" and cls != "B_unresolved" and not s
                    ),
                    successes_in_certified_infeasible=sum(
                        n
                        for (cls, s), n in counts.items()
                        if cls[0] in "AB" and cls != "B_unresolved" and s
                    ),
                    failures_in_unresolved=counts[("B_unresolved", False)],
                    success_rate_vs_best_verified=wins / demonstrated if demonstrated else None,
                )
            )
    summary = Counter((c, cls) for (c, _), cls in classes.items())
    return dict(
        bounds=bounds_rows,
        classes=rows,
        gap=gap,
        class_counts=[
            dict(contract=c, feasibility_class=cls, conditions=n)
            for (c, cls), n in sorted(summary.items())
        ],
    )


# ------------------------------------------------------------------ Exp 3
def experiment3(data, B):
    spec = contracts.CONFIG["causal_ablation"]
    runs = {}
    for r in data["causal"]:
        protocol, mult, run = r["job_id"].split("|", 2)
        runs[(protocol, float(mult), run)] = r
    items = data["plan_grid"]
    mults = [float(m) for m in spec["latency_multipliers"]]
    metrics = []
    for protocol in spec["protocols"]:
        for m in mults:
            for p in POLICIES:
                rs = [runs[(protocol, m, i["run_id"])] for i in items if i["policy"] == p]
                n = len(rs)
                metrics.append(
                    dict(
                        protocol=protocol,
                        multiplier=m,
                        policy=p,
                        runs=n,
                        mean_processed_observations=sum(r["row"]["decision_count"] for r in rs) / n,
                        mean_skipped_observations=sum(
                            r["row"]["skipped_opportunity_count"] for r in rs
                        )
                        / n,
                        mean_valid_confirmations=sum(supported_count(r["summary"]) for r in rs) / n,
                        mean_completed_reports=sum(
                            r["summary"]["focal_correct_reports"] for r in rs
                        )
                        / n,
                        successes=sum(r["summary"]["success"] for r in rs),
                        deadline_misses=sum(
                            r["row"]["task_metrics"]["perception"]["deadline_misses"] for r in rs
                        ),
                        mean_nonflight_energy_j=sum(r["row"]["nonflight_energy_j"] for r in rs) / n,
                        battery_failures=sum(
                            "battery_constraint_success" in r["row"]["failed_constraints"]
                            for r in rs
                        ),
                    )
                )
    # Protocol disagreement and pairwise ranking reversals between protocols.
    disagreement = []
    reversals = []
    for m in mults:
        diff = Counter()
        for i in items:
            a = runs[("causal", m, i["run_id"])]["summary"]["success"]
            b = runs[("no_skip", m, i["run_id"])]["summary"]["success"]
            diff[(a, b)] += 1
        disagreement.append(
            dict(
                multiplier=m,
                both_succeed=diff[(True, True)],
                both_fail=diff[(False, False)],
                only_no_skip_succeeds=diff[(False, True)],
                only_causal_succeeds=diff[(True, False)],
            )
        )
        succ = {
            (pr, p): sum(
                runs[(pr, m, i["run_id"])]["summary"]["success"] for i in items if i["policy"] == p
            )
            for pr in spec["protocols"]
            for p in POLICIES
        }
        for a, b in itertools.combinations(POLICIES, 2):
            dc, dn = (
                succ[("causal", a)] - succ[("causal", b)],
                succ[("no_skip", a)] - succ[("no_skip", b)],
            )
            if dc * dn < 0 or (dc != 0 and dn == 0) or (dc == 0 and dn != 0):
                reversals.append(
                    dict(
                        multiplier=m,
                        policy_a=a,
                        policy_b=b,
                        causal_a_minus_b=dc,
                        no_skip_a_minus_b=dn,
                        strict_reversal=dc * dn < 0,
                    )
                )
    # Nonlinearity across multipliers, per run.
    shapes = Counter()
    flips = []
    for protocol in spec["protocols"]:
        for i in items:
            seq = [runs[(protocol, m, i["run_id"])]["summary"]["success"] for m in mults]
            shapes[(protocol, "".join("S" if s else "F" for s in seq))] += 1
            for m1, m2 in itertools.pairwise(mults):
                r1, r2 = runs[(protocol, m1, i["run_id"])], runs[(protocol, m2, i["run_id"])]
                s1, s2 = r1["summary"]["success"], r2["summary"]["success"]
                if s1 == s2:
                    continue
                lost = []
                if s1 and not s2:
                    t1 = evidence_times(r1["events"])
                    sup1 = set.intersection(*[set(x) for x in r1["summary"]["supported_targets"]])
                    sup2 = set.intersection(*[set(x) for x in r2["summary"]["supported_targets"]])
                    processed2 = {e["t"] for e in r2["events"]}
                    for t in sorted(sup1 - sup2):
                        missing = sorted(set(t1[t]) - processed2)
                        if missing:
                            lost.append(dict(target=t, unavailable_capture_times=missing))
                flips.append(
                    dict(
                        protocol=protocol,
                        run_id=i["run_id"],
                        policy=i["policy"],
                        split=i["split"],
                        from_multiplier=m1,
                        to_multiplier=m2,
                        direction="success_to_failure" if s1 else "failure_to_success",
                        processed_from=r1["row"]["decision_count"],
                        processed_to=r2["row"]["decision_count"],
                        late_from=r1["row"]["task_metrics"]["perception"]["deadline_misses"],
                        late_to=r2["row"]["task_metrics"]["perception"]["deadline_misses"],
                        failed_to=r2["summary"]["failed"]
                        + [k for k, v in r2["summary"]["window_checks"].items() if not v],
                        lost_confirming_observations=lost,
                        confirmation_lost_to_skipping=bool(lost),
                    )
                )
    # Declared timelines.
    tables = {t["flight"]: t for t in data["bounds"]["tables"]}
    timelines = []
    # Predeclared cases, plus one post-hoc illustration chosen by a stated rule.
    disagree = sorted(
        i["run_id"]
        for i in items
        if i["policy"] == "risk_current"
        and not runs[("causal", 1.0, i["run_id"])]["summary"]["success"]
        and runs[("no_skip", 1.0, i["run_id"])]["summary"]["success"]
    )
    selected = [(run, "(predeclared)") for run in spec["timeline_cases"]]
    if disagree:
        selected.append(
            (disagree[0], "(selected after evaluation: first causal-fail/no-skip-success)")
        )
    for run, selection in selected:
        flight = "/".join(run.split("/")[:2])
        table = tables[flight]
        for protocol in spec["protocols"]:
            r = runs[(protocol, 1.0, run)]
            processed = {e["i"]: e for e in r["events"]}
            caps = []
            for c in table["captures"]:
                e = processed.get(c["index"])
                caps.append(
                    dict(
                        index=c["index"],
                        t=c["t"],
                        visible=c["visible"],
                        processed=e is not None,
                        action=e["action"] if e else None,
                        admitted=e["admitted"] if e else False,
                        late=e["late"] if e else False,
                        done=e["done"] if e else None,
                        matched=sorted(set(e["matched"]) & set(table["focal"])) if e else [],
                    )
                )
            timelines.append(
                dict(
                    selection=selection,
                    run_id=run,
                    protocol=protocol,
                    focal=table["focal"],
                    captures=caps,
                    reports=r["reports"],
                    success=r["summary"]["success"],
                    supported=r["summary"]["supported_targets"][0],
                )
            )
    return dict(
        metrics=metrics,
        disagreement=disagreement,
        reversals=reversals,
        shapes=[
            dict(protocol=p, success_by_multiplier=s, runs=n)
            for (p, s), n in sorted(shapes.items())
        ],
        flips=flips,
        timelines=timelines,
    )


# ------------------------------------------------------------------ Exp 4
REQUIREMENT_OF = {
    "repeated_evidence": "confirmation",
    "mission_false_components": "raw_false_components",
    "fresh_report_recall": "reporting_recall",
    "false_report_limit": "reporting_caps",
    "duplicate_report_limit": "reporting_caps",
    "battery_constraint_success": "resources_battery",
    "memory_constraint_success": "resources_memory",
    "communication_constraint_success": "resources_communication",
    "privacy_constraint_success": "privacy",
    "execution": "execution",
    "deadline_success": "execution",
}


def unmet(summary):
    failed = [k for k, v in summary["window_checks"].items() if not v]
    failed += [k for k in summary["failed"] if k != "all_windows"]
    return sorted({REQUIREMENT_OF[k] for k in failed})


def experiment4(data, B):
    tables = {t["flight"]: t for t in data["bounds"]["tables"]}
    per_run = []
    for r in data["reference"]:
        row, s = r["row"], r["summary"]
        focal = tables[row["flight"]]["focal"]
        matched_once = {t for e in r["events"] if e["admitted"] for t in e["matched"] if t in focal}
        req = unmet(s)
        per_run.append(
            dict(
                run_id=row["run_id"],
                policy=row["policy"],
                split=row["split"],
                schedule=row["intent_schedule"],
                link=row["link"],
                energy=row["energy_condition"],
                recording=row["flight"],
                success=s["success"],
                unmet_requirements=req,
                pattern="+".join(req) if req else "none",
                coverage_shortfall=len(matched_once) < 2,
                freshness="not_applicable_single_window_contract",
                perception_deadline_misses=row["task_metrics"]["perception"]["deadline_misses"],
                nonflight_energy_j=row["nonflight_energy_j"],
                final_battery_frac=row["final_battery_frac"],
                confirmed_targets=supported_count(s),
                timely_focal_reports=s["focal_correct_reports"],
                targets_matched_once=len(matched_once),
            )
        )
    patterns = Counter((x["policy"], x["split"], x["pattern"]) for x in per_run if not x["success"])
    pattern_rows = [
        dict(policy=p, split=sp, pattern=pat, failed_runs=n)
        for (p, sp, pat), n in sorted(patterns.items())
    ]
    # Matched urgency-aware vs urgency-blind comparison.
    pairs = (("quality_current", "quality_initial"), ("risk_current", "risk_initial"))
    by = {(x["policy"], x["run_id"].rsplit("/", 1)[0]): x for x in per_run}
    urgency = []
    for aware, blind in pairs:
        for split in ("transfer", "development", "all"):
            for sched in ("all", "no_change", "urgency_increase", "urgency_recovery"):
                keys = sorted(
                    {
                        k
                        for (p, k) in by
                        if p == aware
                        and (split == "all" or by[(p, k)]["split"] == split)
                        and (sched == "all" or by[(p, k)]["schedule"] == sched)
                    }
                )
                if not keys:
                    continue
                A = [by[(aware, k)] for k in keys]
                Z = [by[(blind, k)] for k in keys]
                urgency.append(
                    dict(
                        aware=aware,
                        blind=blind,
                        split=split,
                        schedule=sched,
                        conditions=len(keys),
                        aware_successes=sum(a["success"] for a in A),
                        blind_successes=sum(z["success"] for z in Z),
                        only_aware_succeeds=sum(
                            a["success"] and not z["success"] for a, z in zip(A, Z, strict=True)
                        ),
                        only_blind_succeeds=sum(
                            z["success"] and not a["success"] for a, z in zip(A, Z, strict=True)
                        ),
                        aware_deadline_misses=sum(a["perception_deadline_misses"] for a in A),
                        blind_deadline_misses=sum(z["perception_deadline_misses"] for z in Z),
                        aware_mean_nonflight_energy_j=sum(a["nonflight_energy_j"] for a in A)
                        / len(A),
                        blind_mean_nonflight_energy_j=sum(z["nonflight_energy_j"] for z in Z)
                        / len(Z),
                        aware_unmet=dict(Counter(a["pattern"] for a in A if not a["success"])),
                        blind_unmet=dict(Counter(z["pattern"] for z in Z if not z["success"])),
                        aware_coverage_shortfall=sum(
                            a["coverage_shortfall"] for a in A if not a["success"]
                        ),
                        blind_coverage_shortfall=sum(
                            z["coverage_shortfall"] for z in Z if not z["success"]
                        ),
                    )
                )
    # Per-target evidence funnel.
    funnel = []
    for p in POLICIES:
        stages = Counter()
        for r in data["reference"]:
            if r["row"]["policy"] != p:
                continue
            table = tables[r["row"]["flight"]]
            vis = {c["index"]: set(c["visible"]) for c in table["captures"]}
            processed = {e["i"] for e in r["events"]}
            admitted_match = defaultdict(set)
            for e in r["events"]:
                if e["admitted"]:
                    for t in e["matched"]:
                        admitted_match[t].add(e["t"])
            supported = set(r["summary"]["supported_targets"][0])
            reported = {
                x["target"]
                for x in r["reports"]
                if x["target"] in table["focal"] and x["done"] <= 44.0
            }
            for t in table["focal"]:
                stages["targets"] += 1
                stages["visible_on_recording"] += any(t in v for v in vis.values())
                stages["visible_on_processed_capture"] += any(t in vis[i] for i in processed)
                stages["matched_at_least_once"] += bool(admitted_match[t])
                stages["confirmed_repeated_evidence"] += t in supported
                stages["reported_on_time"] += t in reported
        funnel.append(dict(policy=p, **stages))
    # Counterfactual (evaluation-only) relaxations.
    relax = []
    for p in POLICIES:
        for split in ("transfer", "all"):
            rs = [
                r
                for r in data["reference"]
                if r["row"]["policy"] == p and (split == "all" or r["row"]["split"] == split)
            ]
            base = sum(r["summary"]["success"] for r in rs)
            row = dict(policy=p, split=split, runs=len(rs), observed_successes=base)
            for name in contracts.RELAXATIONS:
                row["relax_" + name] = sum(r["relaxed"][name]["success"] for r in rs)
            relax.append(row)
    return dict(
        per_run=per_run, patterns=pattern_rows, urgency=urgency, funnel=funnel, relaxations=relax
    )


# ------------------------------------------------------------------ per-run CSV
def per_run_rows(data):
    out = []

    def add(experiment, contract, protocol, mult, rec, policy_override=None):
        row, s = rec["row"], rec["summary"]
        out.append(
            dict(
                experiment=experiment,
                contract=contract,
                protocol=protocol,
                latency_multiplier=mult,
                run_id=row["run_id"],
                recording=row["flight"],
                split=row["split"],
                intent_schedule=row["intent_schedule"],
                link=row["link"],
                energy_condition=row["energy_condition"],
                initial_battery=row["battery"],
                policy=policy_override or row["policy"],
                success=s["success"],
                unmet_requirements=unmet(s),
                confirmed_targets=supported_count(s),
                timely_focal_reports=s["focal_correct_reports"],
                processed_observations=row["decision_count"],
                skipped_observations=row["skipped_opportunity_count"],
                perception_deadline_misses=row["task_metrics"]["perception"]["deadline_misses"],
                nonflight_energy_j=row["nonflight_energy_j"],
                final_battery_frac=row["final_battery_frac"],
                communication_mb=row["communication_mb"],
            )
        )

    for r in data["reference"]:
        add("reference", "confirm", "causal", 1, r)
        for c in CONTRACTS:
            if c == "confirm":
                continue
            s, row = r["rescore"][c], r["row"]
            out.append(
                dict(
                    experiment="exp1_rescore",
                    contract=c,
                    protocol="causal",
                    latency_multiplier=1,
                    run_id=row["run_id"],
                    recording=row["flight"],
                    split=row["split"],
                    intent_schedule=row["intent_schedule"],
                    link=row["link"],
                    energy_condition=row["energy_condition"],
                    initial_battery=row["battery"],
                    policy=row["policy"],
                    success=s["success"],
                    unmet_requirements=unmet(s),
                    confirmed_targets=supported_count(s),
                    timely_focal_reports=s["focal_correct_reports"],
                )
            )
    for r in data["reruns"]:
        add("exp1_rerun", r["job_id"].split("|")[0], "causal", 1, r)
    for r in data["causal"]:
        protocol, mult, _ = r["job_id"].split("|", 2)
        add("exp3_causal", "confirm", protocol, float(mult), r)
    for h in data["hindsight"]:
        add("exp2_hindsight", h["contract"], "causal", 1, h, policy_override="hindsight_witness")
    return out


# ------------------------------------------------------------------ build
def build(out, report):
    out, report = Path(out), Path(report)
    data = load(out)
    H = histories(data)
    B = feasibility_tables(data)
    e1, e2, e3, e4 = (
        experiment1(data, H, B),
        experiment2(data, H, B),
        experiment3(data, B),
        experiment4(data, B),
    )
    report.mkdir(parents=True, exist_ok=True)
    for a, rows in e1["success"].items():
        write_csv(report / f"exp1_success_{a}.csv", rows)
    write_csv(report / "exp1_evidence_completion.csv", e1["evidence"])
    write_csv(
        report / "exp1_ranking_reversals.csv",
        e1["reversals"] or [dict(analysis=None)],
        [
            "analysis",
            "policy_a",
            "policy_b",
            "contract_x",
            "contract_y",
            "a_minus_b_x",
            "a_minus_b_y",
            "min_abs_margin",
            "recordings_reversing_same_direction",
        ],
    )
    write_csv(report / "exp1_decision_changes.csv", e1["decision_changes"])
    write_csv(report / "exp1_per_recording.csv", e1["per_recording"])
    write_csv(report / "exp2_bounds.csv", e2["bounds"])
    write_csv(report / "exp2_condition_classes.csv", e2["classes"])
    write_csv(report / "exp2_policy_gap.csv", e2["gap"])
    write_csv(report / "exp3_metrics.csv", e3["metrics"])
    write_csv(report / "exp3_protocol_disagreement.csv", e3["disagreement"])
    write_csv(
        report / "exp3_success_flips.csv",
        e3["flips"],
        [
            "protocol",
            "run_id",
            "policy",
            "split",
            "from_multiplier",
            "to_multiplier",
            "direction",
            "processed_from",
            "processed_to",
            "late_from",
            "late_to",
            "failed_to",
            "confirmation_lost_to_skipping",
        ],
    )
    write_csv(report / "exp4_unmet_patterns.csv", e4["patterns"])
    write_csv(report / "exp4_urgency_comparison.csv", e4["urgency"])
    write_csv(report / "exp4_evidence_funnel.csv", e4["funnel"])
    write_csv(report / "exp4_counterfactual_relaxations.csv", e4["relaxations"])
    write_csv(report / "exp4_per_run.csv", e4["per_run"])
    write_csv(report / "runs.csv", per_run_rows(data))
    sens = None
    if (out / "sweep.jsonl").exists():
        from experiments.validation.sensitivity import sensitivity

        sens = sensitivity(out)
        for name in ("table", "classes", "pairs", "drivers", "per_recording"):
            stem = "settings" if name == "table" else name
            write_csv(report / f"exp1_sensitivity_{stem}.csv", sens[name])
    (report / "exp3_timelines.json").write_text(
        json.dumps(e3["timelines"], indent=1, sort_keys=True) + "\n"
    )
    source = Path(__file__).parent
    summary = dict(
        schema_version="validation_summary_v1",
        config_sha256=hashlib.sha256((source / "config.json").read_bytes()).hexdigest(),
        source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.glob("*.py"))
        },
        parity=data["parity"],
        counts=dict(
            reference_runs=len(data["reference"]),
            rerun_runs=len(data["reruns"]),
            causal_runs=len(data["causal"]),
            hindsight_searches=len(data["hindsight"]),
            hindsight_evaluations=sum(h["evaluations"] for h in data["hindsight"]),
            recordings=6,
            route_families=3,
            maps=1,
            conditions_per_contract=72,
        ),
        exp1=dict(
            best=e1["best"],
            reversal_counts={
                a: sum(1 for r in e1["reversals"] if r["analysis"] == a)
                for a in ("rescore", "rerun")
            },
            decision_changes=e1["decision_changes"],
        ),
        exp2=dict(class_counts=e2["class_counts"], gap=e2["gap"]),
        exp3=dict(
            disagreement=e3["disagreement"],
            reversals=e3["reversals"],
            shapes=e3["shapes"],
            flip_counts=[
                dict(
                    protocol=k[0],
                    from_multiplier=k[1],
                    to_multiplier=k[2],
                    direction=k[3],
                    confirmation_lost_to_skipping=k[4],
                    runs=n,
                )
                for k, n in sorted(
                    Counter(
                        (
                            f["protocol"],
                            f["from_multiplier"],
                            f["to_multiplier"],
                            f["direction"],
                            f["confirmation_lost_to_skipping"],
                        )
                        for f in e3["flips"]
                    ).items()
                )
            ],
        ),
        sensitivity=None
        if sens is None
        else dict(
            receipt=json.loads((out / "sweep_receipt.json").read_text()),
            settings=len(sens["table"]),
            all_zero_settings=sens["all_zero_settings"],
            class_wins=sens["classes"],
            necessary_values=sens["necessary"],
            fast_beats_accurate_fixed=[
                r["setting"] for r in sens["table"] if r["fast_beats_accurate_fixed"]
            ],
            fast_beats_all_others=[
                r["setting"] for r in sens["table"] if r["fast_beats_all_others"]
            ],
        ),
        exp4=dict(
            urgency=[u for u in e4["urgency"] if u["schedule"] == "all"],
            funnel=e4["funnel"],
            relaxations=e4["relaxations"],
        ),
    )
    (report / "summary.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n"
    )
    print("wrote", report)
    return summary
