"""Declared statistics of the 144-setting intent-threshold sweep (Experiment 1c).

Responsibility: from ``sweep.jsonl`` (one success bit per run per setting),
compute exactly the statistics declared in ``config.json`` under
``sensitivity_map`` before any setting was scored: per-setting rankings, the
materiality rule, class-level wins, pairwise orderings versus the reference
setting, and parameter drivers of any fast-model win.

Boundaries: reads receipts only; no execution, no rescoring, no new rules.
"""

from __future__ import annotations

import itertools
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
MARGIN = 3  # declared: >= 3 of 72 conditions, and no recording in the opposite direction
REFERENCE = (2, 2, 2, True, 44.0)


def label(value):
    return "none" if value is None else value


def sensitivity(out):
    spec = contracts.CONFIG["sensitivity_map"]
    classes = spec["classes"]
    settings = contracts.sweep_settings()
    rows = execution.read_jsonl(Path(out) / "sweep.jsonl")
    recordings = sorted({r["row"]["flight"] for r in rows})
    n_set = len(settings)
    succ = [dict.fromkeys(POLICIES, 0) for _ in settings]
    rec = [{p: dict.fromkeys(recordings, 0) for p in POLICIES} for _ in settings]
    for r in rows:
        p, f = r["row"]["policy"], r["row"]["flight"]
        for n, bit in enumerate(r["sweep"]):
            if bit == "1":
                succ[n][p] += 1
                rec[n][p][f] += 1

    def material(n, a, b):
        return succ[n][a] - succ[n][b] >= MARGIN and all(
            rec[n][b][f] <= rec[n][a][f] for f in recordings
        )

    def best(n, cls):
        return max(classes[cls], key=lambda p: (succ[n][p], -classes[cls].index(p)))

    def beats(n, x, y):
        b = best(n, x)
        return all(material(n, b, q) for q in classes[y])

    ref = next(n for n, s in enumerate(settings) if tuple(s.values()) == REFERENCE)
    table = []
    for n, s in enumerate(settings):
        top = max(succ[n].values())
        row = dict(setting=n, **{k: label(v) for k, v in s.items()})
        row.update({p: succ[n][p] for p in POLICIES})
        row.update(
            max_successes=top,
            best_policies=[p for p in POLICIES if succ[n][p] == top] if top else [],
            best_fast=best(n, "fast"),
        )
        for x, y in itertools.permutations(classes, 2):
            row[f"{x}_beats_{y}"] = beats(n, x, y)
        row["fast_beats_all_others"] = (
            row["fast_beats_accurate_fixed"] and row["fast_beats_adaptive"]
        )
        table.append(row)
    class_rows = []
    for x, y in itertools.permutations(classes, 2):
        k = sum(r[f"{x}_beats_{y}"] for r in table)
        class_rows.append(dict(winner=x, loser=y, settings_material=k, fraction=k / n_set))
    pairs = []
    for a, b in itertools.combinations(POLICIES, 2):
        d0 = succ[ref][a] - succ[ref][b]
        diffs = [succ[n][a] - succ[n][b] for n in range(n_set)]
        pairs.append(
            dict(
                policy_a=a,
                policy_b=b,
                reference_a_minus_b=d0,
                a_material=sum(material(n, a, b) for n in range(n_set)),
                b_material=sum(material(n, b, a) for n in range(n_set)),
                a_more=sum(d > 0 for d in diffs),
                b_more=sum(d < 0 for d in diffs),
                tied=sum(d == 0 for d in diffs),
                any_reversal_vs_reference=sum(d * d0 < 0 for d in diffs),
                material_reversal_vs_reference=None
                if d0 == 0
                else sum(material(n, b, a) if d0 > 0 else material(n, a, b) for n in range(n_set)),
            )
        )
    flags = ("fast_beats_accurate_fixed", "fast_beats_all_others")
    drivers = []
    for flag in flags:
        wins = [r for r in table if r[flag]]
        for name, param in spec["parameters"].items():
            for v in param["values"]:
                total = sum(1 for r in table if r[name] == label(v))
                k = sum(1 for r in wins if r[name] == label(v))
                drivers.append(
                    dict(
                        definition=flag,
                        parameter=name,
                        value=label(v),
                        fast_win_settings=k,
                        settings_with_value=total,
                        share_of_fast_wins=k / len(wins) if wins else None,
                        win_rate_at_value=k / total,
                    )
                )
    necessary = {
        flag: {
            name: sorted({str(r[name]) for r in table if r[flag]}) for name in spec["parameters"]
        }
        for flag in flags
    }
    per_recording = [
        dict(setting=n, policy=p, recording=f, successes=rec[n][p][f])
        for n in range(n_set)
        for p in POLICIES
        for f in recordings
    ]
    return dict(
        table=table,
        classes=class_rows,
        pairs=pairs,
        drivers=drivers,
        necessary=necessary,
        per_recording=per_recording,
        all_zero_settings=sum(1 for r in table if r["max_successes"] == 0),
        reference_setting=ref,
    )
