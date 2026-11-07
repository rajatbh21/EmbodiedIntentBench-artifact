"""Evaluator-only report validation; never feed identity consistency back to tracking."""

from __future__ import annotations

from experiments.v4_benchmark.reporting import ReportEvaluator, components


def component_matches(frame, prediction, threshold):
    """Reuse frozen one-to-one mask matching, including its component-size filter."""
    _, ids = components(prediction)
    evaluator = ReportEvaluator(threshold, len(frame.visible_target_ids))
    evaluator.update(
        frame,
        prediction,
        [dict(component_id=cid, completion_time_s=frame.scheduled_capture_time_s) for cid in ids],
    )
    return {r["component_id"]: r["target_id"] for r in evaluator.reports}


class ConsistentReportEvaluator:
    """A report is correct only if every supporting observation matches one identity.

    The historical scorer checks only the mask at emission. This stricter scorer
    is versioned separately and applied identically to every compared tracker.
    """

    def __init__(self, target_ids, min_hits=2):
        self.targets = frozenset(target_ids)
        if not self.targets or type(min_hits) is not int or min_hits < 2:
            raise ValueError("Nonempty scope and repeated support required")
        self.min_hits = min_hits
        self.ledger, self.reports = {}, []
        self.last_capture = -1
        self.report_ids = set()

    def update(self, capture_s, matches, reports):
        if capture_s <= self.last_capture:
            raise ValueError("Evaluator captures must be increasing")
        self.last_capture = capture_s
        if any(target is not None and target not in self.targets for target in matches.values()):
            raise ValueError("Unknown evaluator identity")
        self.ledger.update({(capture_s, cid): target for cid, target in matches.items()})
        for report in reports:
            refs = [(item["capture_s"], item["component_id"]) for item in report["evidence"]]
            if (
                report["report_id"] in self.report_ids
                or len(refs) < self.min_hits
                or len({time for time, _ in refs}) != len(refs)
                or refs != sorted(refs)
                or refs[-1] != (capture_s, report["component_id"])
                or refs[0][0] != report["first_prediction_s"]
                or report["completion_time_s"] < capture_s
                or any(ref not in self.ledger for ref in refs)
            ):
                raise ValueError("Invalid, duplicated or noncausal report evidence")
            self.report_ids.add(report["report_id"])
            labels = [self.ledger[ref] for ref in refs]
            consistent = labels[-1] is not None and all(label == labels[-1] for label in labels)
            self.reports.append(
                {
                    **report,
                    "target_id": labels[-1] if consistent else None,
                    "last_frame_target_id": labels[-1],
                    "support_target_ids": labels,
                    "identity_consistent": consistent,
                }
            )

    def summary(self):
        correct = [r["target_id"] for r in self.reports if r["target_id"] is not None]
        unique = set(correct)
        return dict(
            scoring="consistent_report_support_v1",
            reports=self.reports,
            total_reports=len(self.reports),
            correct_reports=len(correct),
            false_reports=len(self.reports) - len(correct),
            duplicate_correct_reports=len(correct) - len(unique),
            unique_targets_reported=len(unique),
            report_recall=len(unique) / len(self.targets),
            report_precision=len(correct) / len(self.reports) if self.reports else None,
            first_confirmed_correct_report_s=min(
                (r["completion_time_s"] for r in self.reports if r["target_id"] is not None),
                default=None,
            ),
            mixed_identity_reports=sum(
                len({t for t in r["support_target_ids"] if t is not None}) > 1 for r in self.reports
            ),
            last_frame_correct_but_invalid_support=sum(
                r["last_frame_target_id"] is not None and not r["identity_consistent"]
                for r in self.reports
            ),
        )


class RunnerReportEvaluator:
    """Runner-facing GT adapter; only this side receives labeled frames."""

    def __init__(self, threshold, target_ids, min_hits=2):
        self.threshold = threshold
        self.evaluator = ConsistentReportEvaluator(target_ids, min_hits)
        self.first_visible = {}

    def note_visibility(self, frame):
        for target in frame.visible_target_ids:
            self.first_visible.setdefault(target, frame.scheduled_capture_time_s)

    def update(self, frame, prediction, reports):
        self.note_visibility(frame)
        self.evaluator.update(
            frame.scheduled_capture_time_s,
            component_matches(frame, prediction, self.threshold),
            reports,
        )

    def summary(self, duration_s):
        result = self.evaluator.summary()
        result["false_reports_per_mission_minute"] = (
            result["false_reports"] * 60 / duration_s if duration_s else 0
        )
        return result
