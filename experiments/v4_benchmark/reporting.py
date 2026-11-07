"""Prediction-only image-space report baseline and separate hidden-GT scoring.

Association is greedy bounding-box IoU, not physical person identification. Camera
motion/occlusion can fragment tracks; duplicate reports expose that limitation.
"""

import math
from dataclasses import asdict, dataclass

import numpy as np

from missionbench.schemas.loading import SchemaValidationError
from missionbench.v2.evaluation import _target_instances
from missionbench.v2.executors import label_components
from experiments.v4_benchmark.projection import ground_point


@dataclass(frozen=True)
class ReportingConfig:
    association: str = "image_iou"
    max_distance_m: float = 1.0
    min_hits: int = 2
    max_gap_s: float = 2.0
    association_iou: float = 0.2

    @classmethod
    def parse(cls, data):
        if not isinstance(data, dict) or set(data) - set(cls.__dataclass_fields__):
            raise SchemaValidationError("unknown reporting settings")
        obj = cls(**data)
        if obj.association not in ("image_iou", "ground_plane"):
            raise SchemaValidationError("unknown reporting association")
        if type(obj.min_hits) is not int or obj.min_hits < 1:
            raise SchemaValidationError("reporting min_hits must be a positive integer")
        for key in ("max_gap_s", "association_iou", "max_distance_m"):
            value = getattr(obj, key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise SchemaValidationError(f"reporting {key} must be finite")
        if obj.max_gap_s <= 0 or obj.max_distance_m <= 0 or not 0 < obj.association_iou <= 1:
            raise SchemaValidationError("invalid reporting gap or IoU")
        return obj

    def to_dict(self):
        return asdict(self)


def components(mask):
    labels, count = label_components(mask)
    ids = [i for i in range(1, count + 1) if np.count_nonzero(labels == i) >= 4]
    return labels, ids


def box_iou(a, b):
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / union


class ReportTracker:
    def __init__(self, config):
        self.config = config
        self.tracks = {}
        self.next_id = 1
        self.report_count = 0

    def update(self, mask, capture_s, completion_s, *, pose=None, camera_pitch_rad=None):
        labels, ids = components(mask)
        boxes = {}
        for cid in ids:
            y, x = np.where(labels == cid)
            boxes[cid] = (int(x.min()), int(y.min()), int(x.max() + 1), int(y.max() + 1))
        positions = {}
        if self.config.association == "ground_plane":
            if pose is None or camera_pitch_rad is None:
                raise ValueError("ground association requires public pose and camera pitch")
            for cid, box in boxes.items():
                positions[cid] = ground_point(
                    ((box[0] + box[2] - 1) / 2, box[3] - 1), mask.shape, pose, camera_pitch_rad
                )
        self.tracks = {
            tid: t
            for tid, t in self.tracks.items()
            if capture_s - t["last_s"] <= self.config.max_gap_s
        }
        pairs = []
        for tid, track in self.tracks.items():
            for cid, box in boxes.items():
                if self.config.association == "ground_plane":
                    point, previous = positions.get(cid), track.get("position")
                    if point is None or previous is None:
                        continue
                    distance = math.dist(point, previous)
                    if distance <= self.config.max_distance_m:
                        pairs.append((distance, tid, cid))
                else:
                    overlap = box_iou(track["box"], box)
                    if overlap >= self.config.association_iou:
                        pairs.append((-overlap, tid, cid))
        assigned, used = {}, set()
        for _, tid, cid in sorted(pairs):
            if tid not in used and cid not in assigned:
                assigned[cid] = tid
                used.add(tid)
        emitted = []
        for cid, box in boxes.items():
            tid = assigned.get(cid)
            if tid is None:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = dict(first_s=capture_s, hits=0, reported=False)
            track = self.tracks[tid]
            track.update(
                box=box, position=positions.get(cid), last_s=capture_s, hits=track["hits"] + 1
            )
            if not track["reported"] and track["hits"] >= self.config.min_hits:
                track["reported"] = True
                self.report_count += 1
                emitted.append(
                    dict(
                        report_id=tid,
                        component_id=cid,
                        first_prediction_s=track["first_s"],
                        completion_time_s=completion_s,
                        confirmation_latency_s=completion_s - track["first_s"],
                    )
                )
        return emitted


class ReportEvaluator:
    """Score only emitted reports; GT never feeds association or confirmation."""

    def __init__(self, threshold, total_targets):
        self.threshold, self.total_targets = threshold, total_targets
        self.reports = []
        self.first_visible = {}

    def note_visibility(self, observation):
        for target in observation.visible_target_ids:
            self.first_visible.setdefault(target, observation.scheduled_capture_time_s)

    def update(self, observation, mask, reports):
        self.note_visibility(observation)
        labels, ids = components(mask)
        targets = _target_instances(observation)
        candidates = []
        for cid in ids:
            pred = labels == cid
            for instance, target in targets.items():
                gt = observation.instance_gt == instance
                intersection = np.count_nonzero(pred & gt)
                iou = intersection / np.count_nonzero(pred | gt)
                if intersection and iou >= self.threshold:
                    candidates.append((-iou, cid, instance, target))
        matched, used = {}, set()
        for _, cid, instance, target in sorted(candidates):
            if cid not in matched and instance not in used:
                matched[cid] = target
                used.add(instance)
        for report in reports:
            target = matched.get(report["component_id"])
            self.reports.append(
                {
                    **report,
                    "target_id": target,
                    "delay_from_first_visibility_s": None
                    if target is None
                    else report["completion_time_s"] - self.first_visible[target],
                }
            )

    def summary(self, duration_s):
        correct = [r for r in self.reports if r["target_id"] is not None]
        unique = len({r["target_id"] for r in correct})
        false = len(self.reports) - len(correct)
        return dict(
            reports=self.reports,
            total_reports=len(self.reports),
            correct_reports=len(correct),
            false_reports=false,
            duplicate_correct_reports=len(correct) - unique,
            unique_targets_reported=unique,
            report_recall=unique / self.total_targets if self.total_targets else 1.0,
            report_precision=len(correct) / len(self.reports) if self.reports else None,
            false_reports_per_mission_minute=false * 60 / duration_s if duration_s else 0.0,
            first_confirmed_correct_report_s=min(
                (r["completion_time_s"] for r in correct), default=None
            ),
            scoring="capture_mask_iou_at_report_emission_v1",
        )
