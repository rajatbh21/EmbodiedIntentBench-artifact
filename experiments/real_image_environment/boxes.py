"""Single-class diagnostic matching; not COCO AP or unique-person mission recall."""

import math


def validate_box(box):
    if (
        len(box) != 4
        or not all(math.isfinite(v) for v in box)
        or box[2] <= box[0]
        or box[3] <= box[1]
    ):
        raise ValueError("invalid xyxy box")


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union


def match(predictions, targets, threshold=0.5):
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("invalid IoU threshold")
    for box in targets:
        validate_box(box)
    for p in predictions:
        validate_box(p["box"])
        if not math.isfinite(p["score"]) or not 0 <= p["score"] <= 1:
            raise ValueError("invalid confidence")
    used, matches = set(), []
    for index in sorted(range(len(predictions)), key=lambda i: (-predictions[i]["score"], i)):
        candidates = [
            (iou(predictions[index]["box"], box), j)
            for j, box in enumerate(targets)
            if j not in used
        ]
        overlap, target = max(candidates, default=(0, -1))
        if target >= 0 and overlap >= threshold:
            used.add(target)
            matches.append(dict(prediction=index, target=target, iou=overlap))
    return dict(
        tp=len(matches),
        fp=len(predictions) - len(matches),
        fn=len(targets) - len(matches),
        matches=matches,
    )
