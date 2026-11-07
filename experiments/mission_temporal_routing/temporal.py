"""Predict repeated support from current RGB, learning temporal dependence offline."""

from itertools import combinations

from experiments.mission_routing_v5.quality import predict

REQUIREMENTS = ((2, 2), (3, 4))
PATTERNS = tuple((0, *tail) for size in range(4) for tail in combinations((2, 4, 6), size))


def head_key(hits, span, offsets):
    if (hits, span) not in REQUIREMENTS or tuple(offsets) not in PATTERNS:
        raise ValueError("Unsupported repetition requirement or capture pattern")
    return f"k{hits}_s{span}_t" + "_".join(str(int(t)) for t in offsets)


def possible(hits, span, offsets):
    return len(offsets) >= hits and offsets[-1] - offsets[0] >= span


def window_labels(rows, hits, span, offsets):
    """Training-only GT: fraction of currently visible identities supported later.

    The denominator is identical for all configurations. Incomplete windows are
    omitted, never padded as failures. Future identities never enter features.
    """
    head_key(hits, span, offsets)
    if not rows or [r["capture_id"] for r in rows] != list(range(len(rows))):
        raise ValueError("Complete ordered capture rows required")
    if any(abs(r["capture_time_s"] - i * 2) > 1e-8 for i, r in enumerate(rows)):
        raise ValueError("Two-second capture clock required")
    for r in rows:
        if not set(r["matched_target_ids"]) <= set(r["visible_target_ids"]):
            raise ValueError("Matched identities must be visible")
    output = []
    for index in range(len(rows) - int(max(offsets) / 2)):
        visible = set(rows[index]["visible_target_ids"])
        supported = 0
        for identity in visible:
            times = [
                t for t in offsets if identity in rows[index + int(t / 2)]["matched_target_ids"]
            ]
            supported += bool(len(times) >= hits and times[-1] - times[0] >= span)
        output.append((index, supported, len(visible)))
    return output


def support(model, features, hits, span, offsets, *, context=True):
    if not offsets:
        return 0.0
    key = head_key(hits, span, offsets)
    # Structural impossibility must not acquire positive pseudo-count probability.
    if not possible(hits, span, offsets):
        return 0.0
    head = model["heads"][key]
    return predict(head, features)[0] if context else head["global_mean"]
