"""Offline count estimates, not calibrated probabilities or statistical guarantees.

The floor is the largest add-one-smoothed mean among development flights. Ridge
adds image-dependent variation above that floor. Zero observed errors never mean
zero risk. Correlated frames and distribution shift preclude confidence claims.
"""

from __future__ import annotations

import math

import numpy as np

from experiments.mission_routing_v2.features import NAMES, fit


def fit_count(features, counts, groups):
    x, y = np.asarray(features, float), np.asarray(counts, float)
    if (
        x.ndim != 2
        or x.shape[1] != len(NAMES)
        or y.shape != (len(x),)
        or len(groups) != len(x)
        or not len(x)
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or np.any(y < 0)
        or np.any(y != np.floor(y))
    ):
        raise ValueError("Finite complete features and nonnegative integer counts required")
    means = [
        (sum(y[i] for i, g in enumerate(groups) if g == group) + 1)
        / (sum(g == group for g in groups) + 1)
        for group in sorted(set(groups))
    ]
    return dict(
        head=fit(x, y, alpha=1.0),
        floor=float(max(means)),
        pseudocount_per_capture=1 / (len(x) + 1),
        label="false_positive_components_per_capture",
    )


def estimate(model, features, *, context=True):
    x = np.asarray(features, float)
    head = model["head"]
    if x.shape != (len(NAMES),) or not np.isfinite(x).all():
        raise ValueError("Finite RGB feature vector required")
    z = (x - head["mean"]) / head["scale"]
    # Unlike the old quality predictor, a COUNT must not be clipped to one.
    contextual = max(0.0, float(z @ head["coefficient"])) + model["pseudocount_per_capture"]
    value = max(model["floor"], contextual) if context else model["floor"]
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Positive finite estimated risk required")
    return value


def phase_projection(phase, capture_s, slots, rate, spent):
    """Count all capture-local errors, even results arriving after the deadline."""
    if not phase["start_s"] <= capture_s < phase["end_s"]:
        raise ValueError("Risk must be charged to the capture phase")
    if any(not math.isfinite(v) or v < 0 for v in (rate, spent)):
        raise ValueError("Finite nonnegative risk required")
    count = sum(phase["start_s"] <= t < phase["end_s"] for t, _ in slots)
    if not count:
        raise ValueError("Projection must include the current action")
    projected = spent + count * rate
    return dict(
        phase_id=phase["phase_id"],
        expected_spent=spent,
        estimated_fp_per_capture=rate,
        remaining_captures=count,
        projected_false_components=projected,
        max_false_components=phase["max_false_components"],
        within_estimated_budget=projected <= phase["max_false_components"] + 1e-10,
    )
