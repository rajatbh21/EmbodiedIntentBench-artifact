"""RGB-only conditional recall; binomial likelihood is a working loss, not calibration.

Targets within a frame and adjacent captures are dependent. No inferential confidence
intervals are derived from the working binomial loss. Empty-visible frames contribute
zero trials, rather than being mislabeled as detection failures.
"""

from __future__ import annotations

import numpy as np

from experiments.mission_routing_v2.features import NAMES as BASE_NAMES
from experiments.mission_routing_v2.features import encode as base_encode

NAMES = (
    *BASE_NAMES,
    *(
        f"cell_{r}_{c}_{s}"
        for r in range(2)
        for c in range(3)
        for s in ("luminance", "contrast", "dark_fraction", "gradient")
    ),
)
ALPHA = 10.0


def encode(rgb):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 6:
        raise ValueError("Expected uint8 RGB of at least 6 by 6 pixels")
    # Deterministic 48x64 sampling caps work independently of camera resolution.
    image = (
        rgb[np.linspace(0, rgb.shape[0] - 1, 48).astype(int)][
            :, np.linspace(0, rgb.shape[1] - 1, 64).astype(int)
        ].astype(float)
        / 255
    )
    gray = image @ np.array([0.2126, 0.7152, 0.0722])
    values = list(base_encode(rgb))
    for stripe in np.array_split(gray, 2):
        for cell in np.array_split(stripe, 3, axis=1):
            values.extend(
                [
                    cell.mean(),
                    cell.std(),
                    (cell < 0.2).mean(),
                    (np.abs(np.diff(cell, axis=0)).mean() + np.abs(np.diff(cell, axis=1)).mean())
                    / 2,
                ]
            )
    result = np.asarray(values)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite RGB features")
    return result


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -np.asarray(x)))


def fit(x, hits, trials):
    x, hits, trials = np.asarray(x, float), np.asarray(hits, float), np.asarray(trials, float)
    if (
        x.ndim != 2
        or x.shape[1] not in (len(BASE_NAMES), len(NAMES))
        or not len(x)
        or hits.shape != (len(x),)
        or trials.shape != hits.shape
        or not all(np.isfinite(v).all() for v in (x, hits, trials))
        or not np.all(x[:, 0] == 1)
        or np.any(hits < 0)
        or np.any(trials < hits)
        or np.any(hits != np.floor(hits))
        or np.any(trials != np.floor(trials))
    ):
        raise ValueError("Complete finite features and integer binomial counts required")
    active = trials > 0
    if not active.any():
        raise ValueError("At least one visible target trial required")
    x, hits, trials = x[active], hits[active], trials[active]
    mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 0.01)
    mean[0], scale[0] = 0, 1
    z = (x - mean) / scale
    # Add one hit and one miss at standardized feature origin; penalize slopes only.
    z = np.vstack([z, np.eye(1, x.shape[1])[0]])
    hits, trials = np.r_[hits, 1], np.r_[trials, 2]
    penalty = np.diag([0.0] + [ALPHA] * (x.shape[1] - 1))
    coefficient = np.zeros(x.shape[1])
    coefficient[0] = np.log(hits.sum() / (trials - hits).sum())

    def loss(beta):
        eta = z @ beta
        return float(
            np.sum(trials * np.logaddexp(0, eta) - hits * eta) + 0.5 * beta @ penalty @ beta
        )

    for iteration in range(100):  # noqa: B007 (recorded after convergence)
        p = sigmoid(z @ coefficient)
        gradient = z.T @ (trials * p - hits) + penalty @ coefficient
        if np.max(np.abs(gradient)) < 1e-7:
            break
        hessian = z.T @ ((trials * p * (1 - p))[:, None] * z) + penalty
        step = np.linalg.solve(hessian, gradient)
        rate, before = 1.0, loss(coefficient)
        while loss(coefficient - rate * step) > before - 1e-4 * rate * gradient @ step:
            rate /= 2
            if rate < 1e-10:
                raise ValueError("Recall optimizer failed to descend")
        coefficient -= rate * step
    else:
        raise ValueError("Recall optimizer did not converge")
    return dict(
        mean=mean.tolist(),
        scale=scale.tolist(),
        coefficient=coefficient.tolist(),
        global_mean=float(hits.sum() / trials.sum()),
        alpha=ALPHA,
        samples=int(active.sum()),
        empty_frames_excluded=int((~active).sum()),
        target_trials=int(trials.sum() - 2),
        target_hits=int(hits.sum() - 1),
        iterations=iteration,
        train_min=x.min(axis=0).tolist(),
        train_max=x.max(axis=0).tolist(),
    )


def predict(model, features):
    x = np.asarray(features, float)
    if x.shape != (len(model["coefficient"]),) or not np.isfinite(x).all():
        raise ValueError("Wrong recall feature shape or nonfinite input")
    z = (x - model["mean"]) / model["scale"]
    value = float(sigmoid(z @ model["coefficient"]))
    outside = float(
        np.mean(
            (x[1:] < np.asarray(model["train_min"])[1:])
            | (x[1:] > np.asarray(model["train_max"])[1:])
        )
    )
    return value, outside
