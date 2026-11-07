"""Same binomial ridge objective with a Newton-decrement convergence safeguard.

The absolute gradient check can stall at machine precision for sparse temporal
labels. A positive Newton decrement below 1e-12 provides a second criterion.
Regularization, pseudo counts, features and model selection remain unchanged.
"""

import numpy as np

from experiments.mission_routing_v5.quality import ALPHA, BASE_NAMES, NAMES, sigmoid


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
        # Newton decrement bounds local optimization error when loss changes round to zero.
        if 0 <= float(gradient @ step) / 2 < 1e-12:
            break
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
