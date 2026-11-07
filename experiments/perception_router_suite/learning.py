"""Small learned selectors with explicit full-information versus bandit feedback.

The bandit receives only its chosen arm's development reward. Evaluation is a
frozen greedy policy: it neither explores nor consumes evaluator feedback.
"""

from __future__ import annotations

import math

import numpy as np

from experiments.mission_routing_v2.features import NAMES as RGB_FEATURES

SCHEMA_VERSION = "perception_router_learning_v1"
FEATURES = (
    *RGB_FEATURES,
    "log_perception_latency_s",
    "log_perception_budget_s",
    "latency_budget_ratio",
    "log_inference_transport_energy_j",
    "log_communication_mb",
    "remote_placement",
    "battery_fraction",
    "log_available_reserve_kj",
    "required_available_reserve_ratio",
    "minimum_hits",
    "log_minimum_evidence_span_s",
    "log_false_component_allowance",
)


def context_vector(rgb_features, candidate_row, public_context):
    """Encode only RGB and public forecasts; no clock, scene or target identity."""
    rgb = np.asarray(rgb_features, dtype=float)
    if rgb.shape != (len(RGB_FEATURES),) or not np.isfinite(rgb).all() or rgb[0] != 1:
        raise ValueError("Expected finite public RGB features with a unit intercept")
    if "evaluator_target_ids" in public_context:
        raise ValueError("Evaluator identities cannot enter learned routing context")
    latency, budget = candidate_row["latency_s"], public_context["perception_budget_s"]
    energy = candidate_row["estimate"]["energy_j"]
    communication = candidate_row["estimate"]["communication_mb"]
    reserve = public_context["reserve_available_j"]
    required = candidate_row["reserve_required_j"]
    battery = public_context["battery_frac"]
    hits = public_context["min_hits"]
    span = public_context["min_evidence_span_s"]
    false = public_context["max_false_components"]
    values = (latency, budget, energy, communication, required, battery, hits, span, false)
    if (
        any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values)
        or type(reserve) not in (int, float)
        or not math.isfinite(reserve)
        or budget <= 0
        or not 0 <= battery <= 1
        or hits < 1
        or candidate_row["placement"] not in ("local", "remote")
    ):
        raise ValueError("Finite public resource and intent values required")
    # Ratios are clipped to contain extrapolation; feasibility is enforced separately.
    extra = (
        math.log1p(latency),
        math.log1p(budget),
        min(4.0, latency / budget),
        math.log1p(energy),
        math.log1p(communication),
        float(candidate_row["placement"] == "remote"),
        battery,
        math.log1p(max(0.0, reserve) / 1000),
        min(4.0, required / max(1.0, reserve)),
        float(hits),
        math.log1p(span),
        math.log1p(false),
    )
    return np.r_[rgb, extra]


def _standardizer(x):
    x = np.asarray(x, float)
    if (
        x.ndim != 2
        or x.shape[1] != len(FEATURES)
        or not len(x)
        or not np.isfinite(x).all()
        or not np.all(x[:, 0] == 1)
    ):
        raise ValueError("Nonempty finite routing feature matrix required")
    mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 0.05)
    mean[0], scale[0] = 0.0, 1.0
    return mean, scale


def fit_supervised(features_by_action, reward_for, *, alpha=10.0):
    """Full-information ridge; callback is intentionally called for every arm."""
    result = {}
    for action_id in sorted(features_by_action):
        x = np.asarray(features_by_action[action_id], float)
        mean, scale = _standardizer(x)
        y = np.asarray([reward_for(i, action_id) for i in range(len(x))], float)
        if not np.isfinite(y).all() or np.any((y < 0) | (y > 1)) or alpha <= 0:
            raise ValueError("Bounded development rewards and positive ridge penalty required")
        z = (x - mean) / scale
        coefficient = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ y)
        result[action_id] = dict(
            mean=mean.tolist(),
            scale=scale.tolist(),
            coefficient=coefficient.tolist(),
            samples=len(x),
            alpha=alpha,
        )
    return result


def fit_bandit(features_by_action, eligible, reward_for, *, seed=104729, alpha=0.5, ridge=1.0):
    """Disjoint LinUCB, exactly one selected-arm reward query/update per decision.

    Feature standardization may inspect all unlabeled development contexts. No
    unselected reward is passed to this learner or used in its parameter updates.
    """
    if not features_by_action or alpha < 0 or ridge <= 0:
        raise ValueError("Nonempty actions and valid LinUCB hyperparameters required")
    actions = sorted(features_by_action)
    count = len(eligible)
    if any(len(features_by_action[a]) != count for a in actions):
        raise ValueError("Aligned context/eligibility rows required")
    rng = np.random.default_rng(seed)
    means, scales, z, inverse, b, updates = {}, {}, {}, {}, {}, {}
    for action in actions:
        means[action], scales[action] = _standardizer(features_by_action[action])
        z[action] = (np.asarray(features_by_action[action]) - means[action]) / scales[action]
        inverse[action] = np.eye(len(FEATURES)) / ridge
        b[action] = np.zeros(len(FEATURES))
        updates[action] = 0
    trace = []
    for i in rng.permutation(count):
        admitted = sorted(eligible[int(i)])
        if not admitted or not set(admitted) <= set(actions):
            raise ValueError("Every training decision needs a nonempty known eligible set")
        scores = []
        for action in admitted:
            x = z[action][i]
            estimate = float(x @ inverse[action] @ b[action])
            uncertainty = math.sqrt(max(0.0, float(x @ inverse[action] @ x)))
            scores.append(estimate + alpha * uncertainty)
        top = np.flatnonzero(np.isclose(scores, max(scores), rtol=0, atol=1e-12))
        action = admitted[int(rng.choice(top))]
        reward = float(reward_for(int(i), action))
        if not math.isfinite(reward) or not 0 <= reward <= 1:
            raise ValueError("Bounded chosen-arm development reward required")
        x = z[action][i]
        v = inverse[action] @ x
        inverse[action] -= np.outer(v, v) / (1.0 + x @ v)
        b[action] += reward * x
        updates[action] += 1
        trace.append(
            dict(
                step=len(trace),
                context_index=int(i),
                action_id=action,
                reward=reward,
                action_update_count=updates[action],
                feedback_arms=1,
            )
        )
    result = {
        action: dict(
            mean=means[action].tolist(),
            scale=scales[action].tolist(),
            coefficient=(inverse[action] @ b[action]).tolist(),
            inverse=inverse[action].tolist(),
            updates=updates[action],
            exploration_alpha=alpha,
            ridge=ridge,
        )
        for action in actions
    }
    return result, trace


def predict(profile_section, action_id, vector):
    """Frozen clipped reward mean, with no UCB term and no state mutation."""
    model = profile_section[action_id]
    x = np.asarray(vector, float)
    mean, scale, coefficient = (
        np.asarray(model[k], float) for k in ("mean", "scale", "coefficient")
    )
    if any(
        v.shape != (len(FEATURES),) or not np.isfinite(v).all()
        for v in (x, mean, scale, coefficient)
    ) or np.any(scale <= 0):
        raise ValueError("Compatible finite learned-router feature vectors required")
    return float(np.clip(((x - mean) / scale) @ coefficient, 0, 1))
