"""Small learned report gates with public evidence features and an immutable intent floor.

These models classify identity-consistent support. They do not infer target identities,
relax mission requirements, replace the report evaluator, or constitute a language model.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from experiments.mission_routing_v2.models import POOL

CONFIG_IDS = tuple(POOL)
FEATURE_NAMES = (
    "log1p_hits",
    "log1p_span_s",
    "log1p_last_gap_s",
    "unique_source_fraction",
    *("source_fraction:" + name for name in CONFIG_IDS),
)
MODELS = ("logistic", "mlp8")
HYPERPARAMETERS = {
    "seed": 20261027,
    "iterations": 1000,
    "learning_rate": 0.03,
    "l2": 0.03,
    "hidden_width": 8,
    "threshold": 0.5,
    "optimizer": "full_batch_adam",
    "normalization": "training_mean_std_floor_1e-8",
    "class_weighting": "none",
    "early_stopping": False,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def feature_vector(public):
    """Only ordered prediction evidence and source identities affect the model input."""
    require(public.get("ground_truth_access") is False, "Public-only evidence declaration required")
    forbidden = {"label", "target_id", "support_target_ids", "identity_consistent", "correct"}
    require(not forbidden.intersection(public), "Evaluator fields forbidden at learned gate")
    refs = public["evidence"]
    require(isinstance(refs, (tuple, list)) and refs, "Nonempty selected evidence required")
    allowed = {"capture_ns", "component_id", "source_config_id"}
    require(all(set(r) == allowed for r in refs), "Only prediction evidence references allowed")
    times = [r["capture_ns"] for r in refs]
    require(all(type(t) is int and t >= 0 for t in times), "Integer causal evidence times required")
    require(times == sorted(set(times)), "Evidence times must be distinct and ordered")
    require(times[-1] == public["current_capture_ns"], "Current candidate must end its support")
    sources = [r["source_config_id"] for r in refs]
    require(set(sources) <= set(CONFIG_IDS), "Unknown pretrained source configuration")
    require(public["source_config_ids"] == sorted(set(sources)), "Source summary differs")
    hits, span = len(times), times[-1] - times[0]
    gap = times[-1] - times[-2] if hits > 1 else None
    require(
        public["hits"] == hits and public["span_ns"] == span and public["last_gap_ns"] == gap,
        "Candidate evidence summary differs",
    )
    values = [
        math.log1p(hits),
        math.log1p(span / 1e9),
        math.log1p((gap or 0) / 1e9),
        len(set(sources)) / len(CONFIG_IDS),
        *(sources.count(c) / hits for c in CONFIG_IDS),
    ]
    result = np.asarray(values, dtype=np.float64)
    require(np.isfinite(result).all(), "Finite public features required")
    return result


def hard_floor(public, completion_ns, outer_deadline_ns=None):
    """Model scores can reject support; they can never lower a public mission floor."""
    feature_vector(public)
    require(type(completion_ns) is int and completion_ns >= 0, "Integer completion required")
    required_hits, required_span = public["required_hits"], public["required_span_ns"]
    require(type(required_hits) is int and required_hits >= 2, "Repeated public evidence floor")
    require(type(required_span) is int and required_span >= 0, "Nonnegative evidence span")
    deadline = public["phase_deadline_ns"]
    require(type(deadline) is int and deadline >= 0, "Public deadline required")
    if outer_deadline_ns is not None:
        require(
            type(outer_deadline_ns) is int and outer_deadline_ns >= 0, "Outer deadline required"
        )
        deadline = min(deadline, outer_deadline_ns)
    return bool(
        public["current_capture_ns"] <= completion_ns <= deadline
        and public["hits"] >= required_hits
        and public["span_ns"] >= required_span
    )


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


def forward(x, parameters, kind):
    if kind == "logistic":
        return sigmoid(x @ parameters["w"] + parameters["b"]), None
    hidden = np.tanh(x @ parameters["w1"] + parameters["b1"])
    return sigmoid(hidden @ parameters["w2"] + parameters["b2"]), hidden


def fit(x, y, kind, provenance):
    require(kind in MODELS, "Unknown learned model")
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    require(x.ndim == 2 and x.shape[1] == len(FEATURE_NAMES), "Feature matrix shape")
    require(y.shape == (len(x),) and set(y) == {0.0, 1.0}, "Both true label classes required")
    require(len(y) >= 20 and min(np.sum(y == 0), np.sum(y == 1)) >= 5, "Insufficient class support")
    require(np.isfinite(x).all(), "Finite training features required")
    mean, std = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-8)
    x = (x - mean) / std
    h = dict(HYPERPARAMETERS)
    rng = np.random.default_rng(h["seed"])
    if kind == "logistic":
        parameters = {"w": np.zeros(x.shape[1]), "b": np.asarray(0.0)}
    else:
        parameters = dict(
            w1=rng.normal(0, 1 / math.sqrt(x.shape[1]), (x.shape[1], h["hidden_width"])),
            b1=np.zeros(h["hidden_width"]),
            w2=rng.normal(0, 1 / math.sqrt(h["hidden_width"]), h["hidden_width"]),
            b2=np.asarray(0.0),
        )
    first = {k: v.copy() for k, v in parameters.items()}
    momentum, variance = ({k: np.zeros_like(v) for k, v in parameters.items()} for _ in range(2))
    history = []
    for step in range(1, h["iterations"] + 1):
        prob, hidden = forward(x, parameters, kind)
        err = (prob - y) / len(y)
        if kind == "logistic":
            gradient = dict(w=x.T @ err + h["l2"] * parameters["w"], b=err.sum())
        else:
            dh = err[:, None] * parameters["w2"][None, :] * (1 - hidden**2)
            gradient = dict(
                w1=x.T @ dh + h["l2"] * parameters["w1"],
                b1=dh.sum(axis=0),
                w2=hidden.T @ err + h["l2"] * parameters["w2"],
                b2=err.sum(),
            )
        for key in parameters:
            momentum[key] = 0.9 * momentum[key] + 0.1 * gradient[key]
            variance[key] = 0.999 * variance[key] + 0.001 * np.square(gradient[key])
            parameters[key] -= (
                h["learning_rate"]
                * (momentum[key] / (1 - 0.9**step))
                / (np.sqrt(variance[key] / (1 - 0.999**step)) + 1e-8)
            )
        if step in (1, 100, 500, h["iterations"]):
            history.append(dict(iteration=step, training_log_loss=metrics(y, prob)["log_loss"]))
    require(
        any(not np.array_equal(v, first[k]) for k, v in parameters.items()), "No learned update"
    )
    return dict(
        schema_version="learned_report_verifier_v1",
        model=kind,
        feature_names=list(FEATURE_NAMES),
        source_config_ids=list(CONFIG_IDS),
        hyperparameters=h,
        normalizer=dict(mean=mean.tolist(), std=std.tolist()),
        parameters={k: v.tolist() for k, v in parameters.items()},
        parameter_count=sum(v.size for v in parameters.values()),
        optimization_history=history,
        training_examples=len(y),
        training_positive=int(y.sum()),
        training_negative=int((1 - y).sum()),
        training_provenance=provenance,
        fit_uses_held_out_data=False,
        mission_success_claim=False,
    )


def metrics(y, probabilities):
    y, p = np.asarray(y, dtype=float), np.asarray(probabilities, dtype=float)
    require(y.shape == p.shape and y.ndim == 1 and len(y), "Paired nonempty predictions required")
    require(
        set(y) <= {0, 1} and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "Invalid scores"
    )
    pred = p >= HYPERPARAMETERS["threshold"]
    tp, fp = int(np.sum(pred & (y == 1))), int(np.sum(pred & (y == 0)))
    tn, fn = int(np.sum(~pred & (y == 0))), int(np.sum(~pred & (y == 1)))
    clipped = np.clip(p, 1e-12, 1 - 1e-12)
    return dict(
        candidates=len(y),
        positives=int(y.sum()),
        negatives=int((1 - y).sum()),
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
        acceptance_precision=tp / (tp + fp) if tp + fp else None,
        positive_acceptance_recall=tp / (tp + fn) if tp + fn else None,
        brier=float(np.mean((p - y) ** 2)),
        log_loss=float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))),
        threshold=HYPERPARAMETERS["threshold"],
    )


class Model:
    def __init__(self, checkpoint):
        require(checkpoint["schema_version"] == "learned_report_verifier_v1", "Unknown checkpoint")
        require(checkpoint["feature_names"] == list(FEATURE_NAMES), "Changed feature order")
        require(checkpoint["source_config_ids"] == list(CONFIG_IDS), "Changed source pool")
        require(checkpoint["model"] in MODELS, "Unknown fitted architecture")
        require(
            checkpoint["hyperparameters"] == HYPERPARAMETERS, "Changed frozen training settings"
        )
        self.document = json.loads(json.dumps(checkpoint))
        self.kind, self.sha256 = checkpoint["model"], digest(checkpoint)
        self.mean = np.asarray(checkpoint["normalizer"]["mean"], dtype=float)
        self.std = np.asarray(checkpoint["normalizer"]["std"], dtype=float)
        self.parameters = {
            k: np.asarray(v, dtype=float) for k, v in checkpoint["parameters"].items()
        }
        require(self.mean.shape == self.std.shape == (len(FEATURE_NAMES),), "Normalizer shape")
        require(
            np.isfinite(self.mean).all() and np.isfinite(self.std).all() and (self.std > 0).all(),
            "Normalizer values",
        )
        require(all(np.isfinite(v).all() for v in self.parameters.values()), "Finite model weights")
        probe, _ = forward(np.zeros((1, len(FEATURE_NAMES))), self.parameters, self.kind)
        require(probe.shape == (1,), "Architecture output shape")

    @classmethod
    def load(cls, path, expected_sha256=None):
        raw = Path(path).read_bytes()
        if expected_sha256 is not None:
            require(hashlib.sha256(raw).hexdigest() == expected_sha256, "Checkpoint file changed")
        return cls(json.loads(raw))

    def predict(self, public):
        x = (feature_vector(public) - self.mean) / self.std
        p, _ = forward(x[None, :], self.parameters, self.kind)
        return float(p[0])

    def decide(self, public, completion_ns, outer_deadline_ns=None):
        allowed = hard_floor(public, completion_ns, outer_deadline_ns)
        p = self.predict(public)
        return dict(
            accept=allowed and p >= HYPERPARAMETERS["threshold"],
            probability_identity_consistent=p,
            hard_public_floor_passed=allowed,
            threshold=HYPERPARAMETERS["threshold"],
            model_sha256=self.sha256,
            ground_truth_access=False,
        )
