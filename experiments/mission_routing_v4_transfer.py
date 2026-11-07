"""Post-hoc count transfer on dense evaluation labels; never updates any model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.mission_routing_v2.features import encode
from experiments.mission_routing_v4.flights import verify
from experiments.mission_routing_v4.risk import estimate


def run(root):
    p = verify(root)
    profile = read(Path(p["risk_training"]) / "risk_profile.json")
    diagnostics = read(root / "feasibility_diagnostics.json")
    if diagnostics["protocol_sha256"] != sha(root / "protocol.json"):
        raise ValueError("Wrong diagnostic cohort")
    flights = {}
    for flight in p["cases"]:
        path = root / flight / "diagnostic_outcomes.json"
        if sha(path) != diagnostics["flights"][flight]["outcomes_sha256"]:
            raise ValueError("Changed evaluator-only outcomes")
        rows = read(path)
        frames = read(root / flight / "recording/recording.json")["frames"]
        features = []
        for frame in frames:
            path = root / flight / "recording" / frame["file"]
            if sha(path) != frame["sha256"]:
                raise ValueError("Changed physical capture")
            with np.load(path, allow_pickle=False) as data:
                features.append(encode(data["rgb"]))
        models = {}
        for config, data in profile["models"].items():
            observed = sorted(
                (r for r in rows if r["config_id"] == config), key=lambda r: r["capture_id"]
            )
            if [r["capture_id"] for r in observed] != list(range(len(features))):
                raise ValueError("Complete frame/config pairs required")
            truth = np.array([r["false_positive_components"] for r in observed])
            predicted = np.array([estimate(data["risk"], f) for f in features])
            models[config] = dict(
                captures=len(features),
                observed_total=int(truth.sum()),
                estimated_total=float(predicted.sum()),
                global_floor_total=data["risk"]["floor"] * len(features),
                mean_absolute_error=float(np.abs(truth - predicted).mean()),
                mean_signed_error=float((predicted - truth).mean()),
                nonzero_error_captures=int((truth > 0).sum()),
            )
        flights[flight] = models
    result = dict(
        schema_version="v4_dense_risk_transfer_v1",
        gt_aware=True,
        deployable_policy=False,
        posthoc=True,
        models_retrained=False,
        ignores_paid_capture_skipping=True,
        flights=flights,
        protocol_sha256=sha(root / "protocol.json"),
        diagnostics_sha256=sha(root / "feasibility_diagnostics.json"),
        risk_profile_sha256=p["risk_profile_sha256"],
        source_sha256=sha(Path(__file__)),
    )
    save(root / "risk_transfer.json", result)
    lines = [
        "# Dense risk transfer diagnostic",
        "",
        "Evaluator-only, after freezing and execution. No coefficients or thresholds change. "
        "Every model is scored on all 60 captures per flight, ignoring paid skipping. These "
        "totals are not the errors of an executed policy. Estimates are counts, not probability "
        "bounds; overestimation can suppress useful models and underestimation can admit errors.",
        "",
        "| Flight | Configuration | Observed false components | Contextual estimate | "
        "Global floor | MAE/capture |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for flight, models in flights.items():
        for c, m in models.items():
            lines.append(
                f"| {flight} | {c} | {m['observed_total']} | {m['estimated_total']:.2f} | "
                f"{m['global_floor_total']:.2f} | {m['mean_absolute_error']:.3f} |"
            )
    (root / "RISK_TRANSFER.md").write_text("\n".join(lines) + "\n")
    print(root / "RISK_TRANSFER.md", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
