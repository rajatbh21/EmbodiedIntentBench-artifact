"""Evaluator-only transfer/risk audit of frozen quality heads; never retrains them.

Mean quality and accumulated false components are distinct quantities. These
per-frame diagnostics ignore capture skipping and are not deployable policies.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.mission_routing_v2.features import encode, predict
from experiments.mission_routing_v3.flights import verify


def quality(row):
    return (
        len(row["matched_target_ids"])
        / max(1, len(row["visible_target_ids"]))
        * row["matched_components"]
        / max(1, row["predicted_components"])
    )


def run(root):
    p = verify(root)
    screen = Path(p["screen"])
    profile = read(screen / "calibration.json")
    diagnostics = read(root / "feasibility_diagnostics.json")
    if diagnostics["protocol_sha256"] != sha(root / "protocol.json"):
        raise ValueError("Wrong diagnostic cohort")
    development = [
        row
        for env in read(screen / "protocol.json")["environments"]
        for row in read(screen / env / "outcomes.json")
    ]
    results = {}
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
                raise ValueError("Changed capture")
            with np.load(path, allow_pickle=False) as data:
                features.append(encode(data["rgb"]))
        models = {}
        for config, model in profile["models"].items():
            observed = sorted(
                (r for r in rows if r["config_id"] == config), key=lambda r: r["capture_id"]
            )
            if [r["capture_id"] for r in observed] != list(range(len(features))):
                raise ValueError("Complete frame/config pairs required")
            values = np.array([quality(r) for r in observed])
            estimates = np.array([predict(model["quality"], f)[0] for f in features])
            original = [r for r in development if r["config_id"] == config]
            models[config] = dict(
                development_mean_quality=float(np.mean([quality(r) for r in original])),
                observed_mean_quality=float(values.mean()),
                predicted_mean_quality=float(estimates.mean()),
                mean_absolute_error=float(np.abs(estimates - values).mean()),
                mean_signed_error=float((estimates - values).mean()),
                development_fp_per_capture=float(
                    np.mean([r["false_positive_components"] for r in original])
                ),
                observed_fp_per_capture=float(
                    np.mean([r["false_positive_components"] for r in observed])
                ),
                observed_captures_with_fp=sum(r["false_positive_components"] > 0 for r in observed),
                captures=len(observed),
            )
        results[flight] = models
    result = dict(
        schema_version="v3_quality_transfer_diagnostic_v1",
        gt_aware=True,
        deployable_policy=False,
        posthoc=True,
        retrained=False,
        flights=results,
        protocol_sha256=sha(root / "protocol.json"),
        diagnostics_sha256=sha(root / "feasibility_diagnostics.json"),
        implementation_sha256=sha(Path(__file__)),
        interpretation="Per-frame transfer and false-component risk; no mission performance claim",
    )
    save(root / "quality_transfer.json", result)
    lines = [
        "# Frozen quality-head transfer and false-component risk",
        "",
        "Evaluator-only post-hoc diagnostics. No head, policy or contract was changed. "
        "Mean per-frame quality is recall times precision; it is not a calibrated probability. "
        "False components accumulate under the mission contract, even if mean precision is high.",
        "",
        "| Flight | Configuration | Predicted quality | Observed quality | MAE | "
        "Dev FP/capture | New FP/capture |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for flight, models in results.items():
        for config, v in models.items():
            lines.append(
                f"| {flight} | {config} | {v['predicted_mean_quality']:.3f} | "
                f"{v['observed_mean_quality']:.3f} | {v['mean_absolute_error']:.3f} | "
                f"{v['development_fp_per_capture']:.3f} | {v['observed_fp_per_capture']:.3f} |"
            )
    lines.extend(
        [
            "",
            "Each model is evaluated on all 60 captures per flight. These counts ignore "
            "paid capture skipping and are not accumulated errors of any deployed router. "
            "A future risk predictor must be trained on permitted development data and "
            "evaluated on another fresh, frozen cohort.",
            "",
        ]
    )
    (root / "QUALITY_TRANSFER.md").write_text("\n".join(lines))
    print(root / "QUALITY_TRANSFER.md")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
