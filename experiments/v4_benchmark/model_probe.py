"""Dense real-model execution probe; measured latency is NOT mission timing.

Runs existing executors on recorded RGB, keeping evaluator-only labels separate.
This is a model integration diagnostic on synthetic scenes, not human perception
validation or calibration from held-out evaluation data.
"""

from __future__ import annotations

import argparse
import hashlib
import platform
from pathlib import Path

import numpy as np

from missionbench.v2.evaluation import MissionEvaluator
from missionbench.v2.executors import build_executors
from experiments.v4_benchmark.backends import ReplayBackend, save_json
from experiments.v4_benchmark.runner import evaluation_frame, validate_result
from experiments.v4_benchmark.scenario import digest, load_scenario


def probe(
    scenario_path: Path, recording: Path, output: Path, *, max_frames: int = 4, threads: int = 2
) -> dict:
    if max_frames < 1 or threads < 1:
        raise ValueError("frame count and thread count must be positive")
    scenario = load_scenario(scenario_path)
    if any(
        s.kind
        not in (
            "torch_semantic_segmentation",
            "torch_instance_segmentation",
            "torch_cabinet_segmentation",
        )
        for s in scenario.executors
    ):
        raise ValueError("probe requires a local real-model-only pool")
    backend = ReplayBackend(scenario, recording)
    import torch
    import torchvision

    torch.set_num_threads(threads)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    captures = []
    evaluators = {
        spec.config_id: MissionEvaluator(
            scenario.matching_iou_threshold, len(scenario.targets), scenario.observation_interval_s
        )
        for spec in scenario.executors
    }
    try:
        executors = build_executors(scenario.executors)
        for index, timestamp in enumerate(
            range(
                scenario.schedule.epoch_ns, scenario.schedule.end_ns, scenario.schedule.interval_ns
            )
        ):
            if index >= max_frames:
                break
            backend.advance_to(timestamp)
            frame = backend.capture()
            observation = evaluation_frame(
                frame, index, (timestamp - scenario.schedule.epoch_ns) / 1e9, scenario
            )
            captures.append(
                {
                    "observation_id": index,
                    "timestamp_ns": timestamp,
                    "targets": target_visibility(frame.object_ids, scenario.targets),
                }
            )
            for config_id, executor in executors.items():
                result = executor.run(frame.rgb.copy())
                validate_result(result, config_id, frame.rgb.shape[:2])
                if not result.success:
                    raise RuntimeError(f"probe executor failed: {config_id}")
                score = evaluators[config_id].update(observation, result.prediction_mask)
                rows.append(
                    {
                        "observation_id": index,
                        "execution": result.to_summary(),
                        "score": score.to_dict(),
                    }
                )
                np.savez_compressed(
                    output / f"prediction_{index:05d}_{len(rows):05d}.npz",
                    prediction_mask=result.prediction_mask,
                )
                prediction = output / f"prediction_{index:05d}_{len(rows):05d}.npz"
                rows[-1]["prediction_file"] = prediction.name
                rows[-1]["prediction_sha256"] = hashlib.sha256(prediction.read_bytes()).hexdigest()
                save_json(output / "observations.json", rows)
    finally:
        backend.close()
    summaries = {}
    for spec in scenario.executors:
        samples = [
            r["execution"]["measured_wall_clock_s"]
            for r in rows
            if r["execution"]["config_id"] == spec.config_id
        ]
        summaries[spec.config_id] = {
            "sample_count": len(samples),
            "dense_quality": evaluators[spec.config_id].quality_details(
                min(scenario.duration_s, len(captures) * scenario.observation_interval_s)
            ),
            "end_to_end_median_s": float(np.median(samples)),
            "end_to_end_p95_s": float(np.percentile(samples, 95)),
        }
    report = {
        "probe_schema_version": "0.1",
        "evaluation_purpose": "integration_diagnostic",
        "domain_mismatch": "Pretrained person models on synthetic scenes; not human quality.",
        "scope": "Dense observed slots; not a routed mission or population performance estimate.",
        "energy": "configured, not measured",
        "latency": "measured diagnostic only",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torch_threads": torch.get_num_threads(),
        "scenario_sha256": digest(scenario.source),
        "backend": backend.provenance,
        "models": summaries,
        "visibility": {
            "gt_aware": True,
            "scope": "Evaluator-only diagnostic; never supplied to router or executor",
            "captures": captures,
        },
    }
    save_json(output / "probe.json", report)
    return report


def target_visibility(object_ids: np.ndarray, targets: dict[int, str]) -> list[dict]:
    """Keep invisible targets explicit; no invented projection or size adjustment."""
    rows = []
    for label, target_id in sorted(targets.items()):
        ys, xs = np.nonzero(object_ids == label)
        rows.append(
            {
                "target_id": target_id,
                "visible_pixels": len(xs),
                "bbox_xyxy_exclusive": (
                    [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
                    if len(xs)
                    else None
                ),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    result = probe(
        args.scenario, args.recording, args.output, max_frames=args.max_frames, threads=args.threads
    )
    print(result["models"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
