"""Post-hoc orientation diagnosis on four known lake failure images.

These GT-selected diagnostic images cannot estimate population quality. Models
receive whole rotated RGB images, never GT crops. No pool promotion is implied.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from missionbench.v2.evaluation import MissionEvaluator
from missionbench.v2.executors import build_executors
from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.lake_missions.run import freeze
from experiments.lake_missions.scenario import load
from experiments.v4_benchmark.backends import ReplayBackend
from experiments.v4_benchmark.probe_cache import rgb_key
from experiments.v4_benchmark.runner import evaluation_frame, validate_result


def restore_mask(mask, quarter_turns, original_shape):
    restored = np.rot90(mask, -quarter_turns).copy()
    if restored.dtype != np.bool_ or restored.shape != original_shape:
        raise ValueError("Inverse rotation must preserve original boolean mask coordinates")
    return restored


def run(source, output):
    import torch

    freeze(source)
    suite = read(source / "suite.json")
    scenario = load(source / "scenario.json", suite["environments"]["day"], Path(suite["assets"]))
    base = next(s for s in scenario.executors if s.config_id == "CFG_MASKRCNN_1280")
    specs = [
        replace(
            base,
            config_id=f"ROTATION_PARITY_{k}",
            parameters={
                **base.parameters,
                "input_width_px": 960 if k else 1280,
                "input_height_px": 1280 if k else 960,
            },
        )
        for k in (0, 1)
    ]
    output.mkdir(parents=True, exist_ok=False)
    save(
        output / "protocol.json",
        dict(
            purpose="post-hoc failure diagnosis; not representative screening or policy evaluation",
            source=str(source.resolve()),
            source_lock_sha256=sha(source / "lock.json"),
            implementation_sha256=sha(Path(__file__)),
            capture_indices=[121, 144],
            environments=["day", "dusk_haze"],
            selection="GT-area-selected failure examples from the earlier visibility audit",
            rotations_degrees_ccw=[0, 90, 180, 270],
            model_input="entire RGB image, rotated; no GT crops, coordinates or masks",
            configurations=[asdict(s) for s in specs],
            hypothesis="small overhead people may fail due to image-plane orientation",
            host_threads=4,
            energy_measured=False,
            mission_executed=False,
        ),
    )
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    executors = build_executors(specs)
    inference = []
    # Finish all RGB inference before evaluator access to object IDs.
    for environment in ("day", "dusk_haze"):
        scenario = load(
            source / "scenario.json", suite["environments"][environment], Path(suite["assets"])
        )
        folder = output / environment
        folder.mkdir()
        backend = ReplayBackend(scenario, source / environment / "recording")
        try:
            for index in (121, 144):
                timestamp = sorted(backend.frames)[index]
                backend.advance_to(timestamp)
                frame = backend.capture()
                for k in range(4):
                    started = time.perf_counter()
                    rotated = np.rot90(frame.rgb, k).copy()
                    executor = executors[specs[k % 2].config_id]
                    execution = executor.run(rotated)
                    validate_result(execution, executor.config_id, rotated.shape[:2])
                    if not execution.success:
                        raise RuntimeError("Rotation inference failed")
                    mask = restore_mask(execution.prediction_mask, k, frame.rgb.shape[:2])
                    elapsed = time.perf_counter() - started
                    path = folder / f"{index:04d}_rotation_{k * 90}.npz"
                    np.savez_compressed(path, prediction_mask=mask)
                    inference.append(
                        dict(
                            environment=environment,
                            capture_id=index,
                            quarter_turns=k,
                            rgb_sha256=rgb_key(frame.rgb),
                            timestamp_ns=timestamp,
                            file=str(path.relative_to(output)),
                            sha256=sha(path),
                            full_rotation_call_s=elapsed,
                            execution=execution.to_summary(),
                            recording_sha256=backend.provenance["recording_sha256"],
                        )
                    )
                    save(output / "progress.json", inference)
                    print(
                        f"{environment} frame {index}, rotation {k * 90}: {elapsed:.2f}s",
                        flush=True,
                    )
        finally:
            backend.close()
    save(output / "inference.json", inference)
    outcomes = []
    for environment in ("day", "dusk_haze"):
        scenario = load(
            source / "scenario.json", suite["environments"][environment], Path(suite["assets"])
        )
        backend = ReplayBackend(scenario, source / environment / "recording")
        scorer = MissionEvaluator(
            scenario.matching_iou_threshold, len(scenario.targets), scenario.observation_interval_s
        )
        try:
            for index in (121, 144):
                timestamp = sorted(backend.frames)[index]
                backend.advance_to(timestamp)
                frame = backend.capture()
                truth = evaluation_frame(
                    frame, index, (timestamp - scenario.schedule.epoch_ns) / 1e9, scenario
                )
                for row in inference:
                    if row["environment"] != environment or row["capture_id"] != index:
                        continue
                    assert row["timestamp_ns"] == timestamp and row["rgb_sha256"] == rgb_key(
                        frame.rgb
                    )
                    assert row["recording_sha256"] == backend.provenance["recording_sha256"]
                    path = output / row["file"]
                    assert sha(path) == row["sha256"]
                    with np.load(path, allow_pickle=False) as data:
                        mask = data["prediction_mask"].copy()
                    outcomes.append(
                        dict(
                            environment=environment,
                            capture_id=index,
                            rotation_degrees_ccw=row["quarter_turns"] * 90,
                            **scorer.update(truth, mask).to_dict(),
                        )
                    )
        finally:
            backend.close()
    save(output / "outcomes.json", outcomes)
    save(
        output / "completion.json",
        dict(
            complete=True,
            actual_calls=len(inference),
            protocol_sha256=sha(output / "protocol.json"),
            inference_sha256=sha(output / "inference.json"),
            outcomes_sha256=sha(output / "outcomes.json"),
            scope="four selected diagnostic images; no generalization or mission success claim",
        ),
    )
    lines = [
        "# Orientation failure diagnosis",
        "",
        "Four GT-selected images; whole-image RGB inference. This is post-hoc diagnosis, "
        "not representative evaluation. Matching IoU and score thresholds are unchanged.",
        "",
        "| Environment | Capture | CCW degrees | Matched target IDs | False components |",
        "|---|---:|---:|---|---:|",
    ]
    for r in outcomes:
        lines.append(
            f"| {r['environment']} | {r['capture_id']} | {r['rotation_degrees_ccw']} | "
            f"{', '.join(r['matched_target_ids']) or 'none'} | {r['false_positive_components']} |"
        )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.source, args.output)
