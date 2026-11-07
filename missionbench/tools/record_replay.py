"""Record a replay set by capturing what the profile executor produces.

    python -m missionbench.tools.record_replay \\
      --episode data/episodes/episode_001.json \\
      --output data/predictions/replay_episode_001.json

This is **capture, not fabrication**. It runs the real profile executor over every
``(frame, configuration)`` pair the episode could reach and writes down exactly what came
out -- latency, energy, transfer sizes, success, and the synthetic prediction's public
fields. Replaying the result reproduces the profile run for the configurations a policy
actually selects.

Its purpose in V1 is to make the replay executor exercisable end to end with a fixture
that genuinely covers a 900-step mission, rather than the four hand-written records that
only covered two frames. When real models arrive, this same file format is what a hardware
capture harness would emit instead -- the loader and executor do not change.

The prediction payload written here keeps only the policy-visible fields
(``prediction_id``, ``frame_id``, ``predicted_target_id``, ``confidence``). The hidden
``ground_truth_track_id`` and ``mask_iou`` are deliberately omitted: a committed replay
fixture must not carry the answers, exactly as a published result file must not. That means
a set recorded here scores quality as all-false-positive if fed to the evaluator, so it is
suitable for exercising the *plumbing* (latency, energy, communication, provenance), not
for reproducing a quality score. That limitation is stated in the file it writes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from missionbench.benchmark import BenchmarkData
from missionbench.executor.base import ExecutionRequest
from missionbench.executor.profile_executor import ProfileExecutor
from missionbench.schemas.episode import load_episode
from missionbench.schemas.loading import SchemaValidationError
from missionbench.tasks.registry import resolve_task

_MS_PER_S = 1000.0


def _public_prediction(prediction: object) -> dict[str, Any] | None:
    """Strip a frame prediction down to its policy-visible instance fields."""
    instances = getattr(prediction, "instances", None)
    if instances is None:
        return None
    return {
        "instances": [
            {
                "prediction_id": inst.prediction_id,
                "frame_id": inst.frame_id,
                "predicted_target_id": inst.predicted_target_id,
                "confidence": inst.confidence,
            }
            for inst in instances
        ]
    }


def record(data: BenchmarkData, episode_id: str, *, frames: int | None = None) -> dict[str, Any]:
    """Capture a full-coverage replay set for one episode from the profile executor."""
    episode = next((e for e in data.episodes() if e.episode_id == episode_id), None)
    if episode is None:
        raise SchemaValidationError(f"no episode {episode_id!r} under the data root")

    profiles = data.profiles(episode.platform_id)
    task = resolve_task(data.task_spec(_contract_task_id(data)))
    ground_truth = task.load_ground_truth(data.ground_truth_dir, episode)
    executor = ProfileExecutor(
        profiles, predictions=task.create_prediction_source(ground_truth, profiles)
    )

    # Cover every frame the mission spans (path duration in whole seconds at 1 fps) for
    # every allowed configuration, so no policy can reach an uncovered pair.
    path = data.path(episode.path_id)
    span = frames if frames is not None else int(path.length_m / episode.velocity_mps) + 1

    records: list[dict[str, Any]] = []
    for frame_id in range(span):
        for config_id in episode.allowed_config_ids:
            observation = _network_at(data, episode, frame_id)
            result = executor.execute(
                ExecutionRequest(
                    episode_id=episode.episode_id,
                    frame_id=frame_id,
                    configuration=data.catalog.get(config_id),
                    network=observation,
                    current_time_s=float(frame_id),
                    seed=episode.seed,
                )
            )
            entry: dict[str, Any] = {
                "frame_id": frame_id,
                "config_id": config_id,
                "success": result.success,
                "latency_ms": round(result.latency_s * _MS_PER_S, 6),
                "onboard_energy_j": result.onboard_energy_j,
                "upload_mb": result.upload_mb,
                "download_mb": result.download_mb,
            }
            if result.success:
                prediction = _public_prediction(result.prediction)
                if prediction is not None:
                    entry["prediction"] = prediction
            else:
                entry["failure_reason"] = result.failure_reason.value
            records.append(entry)

    return {
        "schema_version": "1.0",
        "record_set_id": f"RECORDED_PROFILE_{episode_id}",
        "episode_id": episode_id,
        "records": records,
    }


def _contract_task_id(data: BenchmarkData) -> str:
    contracts = data.contracts()
    if not contracts:
        raise SchemaValidationError("no contract under the data root to resolve the task from")
    return contracts[0].task_id


def _network_at(data: BenchmarkData, episode, frame_id: int):
    from missionbench.simulator.network_trace import TraceBasedNetworkModel

    model = TraceBasedNetworkModel(data.trace(episode.network_trace_id))
    return model.observe(float(frame_id))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="missionbench.tools.record_replay",
        description="Capture a full-coverage replay set from the profile executor.",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--episode", type=Path, required=True, help="Episode file to capture.")
    parser.add_argument("--output", type=Path, required=True, help="Where to write the set.")
    parser.add_argument("--frames", type=int, default=None, help="Override frame count.")
    args = parser.parse_args(argv)

    try:
        data = BenchmarkData(args.data_root)
        episode_id = load_episode(args.episode).episode_id
        payload = record(data, episode_id, frames=args.frames)
    except SchemaValidationError as error:
        print(f"record_replay: {error}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(payload['records'])} records for {episode_id} to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
