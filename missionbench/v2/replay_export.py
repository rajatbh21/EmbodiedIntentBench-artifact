"""Export one V2 mission as a self-contained replay bundle for the static HTML viewer.

The exporter runs a scenario/policy pair through the ordinary :class:`MissionRunner`
(with runtime snapshots enabled), then writes a directory a viewer can consume without
any live Python objects::

    replay_bundle/
    ├── manifest.json    # identity, contract, configs, map geometry, honesty labels
    ├── events.json      # ordered observation timeline + snapshots + constraint status
    ├── frames/          # per-observation RGB / prediction / debug-GT overlay PNGs
    ├── overview.png     # world overview (trajectory + objects)
    └── index.html       # the self-contained viewer (replay data embedded)

Boundaries that must hold:

- **Single source of truth.** Every frame is re-rendered through the mission's own
  ``CameraRenderer``/executor (deterministic; position is a pure function of mission
  time), and every score shown is the ``MissionEvaluator``'s own per-observation
  output. Nothing here re-implements matching or success evaluation; the final
  ``mission_success`` comes verbatim from the mission result, and the exporter
  *verifies* its cumulative tallies against it rather than recomputing them.
- **GT stays labelled.** Ground-truth overlays and true object positions are exported
  under explicit ``debug_gt`` keys and the viewer shows them only behind a
  "Debug GT" toggle — they are evaluator-side data, never policy-visible.
- **Strict JSON.** ``allow_nan=False``; output is deterministic for identical inputs
  except the documented ``execution.measured_wall_clock_s`` diagnostic.

Constraint status is a *replay-time presentation layer*: SAFE / AT_RISK / VIOLATED /
NOT_APPLICABLE / UNKNOWN per hard constraint, using the mission's existing semantics
plus simple, explicit at-risk margins. The final event's statuses are derived from the
evaluator's own constraint booleans — the viewer never runs a second success evaluator.

Privacy (V3 P1): the fifth constraint is active and aligned with V1. It is reported
``NOT_APPLICABLE`` only for scenarios that declare no remote execution path (nothing
can violate it there); scenarios with a ``simulated_remote`` configuration report
SAFE/VIOLATED from the mission's own privacy-violation record.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Final

from missionbench.schemas.contract import Contract
from missionbench.schemas.loading import SchemaValidationError, SchemaVersionError
from missionbench.v2.runner import MissionRunner, ObservationLog, V2MissionResult
from missionbench.v2.scenario import V2Scenario, load_scenario

__all__ = [
    "REPLAY_SCHEMA_VERSION",
    "STATUS_AT_RISK",
    "STATUS_NOT_APPLICABLE",
    "STATUS_SAFE",
    "STATUS_UNKNOWN",
    "STATUS_VIOLATED",
    "battery_status",
    "communication_status",
    "deadline_status",
    "export_replay_bundle",
    "main",
    "overall_status",
    "quality_status",
]

#: Version of the replay bundle contract (manifest + events + frame layout).
REPLAY_SCHEMA_VERSION: Final = "1.0"

STATUS_SAFE: Final = "SAFE"
STATUS_AT_RISK: Final = "AT_RISK"
STATUS_VIOLATED: Final = "VIOLATED"
STATUS_NOT_APPLICABLE: Final = "NOT_APPLICABLE"
STATUS_UNKNOWN: Final = "UNKNOWN"

#: At-risk margins: conservative, explicit, and deliberately simple — no prediction of
#: future failure. Deadline: less than this fraction of the deadline remains. Battery:
#: within this many battery-fraction points of the contract floor. Communication: this
#: fraction of the budget already spent.
DEADLINE_AT_RISK_FRACTION: Final = 0.2
BATTERY_AT_RISK_MARGIN_FRAC: Final = 0.1
COMMUNICATION_AT_RISK_FRACTION: Final = 0.8

_PREDICTION_RGBA: Final = (255, 80, 80, 170)
_GT_TARGET_RGBA: Final = (0, 220, 60, 170)
_GT_DISTRACTOR_RGBA: Final = (250, 210, 0, 170)

_PRIVACY_NOTE: Final = (
    "this scenario declares no remote execution path, so no action can violate the "
    "privacy constraint (V3 P1 activates it for scenarios with simulated_remote configs)"
)


def _privacy_applicable(scenario: V2Scenario) -> bool:
    """Privacy is exercised only where a remote execution path exists to violate it."""
    return any(spec.kind == "simulated_remote" for spec in scenario.executor_configs)


# --- constraint status (replay-time presentation over the existing semantics) ----------------


def deadline_status(remaining_s: float, deadline_s: float) -> str:
    if remaining_s < 0.0:
        return STATUS_VIOLATED
    if remaining_s <= DEADLINE_AT_RISK_FRACTION * deadline_s:
        return STATUS_AT_RISK
    return STATUS_SAFE


def battery_status(battery_frac: float, min_final_battery_frac: float) -> str:
    margin = battery_frac - min_final_battery_frac
    if margin < 0.0:
        return STATUS_VIOLATED
    if margin <= BATTERY_AT_RISK_MARGIN_FRAC:
        return STATUS_AT_RISK
    return STATUS_SAFE


def communication_status(used_mb: float, budget_mb: float) -> str:
    if used_mb > budget_mb:
        return STATUS_VIOLATED
    if budget_mb > 0.0 and used_mb >= COMMUNICATION_AT_RISK_FRACTION * budget_mb:
        return STATUS_AT_RISK
    return STATUS_SAFE


def quality_status(contract: Contract, interim_value: float | None) -> str:
    """Interim quality standing. Quality is only *judged* at mission end (existing
    semantics), so an unsatisfied interim value is AT_RISK — progress, not a verdict."""
    if interim_value is None:
        return STATUS_UNKNOWN
    return STATUS_SAFE if contract.quality_satisfied(interim_value) else STATUS_AT_RISK


def overall_status(statuses: dict[str, str]) -> str:
    """VIOLATED > AT_RISK > UNKNOWN > SAFE; NOT_APPLICABLE constraints are ignored."""
    considered = [s for s in statuses.values() if s != STATUS_NOT_APPLICABLE]
    for level in (STATUS_VIOLATED, STATUS_AT_RISK, STATUS_UNKNOWN):
        if level in considered:
            return level
    return STATUS_SAFE


def _final_constraint_status(
    result: V2MissionResult, *, privacy_applicable: bool
) -> dict[str, Any]:
    """The final statuses, derived from the evaluator's own constraint booleans."""
    mapping = {
        "quality": result.constraints["quality_success"],
        "deadline": result.constraints["deadline_success"],
        "battery": result.constraints["battery_constraint_success"],
        "communication": result.constraints["communication_constraint_success"],
    }
    statuses = {
        name: (STATUS_SAFE if satisfied else STATUS_VIOLATED) for name, satisfied in mapping.items()
    }
    if privacy_applicable:
        statuses["privacy"] = (
            STATUS_SAFE
            if result.constraints.get("privacy_constraint_success", True)
            else STATUS_VIOLATED
        )
    else:
        # No remote execution path exists in this scenario, so no action can violate
        # privacy -- the constraint is not exercised, not merely satisfied.
        statuses["privacy"] = STATUS_NOT_APPLICABLE
    statuses["overall"] = STATUS_SAFE if result.mission_success else STATUS_VIOLATED
    return statuses


# --- export ----------------------------------------------------------------------------------


def export_replay_bundle(scenario_path: Path, policy_name: str, output_dir: Path) -> Path:
    """Run one mission and write a self-contained replay bundle to ``output_dir``."""
    from PIL import Image

    from missionbench.v2.visualize import render_overview

    scenario = load_scenario(scenario_path)
    runner = MissionRunner(scenario, policy_name, record_runtime_snapshots=True)
    result = runner.run()

    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    render_overview(
        scenario, runner.world, runner.trajectory, runner.objects, output_dir / "overview.png"
    )
    background = _write_map_background(runner, scenario, output_dir, Image)

    events: list[dict[str, Any]] = []
    found_ids: set[str] = set()
    matched_detections = 0
    total_predictions = 0
    false_positives = 0
    for index, log in enumerate(result.observations):
        frame_refs = _write_frames(runner, log, frames_dir, Image)
        events.append(_build_event(scenario, log, index, frame_refs, found_ids, result))
        matched_detections += log.score["matched_components"]
        total_predictions += log.score["predicted_components"]
        false_positives += log.score["false_positive_components"]

    _verify_against_evaluator(
        result, found_ids, matched_detections, total_predictions, false_positives
    )

    interval = scenario.simulation.observation_interval_s
    skipped = [
        {
            "observation_id": obs_id,
            "capture_time_s": obs_id * interval,
            "position_m": list(runner.trajectory.position_at(obs_id * interval)),
        }
        for obs_id in result.skipped_observation_ids
    ]

    events_doc = {
        "replay_schema_version": REPLAY_SCHEMA_VERSION,
        "scenario_id": result.scenario_id,
        "policy": result.policy_name,
        "events": events,
        "skipped_observations": skipped,
        "final": {
            "mission_success": result.mission_success,
            "constraints": dict(result.constraints),
            "constraint_status": _final_constraint_status(
                result, privacy_applicable=_privacy_applicable(scenario)
            ),
            "termination_reason": result.termination_reason,
            "final_time_s": result.final_time_s,
            "final_battery_frac": result.final_battery_frac,
            "cumulative_communication_mb": result.cumulative_communication_mb,
            "cumulative_energy_j": result.cumulative_energy_j,
            "path_progress": result.path_progress,
            "processed_observation_count": result.processed_observation_count,
            "skipped_observation_count": result.skipped_observation_count,
            "mean_executor_latency_s": result.mean_executor_latency_s,
            "config_selection_history": list(result.config_selection_history),
            "quality": dict(result.quality),
            "notes": dict(result.notes),
        },
    }
    manifest = _build_manifest(
        scenario, scenario_path, result, len(events), len(skipped), runner, background
    )

    _dump_json(output_dir / "events.json", events_doc)
    _dump_json(output_dir / "manifest.json", manifest)

    from missionbench.v2.replay_viewer import build_viewer_html

    (output_dir / "index.html").write_text(
        build_viewer_html({"manifest": manifest, "replay": events_doc}), encoding="utf-8"
    )
    return output_dir


def _build_event(
    scenario: V2Scenario,
    log: ObservationLog,
    index: int,
    frame_refs: dict[str, str],
    found_ids: set[str],
    result: V2MissionResult,
) -> dict[str, Any]:
    contract = scenario.contract
    runtime = log.runtime
    if runtime is None:  # pragma: no cover - the exporter always enables snapshots
        raise RuntimeError("replay export requires runtime snapshots; none were recorded")
    after = runtime["after_completion"]

    newly_found = sorted(set(log.score["matched_target_ids"]) - found_ids)
    found_ids.update(log.score["matched_target_ids"])
    total_targets = int(result.quality["total_unique_targets"])
    interim_recall = 1.0 if total_targets == 0 else len(found_ids) / total_targets

    interim_value = {
        "target_recall": interim_recall,
        # detection_precision is cumulative over the evaluator's own per-observation
        # tallies; recomputed as a running ratio for display only.
    }.get(contract.quality_metric)

    privacy_applicable = _privacy_applicable(scenario)
    violations_so_far = int(runtime.get("privacy_violations_so_far", 0))
    if not privacy_applicable:
        privacy = STATUS_NOT_APPLICABLE
    else:
        privacy = STATUS_VIOLATED if violations_so_far > 0 else STATUS_SAFE
    statuses = {
        "quality": quality_status(contract, interim_value),
        "deadline": deadline_status(after["remaining_deadline_s"], contract.deadline_s),
        "battery": battery_status(after["battery_frac"], contract.min_final_battery_frac),
        "communication": communication_status(
            after["cumulative_communication_mb"], contract.communication_budget_mb
        ),
        "privacy": privacy,
    }
    statuses["overall"] = overall_status(statuses)

    constraint_status = {
        "quality": {
            "status": statuses["quality"],
            "metric": contract.quality_metric,
            "interim_value": interim_value,
            "threshold": contract.quality_threshold,
            "operator": contract.quality_operator.value,
        },
        "deadline": {
            "status": statuses["deadline"],
            "elapsed_s": log.completion_time_s,
            "deadline_s": contract.deadline_s,
            "remaining_s": after["remaining_deadline_s"],
        },
        "battery": {
            "status": statuses["battery"],
            "battery_frac": after["battery_frac"],
            "min_final_battery_frac": contract.min_final_battery_frac,
            "margin_frac": after["battery_frac"] - contract.min_final_battery_frac,
        },
        "communication": {
            "status": statuses["communication"],
            "used_mb": after["cumulative_communication_mb"],
            "budget_mb": contract.communication_budget_mb,
            "remaining_mb": contract.communication_budget_mb - after["cumulative_communication_mb"],
        },
        "privacy": {
            "status": statuses["privacy"],
            "privacy_level": contract.privacy_level.value,
            "violations_so_far": violations_so_far,
            "reason": "" if privacy_applicable else _PRIVACY_NOTE,
        },
        "overall": {"status": statuses["overall"]},
    }

    return {
        "event_index": index,
        "observation_id": log.observation_id,
        "capture_time_s": log.capture_time_s,
        "completion_time_s": log.completion_time_s,
        "capture_position_m": list(log.capture_position_m),
        "completion_position_m": list(log.completion_position_m),
        "requested_config_id": log.requested_config_id,
        "executed_config_id": log.executed_config_id,
        "action_valid": log.action_valid,
        "config_switched": runtime["config_switched"],
        "fallback_used": runtime["fallback_used"],
        "skipped_after": list(log.skipped_after),
        "runtime": runtime,
        "execution": log.execution,
        "score": log.score,
        "cumulative_quality": {
            "unique_targets_found": len(found_ids),
            "total_unique_targets": total_targets,
            "target_recall": interim_recall,
        },
        "newly_found_target_ids": newly_found,
        "constraint_status": constraint_status,
        "frames": frame_refs,
    }


#: Longest edge of the pre-rendered mission-map background, in pixels.
_MAP_BACKGROUND_MAX_PX: Final = 1600


def _write_map_background(
    runner: MissionRunner, scenario: V2Scenario, output_dir: Path, image_module: Any
) -> dict[str, Any]:
    """Pre-render the mission area from the mission's own world source.

    This is the viewer's map background: the real aerial raster (or test image) the
    mission flew over, read through the same ``read_window_m`` the camera uses, so the
    trajectory overlays register exactly. The extent comes from the trajectory alone,
    padded by one camera footprint — never from ground-truth object placement. The
    bundle stays self-contained and offline: no tiles, no external map service.
    """
    xs = [p[0] for p in runner.trajectory.waypoints_m]
    ys = [p[1] for p in runner.trajectory.waypoints_m]
    pad_x = scenario.camera.footprint_width_m
    pad_y = scenario.camera.footprint_height_m
    x0, x1 = min(xs) - pad_x, max(xs) + pad_x
    y0, y1 = min(ys) - pad_y, max(ys) + pad_y
    width_m, height_m = x1 - x0, y1 - y0
    scale = min(_MAP_BACKGROUND_MAX_PX / width_m, _MAP_BACKGROUND_MAX_PX / height_m)
    out_w = max(64, round(width_m * scale))
    out_h = max(64, round(height_m * scale))
    read = runner.world.read_window_m(
        ((x0 + x1) / 2.0, (y0 + y1) / 2.0), width_m, height_m, out_w, out_h
    )
    image_module.fromarray(read.rgb).save(output_dir / "map_background.png")
    return {
        "file": "map_background.png",
        "extent_m": [x0, y0, x1, y1],
        "width_px": out_w,
        "height_px": out_h,
    }


def _write_frames(
    runner: MissionRunner, log: ObservationLog, frames_dir: Path, image_module: Any
) -> dict[str, str]:
    """Re-render one observation deterministically and write its three frame PNGs.

    The RGB and ground truth come from the mission's own renderer at the logged capture
    time; the prediction comes from the mission's own executor for the logged config —
    the exact inputs the evaluator scored.
    """
    import numpy as np

    observation = runner.render_at(log.capture_time_s, log.observation_id)
    executor = runner.executor_for(log.executed_config_id)
    if hasattr(executor, "run_with_context"):
        prediction = executor.run_with_context(
            observation.rgb,
            runner.execution_context_at(log.capture_time_s, log.observation_id),
        )
    else:
        prediction = executor.run(observation.rgb)

    h, w = observation.rgb.shape[:2]
    pred_overlay = np.zeros((h, w, 4), dtype=np.uint8)
    pred_overlay[prediction.prediction_mask] = _PREDICTION_RGBA
    gt_overlay = np.zeros((h, w, 4), dtype=np.uint8)
    gt_overlay[observation.semantic_gt == 1] = _GT_TARGET_RGBA
    gt_overlay[observation.semantic_gt == 2] = _GT_DISTRACTOR_RGBA

    stem = f"obs_{log.observation_id:06d}"
    refs = {
        "rgb": f"frames/{stem}_rgb.png",
        "prediction": f"frames/{stem}_pred.png",
        "gt_debug": f"frames/{stem}_gt.png",
    }
    image_module.fromarray(observation.rgb).save(frames_dir / f"{stem}_rgb.png")
    image_module.fromarray(pred_overlay).save(frames_dir / f"{stem}_pred.png")
    image_module.fromarray(gt_overlay).save(frames_dir / f"{stem}_gt.png")
    return refs


def _verify_against_evaluator(
    result: V2MissionResult,
    found_ids: set[str],
    matched_detections: int,
    total_predictions: int,
    false_positives: int,
) -> None:
    """The exporter's tallies must reproduce the evaluator's — never diverge silently."""
    quality = result.quality
    checks = [
        ("unique_targets_found", len(found_ids), quality["unique_targets_found"]),
        ("matched_detections", matched_detections, quality["matched_detections"]),
        ("total_predictions", total_predictions, quality["total_predictions"]),
        ("false_positive_detections", false_positives, quality["false_positive_detections"]),
    ]
    for name, derived, evaluator_value in checks:
        if derived != evaluator_value:
            raise RuntimeError(
                f"replay export tally {name}={derived} contradicts the evaluator's "
                f"{evaluator_value}; the bundle would misrepresent the mission"
            )
    total = int(quality["total_unique_targets"])
    derived_recall = 1.0 if total == 0 else len(found_ids) / total
    if not math.isclose(derived_recall, float(quality["target_recall"]), abs_tol=1e-9):
        raise RuntimeError(
            f"replay export target_recall {derived_recall} contradicts the evaluator's "
            f"{quality['target_recall']}"
        )


def _build_manifest(
    scenario: V2Scenario,
    scenario_path: Path,
    result: V2MissionResult,
    event_count: int,
    skipped_count: int,
    runner: MissionRunner,
    background: dict[str, Any],
) -> dict[str, Any]:
    contract = scenario.contract
    found = {
        target_id for log in result.observations for target_id in log.score["matched_target_ids"]
    }
    return {
        "replay_schema_version": REPLAY_SCHEMA_VERSION,
        "generator": "missionbench.v2.replay_export",
        "scenario_id": scenario.scenario_id,
        "scenario_path": scenario_path.name,
        "policy": result.policy_name,
        "mission_id": f"{scenario.scenario_id}__{result.policy_name}",
        "evaluation_purpose": scenario.evaluation_purpose,
        "contract": {
            "contract_id": contract.contract_id,
            "task_id": contract.task_id,
            "quality_metric": contract.quality_metric,
            "quality_operator": contract.quality_operator.value,
            "quality_threshold": contract.quality_threshold,
            "deadline_s": contract.deadline_s,
            "communication_budget_mb": contract.communication_budget_mb,
            "min_final_battery_frac": contract.min_final_battery_frac,
            "privacy_level": contract.privacy_level.value,
        },
        "executor_configs": [
            {
                "config_id": spec.config_id,
                "model_strategy_id": spec.model_strategy_id,
                "kind": spec.kind,
                "mission_latency_s": spec.mission_latency_s,
                "energy_j_per_call": spec.energy_j_per_call,
                "communication_mb_per_call": spec.communication_mb_per_call,
                "quality_tier": spec.quality_tier,
            }
            for spec in scenario.executor_configs
        ],
        "map": {
            "waypoints_m": [list(p) for p in runner.trajectory.waypoints_m],
            "footprint_width_m": scenario.camera.footprint_width_m,
            "footprint_height_m": scenario.camera.footprint_height_m,
            "meters_per_pixel": scenario.world.meters_per_pixel,
            "observation_interval_s": scenario.simulation.observation_interval_s,
            # Added after the initial 1.0 bundles, before any release: the viewer's
            # registration frame and mission-time interpolation basis. All additive;
            # the viewer falls back to schematic drawing / discrete stepping if absent.
            "background": background,
            "drone_speed_mps": scenario.drone.speed_mps,
            "trajectory_duration_s": runner.trajectory.duration_s,
        },
        "matching_iou_threshold": scenario.simulation.matching_iou_threshold,
        "debug_gt": {
            "label": ("Debug GT — evaluator-side ground truth; never visible to the policy"),
            "targets": [
                {
                    "object_id": obj.object_id,
                    "position_m": list(obj.position_m),
                    "found": obj.object_id in found,
                }
                for obj in scenario.targets
            ],
            "distractors": [
                {"object_id": obj.object_id, "position_m": list(obj.position_m)}
                for obj in scenario.distractors
            ],
        },
        "files": {
            "events": "events.json",
            "overview": "overview.png",
            "map_background": "map_background.png",
            "viewer": "index.html",
            "frames_dir": "frames",
        },
        "counts": {"events": event_count, "skipped_observations": skipped_count},
        "honesty": {
            "notes": dict(result.notes),
            "privacy": _PRIVACY_NOTE,
            "nondeterminism": (
                "execution.measured_wall_clock_s is a measured diagnostic and varies "
                "between runs; all other bundle content is deterministic for identical "
                "inputs"
            ),
        },
    }


def _dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


# --- CLI -------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="missionbench.v2.replay_export",
        description=(
            "Run one V2 mission and export a self-contained replay bundle "
            "(deterministic re-run; the viewer depends only on the bundle)."
        ),
    )
    parser.add_argument("--scenario", type=Path, required=True, help="V2 scenario JSON file.")
    parser.add_argument("--policy", default="rule_based", help="Policy name (as v2.cli run).")
    parser.add_argument("--output", type=Path, required=True, help="Bundle output directory.")
    args = parser.parse_args(argv)

    try:
        bundle = export_replay_bundle(args.scenario, args.policy, args.output)
    except (SchemaValidationError, SchemaVersionError) as error:
        print(f"missionbench.v2.replay_export: {error}", file=sys.stderr)
        return 2
    print(
        f"wrote replay bundle {bundle}\n"
        f"open {bundle / 'index.html'} directly in a browser, or serve it:\n"
        f"  python -m http.server --directory {bundle} 8000"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
