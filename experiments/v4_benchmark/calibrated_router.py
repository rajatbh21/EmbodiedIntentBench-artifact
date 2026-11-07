"""Development-fitted RGB-context profiles for local, resource-aware routing.

Profiles contain aggregate frame-level evidence, not mission GT or a lookup of test
frames. Estimates are empirical proxies, never certificates of mission-level recall.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from missionbench.simulator.action_validator import privacy_permits
from experiments.v4_benchmark.backends import ReplayBackend, save_json
from experiments.v4_benchmark.intent_visual import visual_features
from experiments.v4_benchmark.scenario import digest, finite_tree, load_scenario


def context_key(rgb, altitude_m=None):
    gray, edges = visual_features(rgb)
    if float(gray.std()) < 0.05:
        key = "low_contrast"
    elif edges.size and float(np.mean(edges > 0.15)) > 0.2:
        key = "dense_edges"
    else:
        key = "ordinary"
    if altitude_m is not None:
        key = ("below_6m/" if altitude_m < 6 else "at_least_6m/") + key
    return key


def model_signature(spec):
    return digest(
        {
            "kind": spec.kind,
            "parameters": spec.parameters,
            "model_strategy_id": spec.model_strategy_id,
        }
    )


def aggregate(rows):
    matched = sum(len(r["score"]["matched_target_ids"]) for r in rows)
    visible = sum(len(r["score"]["visible_target_ids"]) for r in rows)
    true_predictions = sum(r["score"]["matched_components"] for r in rows)
    predictions = sum(r["score"]["predicted_components"] for r in rows)
    return {
        "frames": len(rows),
        "scene_count": 1,
        "visible_target_observations": visible,
        "predictions": predictions,
        "visible_frame_recall": matched / visible if visible else 0.0,
        # Empty predictions must never advertise a perfect precision model to the router.
        "detection_precision": true_predictions / predictions if predictions else 0.0,
    }


def fit(scenario_path: Path, recording: Path, probe_dir: Path, output: Path, *, use_altitude=False):
    scenario = load_scenario(scenario_path)
    report = json.loads((probe_dir / "probe.json").read_text())
    rows = json.loads((probe_dir / "observations.json").read_text())
    if report["scenario_sha256"] != digest(scenario.source):
        raise ValueError("probe/scenario mismatch")
    if report["backend"]["recording_identity"] != scenario.recording_identity:
        raise ValueError("probe/recording mismatch")
    expected = {
        (i, s.config_id)
        for i in range(len(report["visibility"]["captures"]))
        for s in scenario.executors
    }
    observed = [(r["observation_id"], r["execution"]["config_id"]) for r in rows]
    if len(observed) != len(set(observed)) or set(observed) != expected or not expected:
        raise ValueError("probe must contain paired nonempty model observations")
    keys = {}
    backend = ReplayBackend(scenario, recording)
    try:
        for frame in report["visibility"]["captures"]:
            backend.advance_to(frame["timestamp_ns"])
            captured = backend.capture()
            keys[frame["observation_id"]] = context_key(
                captured.rgb, captured.body_pose["position_m"][2] if use_altitude else None
            )
    finally:
        backend.close()
    models = {}
    for spec in scenario.executors:
        selected = [r for r in rows if r["execution"]["config_id"] == spec.config_id]
        if any(not r["execution"]["success"] for r in selected):
            raise ValueError("failed model in calibration")
        models[spec.config_id] = {
            "signature": model_signature(spec),
            "measured_latency_p95_s": report["models"][spec.config_id]["end_to_end_p95_s"],
            "global": aggregate(selected),
            "contexts": {
                key: aggregate([r for r in selected if keys[r["observation_id"]] == key])
                for key in sorted(set(keys.values()))
            },
        }
    artifact = {
        "calibration_schema_version": "0.2" if use_altitude else "0.1",
        "feature_encoding": "rgb_altitude_v1" if use_altitude else "rgb_v1",
        "purpose": "development_only",
        "training_recording_identities": [scenario.recording_identity],
        "training_world_hashes": [digest(scenario.world)],
        "training_probe_sha256": digest(report),
        "training_rows_sha256": digest(rows),
        "latency_scope": "Local CPU diagnostic; may include contention; not embedded hardware",
        "matching_iou_threshold": scenario.matching_iou_threshold,
        "quality_semantics": (
            "Visible frame recall and nonempty detection precision; not mission recall guarantees"
        ),
        "models": models,
    }
    if output.exists():
        raise ValueError("refusing to overwrite frozen calibration")
    save_json(output, artifact)
    return artifact


def merge_profiles(paths: list[Path], output: Path):
    """Give each development recording equal weight, regardless of flight length."""
    if not paths or output.exists():
        raise ValueError("nonempty sources and a new output path are required")
    sources = [json.loads(path.read_text()) for path in paths]
    identities = [
        identity for source in sources for identity in source["training_recording_identities"]
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate development recording")
    reference = sources[0]
    for source in sources:
        if source.get("purpose") != "development_only" or source.get(
            "feature_encoding"
        ) != reference.get("feature_encoding"):
            raise ValueError("incompatible calibration sources")
        if source["matching_iou_threshold"] != reference["matching_iou_threshold"] or set(
            source["models"]
        ) != set(reference["models"]):
            raise ValueError("incompatible calibration pool/evaluator")
        for key, model in source["models"].items():
            if model["signature"] != reference["models"][key]["signature"]:
                raise ValueError("incompatible model signatures")

    def combine(stats):
        count = sum(s.get("scene_count", 1) for s in stats)
        return {
            **{
                key: sum(s[key] for s in stats)
                for key in ("frames", "predictions", "visible_target_observations")
            },
            "scene_count": count,
            **{
                key: sum(s[key] * s.get("scene_count", 1) for s in stats) / count
                for key in ("visible_frame_recall", "detection_precision")
            },
        }

    models = {}
    for key in reference["models"]:
        group = [source["models"][key] for source in sources]
        contexts = sorted({context for model in group for context in model["contexts"]})
        models[key] = dict(
            signature=group[0]["signature"],
            measured_latency_p95_s=max(model["measured_latency_p95_s"] for model in group),
            **{"global": combine([model["global"] for model in group])},
            contexts={
                context: combine(
                    [model["contexts"][context] for model in group if context in model["contexts"]]
                )
                for context in contexts
            },
        )
    result = dict(
        calibration_schema_version="0.2",
        purpose="development_only",
        feature_encoding=reference.get("feature_encoding", "rgb_v1"),
        aggregation="equal_recording_weight; not an independence claim",
        training_recording_identities=identities,
        training_world_hashes=sorted(
            {value for source in sources for value in source.get("training_world_hashes", [])}
        ),
        source_profile_hashes=[digest(source) for source in sources],
        matching_iou_threshold=reference["matching_iou_threshold"],
        models=models,
        quality_semantics=reference["quality_semantics"],
        latency_scope="maximum source p95; measured diagnostics, not drone hardware",
    )
    save_json(output, result)
    return result


class CalibratedRouter:
    def __init__(self, *, artifact_path, interval_s, min_context_frames=8, use_context=True):
        raw = json.loads(Path(artifact_path).read_text())
        finite_tree(raw)
        if (
            raw.get("calibration_schema_version") not in ("0.1", "0.2")
            or raw.get("purpose") != "development_only"
        ):
            raise ValueError("unsupported calibration artifact")
        if type(min_context_frames) is not int or min_context_frames < 1:
            raise ValueError("min_context_frames must be positive")
        for model in raw["models"].values():
            for stats in [model["global"], *model["contexts"].values()]:
                for key in ("visible_frame_recall", "detection_precision"):
                    if not 0 <= stats[key] <= 1:
                        raise ValueError("invalid calibration quality")
                if stats["frames"] < 1:
                    raise ValueError("empty calibration")
        self.artifact = raw
        if type(use_context) is not bool:
            raise ValueError("use_context must be boolean")
        self.use_context = use_context
        if raw.get("feature_encoding", "rgb_v1") not in ("rgb_v1", "rgb_altitude_v1"):
            raise ValueError("unsupported feature encoder")
        self.artifact_sha256 = digest(raw)
        self.interval_s = interval_s
        self.min_context_frames = min_context_frames
        self.last_decision = None
        self.costs = None

    def bind_public_mission(
        self,
        *,
        executors,
        duration_s,
        capacity_wh,
        flight_power_w,
        router_latency_s,
        router_energy_j,
        matching_iou_threshold,
    ):
        """Called by the runner with configuration data only; no world or labels."""
        if set(self.artifact["models"]) != {s.config_id for s in executors}:
            raise ValueError("calibration pool mismatch")
        if matching_iou_threshold != self.artifact["matching_iou_threshold"]:
            raise ValueError("calibration evaluator threshold mismatch")
        for spec in executors:
            if spec.kind == "simulated_remote":
                raise ValueError("calibrated router currently supports local models only")
            if self.artifact["models"][spec.config_id]["signature"] != model_signature(spec):
                raise ValueError("calibration model signature mismatch")
        self.costs = {s.config_id: (s.mission_latency_s, s.energy_j_per_call) for s in executors}
        self.duration_s, self.capacity_j, self.flight_power_w = (
            duration_s,
            capacity_wh * 3600,
            flight_power_w,
        )
        self.router_latency_s, self.router_energy_j = router_latency_s, router_energy_j

    def select(self, observation):
        if self.costs is None:
            raise ValueError("calibrated router must bind public mission configuration")
        contract, state = observation.contract, observation.state
        if str(contract.quality_operator) not in (">", ">="):
            raise ValueError("calibrated router supports lower-bound quality only")
        metric = {
            "target_recall": "visible_frame_recall",
            "detection_precision": "detection_precision",
        }.get(contract.quality_metric)
        if metric is None:
            raise ValueError("unsupported quality metric")
        context = (
            context_key(
                observation.rgb,
                observation.position_m[2]
                if self.artifact.get("feature_encoding") == "rgb_altitude_v1"
                else None,
            )
            if self.use_context
            else "global_only"
        )
        remaining = max(0.0, self.duration_s - state.current_time_s)
        available_j = (state.battery_frac - contract.min_final_battery_frac) * self.capacity_j
        requirements = observation.requirements
        report_remaining = (
            requirements.first_correct_report_deadline_s - state.current_time_s
            if requirements.first_correct_report_deadline_s is not None
            and observation.detections_so_far == 0
            else None
        )
        options = []
        for config in observation.configs:
            if not privacy_permits(contract.privacy_level, config) or config.strategy.is_remote:
                continue
            model = self.artifact["models"][config.config_id]
            stats = model["contexts"].get(context)
            fallback = (
                not self.use_context or stats is None or stats["frames"] < self.min_context_frames
            )
            stats = model["global"] if fallback else stats
            latency, energy = self.costs[config.config_id]
            resource = observation.resources.for_config(config.config_id)
            loading = state.current_config_id != config.config_id
            load_s = resource.load_latency_s if loading else 0.0
            load_j = resource.load_energy_j if loading else 0.0
            memory_fits = (
                observation.resources.memory_capacity_mb is None
                or resource.memory_mb <= observation.resources.memory_capacity_mb
            )
            busy = math.ceil((latency + self.router_latency_s) / 0.01 - 1e-9) * 0.01
            cadence = max(
                self.interval_s, math.ceil(busy / self.interval_s - 1e-9) * self.interval_s
            )
            projected = remaining * self.flight_power_w + math.ceil(remaining / cadence) * (
                energy + self.router_energy_j
            )
            projected += load_j + load_s * self.flight_power_w
            first_busy = math.ceil((latency + self.router_latency_s + load_s) / 0.01 - 1e-9) * 0.01
            quality = stats[metric]
            enough = (
                quality > contract.quality_threshold
                if str(contract.quality_operator) == ">"
                else quality >= contract.quality_threshold
            )
            supported = stats["visible_target_observations"] > 0 and stats["predictions"] > 0
            joint_checks = {}
            for name, value in (
                ("visible_frame_recall", requirements.min_target_recall),
                ("detection_precision", requirements.min_detection_precision),
            ):
                if value is not None:
                    joint_checks[name] = stats[name] >= value
            enough = enough and all(joint_checks.values())
            time_budget = min(state.remaining_deadline_s, remaining)
            if report_remaining is not None and report_remaining > 0:
                time_budget = min(time_budget, report_remaining)
            options.append(
                dict(
                    config_id=config.config_id,
                    quality_proxy=quality,
                    has_evidence=supported,
                    proxy_meets_request=enough and supported,
                    joint_proxy_checks=joint_checks,
                    latency_s=latency + load_s,
                    model_load_latency_s=load_s,
                    model_load_energy_j=load_j,
                    memory_fits=memory_fits,
                    projected_remaining_energy_j=projected,
                    context_fallback=fallback,
                    supporting_scenes=stats.get("scene_count", 1),
                    deadline_fits=first_busy <= time_budget,
                    battery_fits=projected <= available_j,
                    cadence_s=cadence,
                )
            )
        if not options:
            raise ValueError("no legal local candidate")
        memory_options = [o for o in options if o["memory_fits"]]
        feasible = [o for o in memory_options if o["deadline_fits"] and o["battery_fits"]]
        qualified = [o for o in feasible if o["proxy_meets_request"]]
        if qualified:
            chosen = min(
                qualified,
                key=lambda o: (o["latency_s"], o["projected_remaining_energy_j"], o["config_id"]),
            )
            reason = "fastest_meeting_development_quality_proxy"
        elif feasible:
            # Do not equate per-frame quality/latency with deduplicated mission recall.
            chosen = max(
                feasible, key=lambda o: (o["has_evidence"], o["quality_proxy"], -o["latency_s"])
            )
            reason = "best_available_proxy_request_unmet"
        else:
            chosen = min(
                memory_options or options,
                key=lambda o: (
                    not o["deadline_fits"],
                    o["projected_remaining_energy_j"],
                    o["latency_s"],
                ),
            )
            reason = "resource_budget_infeasible_best_effort"
        self.last_decision = dict(
            selected_config_id=chosen["config_id"],
            reason=reason,
            context=context,
            uses_context=self.use_context,
            context_supported=any(not option["context_fallback"] for option in options),
            quality_estimate_extrapolated=self.use_context
            and all(option["context_fallback"] for option in options),
            calibration_sha256=self.artifact_sha256,
            mission_quality_guaranteed=False,
            mission_requirements=requirements.to_dict(),
            report_deadline_guard="until first unverified prediction; correctness unknown online",
            mission_requirements_consumed=True,
            report_requirements_consumed=False,
            report_requirements_note="report-level calibration unavailable; detection proxies only",
            residency_costs_consumed=True,
            report_window_expired=report_remaining is not None and report_remaining <= 0,
            quality_proxy_metric=metric,
            candidates=options,
        )
        return chosen["config_id"]


def bind_router(router, scenario):
    """Keep train/test isolation checks outside the policy observation boundary."""
    if not isinstance(router, CalibratedRouter):
        return
    if scenario.recording_identity in router.artifact["training_recording_identities"]:
        raise ValueError("refusing to score a calibrated router on its training recording")
    if digest(scenario.world) in router.artifact.get("training_world_hashes", []):
        raise ValueError("refusing to score a calibrated router on its training world")
    if hasattr(router, "validate_protocol"):
        router.validate_protocol(scenario)
    router.bind_public_mission(
        executors=scenario.executors,
        duration_s=scenario.duration_s,
        capacity_wh=scenario.battery_capacity_wh,
        flight_power_w=scenario.flight_power_w,
        router_latency_s=scenario.router_latency_s,
        router_energy_j=scenario.router_energy_j,
        matching_iou_threshold=scenario.matching_iou_threshold,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("scenario", "recording", "probe", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--development",
        action="store_true",
        required=True,
        help="Explicitly declare this source is development data, never held-out evaluation",
    )
    parser.add_argument(
        "--use-altitude", action="store_true", help="Use versioned RGB/altitude contexts"
    )
    args = parser.parse_args()
    fit(args.scenario, args.recording, args.probe, args.output, use_altitude=args.use_altitude)
    print(args.output)


if __name__ == "__main__":
    main()
