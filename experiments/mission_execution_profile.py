"""Versioned CPU inference/tracking calibration on fixed DEVELOPMENT inputs only.

The profile is a nominal host-latency sensitivity input, never measured energy or
an onboard timing guarantee. Existing prediction banks and costs are untouched.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

from experiments.mission_runtime_profile import (
    DEVELOPMENT,
    REPO,
    digest,
    host_state,
    implementation_inventory,
    read,
    save,
    sha,
    summarize,
)

OUTPUT = REPO / "results/mission_execution_profile_v1_20261027"
ROUTING_PROFILE = REPO / "results/mission_runtime_profile_v1_20261027"
FLIGHTS = ("east_shore_day", "west_rocks_day")
CONTROL = "CFG_MASKRCNN_1280"
ROTATED = "CFG_MASKRCNN_1280_ROT90"
UNION = "CFG_MASKRCNN_1280_ROT0_90"
YOLO = "CFG_YOLO11S_960_ROT0"
NANO = "CFG_YOLO11N_640_ROT0"
TILED = "CFG_YOLO11S_TILED_640"
TILED_ROTATED = "CFG_YOLO11S_TILED_640_ROT90"
TILED_GAMMA = "CFG_YOLO11S_TILED_640_GAMMA"
CABINET = "CFG_CABINET_UAVID_1280"
CONFIGS = (CONTROL, ROTATED, UNION, YOLO, NANO, TILED, TILED_ROTATED, TILED_GAMMA, CABINET)
REPETITIONS = 2
FRAMES_PER_FLIGHT = 6


def even_indices(count, requested=FRAMES_PER_FLIGHT):
    if type(count) is not int or type(requested) is not int or not 2 <= requested <= count:
        raise ValueError("At least two distinct evenly spaced frames required")
    result = [round(i * (count - 1) / (requested - 1)) for i in range(requested)]
    if len(set(result)) != requested:
        raise ValueError("Repeated sampling index")
    return result


def timed_call(function, *args, **kwargs):
    wall0, cpu0 = time.perf_counter_ns(), time.process_time_ns()
    value = function(*args, **kwargs)
    cpu1, wall1 = time.process_time_ns(), time.perf_counter_ns()
    return value, dict(wall_s=(wall1 - wall0) / 1e9, process_s=(cpu1 - cpu0) / 1e9)


def native_executor(spec):
    """Fresh native backends avoid hidden cross-configuration model-cache reuse."""
    if spec.kind == "torch_instance_segmentation":
        from missionbench.v2.instance_models import (
            InstanceBackend,
            TorchInstanceSegmentationExecutor,
        )

        return TorchInstanceSegmentationExecutor(spec, backend=InstanceBackend(spec.parameters))
    if spec.kind == "torch_cabinet_segmentation":
        from missionbench.v2.cabinet_models import (
            CabinetBackend,
            TorchCabinetSegmentationExecutor,
        )

        return TorchCabinetSegmentationExecutor(spec, backend=CabinetBackend(spec.parameters))
    raise ValueError("Only the frozen native Mask R-CNN/CABiNet backends are supported")


class PipelineExecutor:
    """Exactly the existing configuration transforms, enclosed in one run call.

    Unlike the dense-bank acquisition, a union call executes both component
    models itself. It never reuses an uncharged prediction from another config.
    """

    def __init__(self, spec, control, *, factory=None, segmenter=None):
        from experiments.lake_fast_perception import Segmenter

        self.spec = spec
        factory, segmenter = factory or native_executor, segmenter or Segmenter
        name = spec.config_id
        if name not in CONFIGS:
            raise ValueError("Unknown frozen configuration")
        self.direct = factory(spec) if name in (CONTROL, CABINET) else None
        self.rotated = None
        if name in (ROTATED, UNION):
            rotated = replace(
                control,
                parameters={**control.parameters, "input_width_px": 960, "input_height_px": 1280},
            )
            self.rotated = factory(rotated)
        if name == UNION:
            self.direct = factory(control)
        if name in (YOLO, NANO, TILED, TILED_ROTATED, TILED_GAMMA):
            size = spec.parameters["imgsz"] if name in (YOLO, NANO) else 640
            self.yolo = segmenter(Path(spec.parameters["checkpoint"]), size)
        self.neural_calls = (
            2 if name == UNION else 4 if name in (TILED, TILED_ROTATED, TILED_GAMMA) else 1
        )

    def run(self, rgb):
        import numpy as np

        from experiments.mission_routing_v2.models import tiled_mask

        name = self.spec.config_id
        if name in (CONTROL, CABINET):
            return self.direct.run(rgb.copy()).prediction_mask
        if name in (ROTATED, UNION):
            direct = self.direct.run(rgb.copy()).prediction_mask if name == UNION else None
            rotated = self.rotated.run(np.rot90(rgb, 1).copy()).prediction_mask
            restored = np.rot90(rotated, -1).copy()
            return restored if name == ROTATED else np.logical_or(direct, restored)
        if name in (YOLO, NANO):
            return self.yolo.predict(rgb, 0)[0]
        return tiled_mask(
            rgb, self.yolo, turns=self.spec.parameters["turns"], gamma=self.spec.parameters["gamma"]
        )


def compare_masks(actual, expected):
    import numpy as np

    if actual.dtype != np.bool_ or actual.shape != expected.shape or expected.dtype != np.bool_:
        raise ValueError("Native boolean mask geometry required")
    different = int(np.count_nonzero(actual != expected))
    return dict(
        exact=different == 0,
        differing_pixels=different,
        pixels=int(actual.size),
        predicted_pixels=int(actual.sum()),
        expected_pixels=int(expected.sum()),
        prediction_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
        expected_mask_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),
    )


def load_mask(item):
    import numpy as np

    path = Path(item["path"])
    if sha(path) != item["sha256"]:
        raise ValueError("Changed pinned prediction mask file")
    with np.load(path, allow_pickle=False) as data:
        mask = data["prediction_mask"].copy()
    if mask.dtype != np.bool_ or hashlib.sha256(mask.tobytes()).hexdigest() != item["mask_sha256"]:
        raise ValueError("Prediction mask semantic digest mismatch")
    return mask


def checkpoint_inventory(specs, api):
    import torch
    from torchvision.models import get_model_weights

    inputs, mapping = {}, {}
    for spec in specs:
        p = spec.parameters
        if "checkpoint" in p:
            path = Path(p["checkpoint"])
            if sha(path) != p["checkpoint_sha256"]:
                raise ValueError("YOLO checkpoint differs from frozen configuration")
        elif "bundle_manifest" in p:
            manifest = Path(p["bundle_manifest"])
            bundle = api["verify_bundle"](manifest, p["bundle_sha256"])
            inputs[str(manifest)] = sha(manifest)
            for name, expected in bundle["files"].items():
                source = manifest.parent / name
                if sha(source) != expected:
                    raise ValueError("Changed pinned CABiNet bundle")
                inputs[str(source)] = expected
            path = manifest.parent / bundle["weights"]
        else:
            weights = get_model_weights(p["model_id"])[p["weights_id"]]
            path = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
            if not path.is_file():
                raise FileNotFoundError(
                    "Local cached checkpoint required; no download: " + str(path)
                )
            if not sha(path).startswith("73cbd019"):
                raise ValueError("Official Mask R-CNN checkpoint prefix differs")
        mapping[spec.config_id] = str(path.resolve())
        inputs[str(path.resolve())] = sha(path)
    return inputs, mapping


def imports():
    from missionbench.v2.cabinet_models import CabinetBackend, verify_bundle
    from missionbench.v2.instance_models import InstanceBackend
    from missionbench.v2.scenario import ExecutorConfigSpec
    from experiments.lake_fast_perception import Segmenter
    from experiments.lake_tracking import SurfaceProjector, TrackingConfig, TrackingPipeline
    from experiments.lake_tracking_runtime import load_public_terrain
    from experiments.mission_policy_selection.protocol import verify as verify_development
    from experiments.mission_routing_v2.models import POOL, InferencePool
    from experiments.mission_routing_v2.train import checked_manifest
    from experiments.v4_benchmark.calibrated_router import model_signature
    from experiments.v4_benchmark.probe_cache import rgb_key

    return dict(
        verify_bundle=verify_bundle,
        CabinetBackend=CabinetBackend,
        InstanceBackend=InstanceBackend,
        ExecutorConfigSpec=ExecutorConfigSpec,
        Segmenter=Segmenter,
        SurfaceProjector=SurfaceProjector,
        TrackingConfig=TrackingConfig,
        TrackingPipeline=TrackingPipeline,
        load_public_terrain=load_public_terrain,
        verify_development=verify_development,
        POOL=POOL,
        InferencePool=InferencePool,
        checked_manifest=checked_manifest,
        model_signature=model_signature,
        rgb_key=rgb_key,
    )


def prepare(output, api, runtime):
    import numpy as np

    from experiments.mission_runtime_profile import verify as verify_routing

    if output.resolve() != OUTPUT.resolve() or output.exists():
        raise ValueError("Use the new, nonexistent versioned execution-profile directory")
    protocol = api["verify_development"](DEVELOPMENT)
    if protocol["scope"] != "development_selection_known_map" or protocol["fresh_test"]:
        raise ValueError("Development recordings only")
    verify_routing(ROUTING_PROFILE)
    inputs = {str(DEVELOPMENT / "protocol.json"): sha(DEVELOPMENT / "protocol.json")}
    for filename in ("plan.json", "completion.json", "summary.json"):
        path = ROUTING_PROFILE / filename
        inputs[str(path)] = sha(path)
    source = Path(protocol["cohort"])
    inputs[str(source / "protocol.json")] = sha(source / "protocol.json")
    specs = None
    frames, banks, rgb_by_id, selected = {}, {}, {}, []
    for flight in FLIGHTS:
        if protocol["cases"][flight]["role"] != "development":
            raise ValueError("Evaluation flight forbidden")
        scenario = DEVELOPMENT / flight / "rapid_survey/scenario.json"
        inputs[str(scenario)] = sha(scenario)
        document = read(scenario)
        actual = tuple(api["ExecutorConfigSpec"](**s) for s in document["executor_configs"])
        if specs is None:
            specs = actual
        if actual != specs or tuple(s.config_id for s in specs) != tuple(api["POOL"]):
            raise ValueError("Development model pool differs")
        recording_path = source / flight / "recording/recording.json"
        inputs[str(recording_path)] = sha(recording_path)
        recording = read(recording_path)
        if recording["recording_identity"] != protocol["cases"][flight]["lineage"]["recording"]:
            raise ValueError("Development lineage differs")
        banks[flight] = api["checked_manifest"](
            source / flight / "predictions", source / "protocol.json"
        )
        bank = banks[flight]
        if bank["recording_sha256"] != sha(recording_path):
            raise ValueError("Prediction bank does not match recording")
        for filename in ("manifest.json", "completion.json", "inference.json"):
            path = source / flight / "predictions" / filename
            inputs[str(path)] = sha(path)
        for spec in specs:
            if bank["models"][spec.config_id]["signature"] != api["model_signature"](spec):
                raise ValueError("Prediction model signature differs")
        sample_ids = even_indices(len(recording["frames"]))
        for index, entry in enumerate(recording["frames"]):
            path = source / flight / "recording" / entry["file"]
            if sha(path) != entry["sha256"]:
                raise ValueError("Changed development RGB recording")
            inputs[str(path)] = entry["sha256"]
            with np.load(path, allow_pickle=False) as data:
                rgb = data["rgb"].copy()  # Never read semantic object_ids.
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError("Invalid RGB")
            key = api["rgb_key"](rgb)
            rgb_by_id[flight, index] = key
            if index in sample_ids:
                frame = dict(
                    flight=flight,
                    capture_id=index,
                    timestamp_ns=entry["timestamp_ns"],
                    path=str(path),
                    file_sha256=entry["sha256"],
                    rgb_key=key,
                    rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                )
                selected.append(frame)
                frames[flight, index] = rgb
    checkpoint_inputs, checkpoints = checkpoint_inventory(specs, api)
    inputs.update(checkpoint_inputs)
    expected = {}
    for frame in selected:
        for name in CONFIGS:
            item = banks[frame["flight"]]["models"][name]["predictions"][frame["rgb_key"]]
            path = source / frame["flight"] / "predictions" / item["file"]
            expected[frame["flight"], frame["capture_id"], name] = dict(
                path=str(path), sha256=item["sha256"], mask_sha256=item["mask_sha256"]
            )
            inputs[str(path)] = item["sha256"]
    traces = []
    prior = read(ROUTING_PROFILE / "summary.json")
    for pair in prior["pairs"]:
        root = ROUTING_PROFILE / pair["path"] / "control"
        paths = {
            name: root / name
            for name in (
                "execution/events.json",
                "execution/results.json",
                "outcome.json",
                "protocol.json",
            )
        }
        for path in paths.values():
            inputs[str(path)] = sha(path)
        events, result = read(paths["execution/events.json"]), read(paths["execution/results.json"])
        steps = []
        for e in events:
            name, index = e["execution"]["config_id"], e["observation_id"]
            key = rgb_by_id[pair["flight"], index]
            item = banks[pair["flight"]]["models"][name]["predictions"][key]
            path = source / pair["flight"] / "predictions" / item["file"]
            inputs[str(path)] = item["sha256"]
            steps.append(
                dict(
                    capture_id=index,
                    config_id=name,
                    capture_s=e["capture_time_s"],
                    completion_s=e["completion_time_s"],
                    pose=e["capture_body_pose"],
                    expected_reports=e["emitted_reports"],
                    mask=dict(
                        path=str(path), sha256=item["sha256"], mask_sha256=item["mask_sha256"]
                    ),
                )
            )
        scenario_path = DEVELOPMENT / pair["flight"] / pair["mission"] / "scenario.json"
        inputs[str(scenario_path)] = sha(scenario_path)
        document = read(scenario_path)
        traces.append(
            dict(
                flight=pair["flight"],
                mission=pair["mission"],
                policy=pair["policy"],
                events_path=str(paths["execution/events.json"]),
                steps=steps,
                camera_pitch_rad=document["world"].get("camera_pitch_rad", math.pi / 2),
                tracking=result["tracking"],
            )
        )
    terrain, metadata = api["load_public_terrain"](Path(protocol["assets"]))
    for path in (
        Path(metadata["raster_path"]),
        Path(protocol["assets"]) / "harmonic_terrain/model.sdf",
    ):
        inputs[str(path)] = sha(path)
    plan = dict(
        schema_version="mission_execution_host_plan_v1",
        purpose="development_only_cpu_latency",
        evaluation_recordings_used=False,
        development=str(DEVELOPMENT),
        cohort=str(source),
        frames=selected,
        frame_selection="Six evenly spaced indices including endpoints per day bank",
        repetitions=REPETITIONS,
        configs=list(CONFIGS),
        specs=[asdict(s) for s in specs],
        checkpoints=checkpoints,
        warm_samples_per_config=len(selected) * REPETITIONS,
        explicit_warmup_frames=[selected[0], selected[-1]],
        constructor_semantics="Fresh native models include original blank-image warmups. "
        "Separate first real-frame call and one additional warmup; no cold-boot claim",
        timing_boundary="Outer PipelineExecutor.run: RGB preparation and native forward, "
        "thresholding, inverse geometry and mask union. Excludes recorded-file decode, "
        "hash checks, comparisons and output saving. The live executor receives in-memory RGB",
        union_semantics="Both Mask R-CNN views execute on every union run; no shared predictions",
        tracking_boundary="TrackingPipeline.update: predicted-mask components, surface projection, "
        "association and report generation; no GT matching/scoring or file I/O",
        tracking_trace_count=len(traces),
        tracking_updates=sum(len(t["steps"]) for t in traces),
        tracking_source=str(ROUTING_PROFILE),
        tracking_battery_frac=0.65,
        tracking_caveat="Nominal cost on original development schedules; not a changed-schedule "
        "or low-battery cost guarantee",
        strict_bitwise_prediction_parity=True,
        no_threshold_tuning=True,
        latency_estimator="Unrounded warm wall p95, linear interpolation, 24 samples/config",
        independent_units="Two recordings; repeated frames are dependent, not trial replicates",
        statistical_inference=False,
        measured_electrical_power=False,
        onboard_validation=False,
        runtime=runtime,
        input_sha256=inputs,
        implementation_sha256=implementation_inventory(),
        package_versions={
            n: importlib.metadata.version(n)
            for n in (
                "torch",
                "torchvision",
                "ultralytics",
                "numpy",
                "scipy",
                "opencv-python-headless",
            )
        },
    )
    for path, expected_sha in inputs.items():
        if sha(path) != expected_sha:
            raise ValueError("Changed calibration input before freeze: " + path)
    output.mkdir(parents=True)
    save(output / "plan.json", plan)
    save(output / "tracking_inputs.json", traces)
    save(
        output / "expected_masks.json",
        [dict(flight=f, capture_id=i, config_id=c, **item) for (f, i, c), item in expected.items()],
    )
    save(
        output / "input_lock.json",
        {
            name: sha(output / name)
            for name in ("plan.json", "tracking_inputs.json", "expected_masks.json")
        },
    )
    return plan, specs, frames, expected, traces, terrain


def verify_inputs(output, plan):
    for name, expected in read(output / "input_lock.json").items():
        if sha(output / name) != expected:
            raise ValueError("Changed consumed calibration specification: " + name)
    for field in ("input_sha256", "implementation_sha256"):
        for path, expected in plan[field].items():
            if sha(path) != expected:
                raise ValueError("Changed calibration source/input: " + path)


def tracker_snapshot(tracker):
    state = vars(tracker.tracker).copy()
    state["config"] = asdict(state["config"])
    return json.loads(json.dumps(state, allow_nan=False, sort_keys=True))


def track_pair(trace, terrain, api):
    parameters = {k: trace["tracking"][k] for k in api["TrackingConfig"]().to_dict()}
    config = api["TrackingConfig"](**parameters)

    def factory():
        return api["TrackingPipeline"](
            config,
            api["SurfaceProjector"](
                terrain,
                anchor_height_m=trace["tracking"]["anchor_height_m"],
                hfov=trace["tracking"]["hfov_rad"],
            ),
        )

    baseline, profiled = factory(), factory()
    rows, emitted = [], []
    for index, step in enumerate(trace["steps"]):
        mask = load_mask(step["mask"])
        args = (mask, step["capture_s"], step["completion_s"])
        kwargs = dict(pose=step["pose"], camera_pitch_rad=trace["camera_pitch_rad"])
        if index % 2:
            actual, times = timed_call(profiled.update, *args, **kwargs)
            control = baseline.update(*args, **kwargs)
        else:
            control = baseline.update(*args, **kwargs)
            actual, times = timed_call(profiled.update, *args, **kwargs)
        if actual != control or actual != step["expected_reports"]:
            raise ValueError("Prediction-only tracking changed emitted reports")
        left, right = tracker_snapshot(baseline), tracker_snapshot(profiled)
        if left != right:
            raise ValueError("Tracking instrumentation changed state")
        emitted.extend(actual)
        rows.append(
            dict(
                capture_id=step["capture_id"],
                config_id=step["config_id"],
                instance_cold=index == 0,
                **times,
                report_count=len(actual),
                state_sha256=digest(right),
            )
        )
    return dict(
        flight=trace["flight"],
        mission=trace["mission"],
        policy=trace["policy"],
        parity=True,
        samples=rows,
        emitted_reports=emitted,
        final_state=tracker_snapshot(profiled),
    )


def execute(output, api, runtime):
    import numpy as np
    import torch

    plan, specs, frames, expected, traces, terrain = prepare(output, api, runtime)
    profiles, raw = {}, []
    control = next(s for s in specs if s.config_id == CONTROL)
    for spec in specs:
        verify_inputs(output, plan)
        before = host_state()
        executor, construction = timed_call(PipelineExecutor, spec, control)
        if torch.get_num_threads() != 4 or torch.get_num_interop_threads() != 1:
            raise ValueError("Executor changed declared Torch parallelism")
        schedule = [
            ("first_real_call", 0, plan["frames"][0]),
            ("additional_warmup", 0, plan["frames"][-1]),
        ]
        schedule.extend(
            ("warm", rep, frame) for rep in range(REPETITIONS) for frame in plan["frames"]
        )
        rows = []
        for number, (stage, rep, frame) in enumerate(schedule):
            rgb = frames[frame["flight"], frame["capture_id"]]
            prediction, times = timed_call(executor.run, rgb)
            reference = load_mask(expected[frame["flight"], frame["capture_id"], spec.config_id])
            comparison = compare_masks(prediction, reference)
            path = output / "predictions" / spec.config_id / f"{number:02d}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, prediction_mask=prediction)
            row = dict(
                stage=stage,
                repetition=rep,
                flight=frame["flight"],
                capture_id=frame["capture_id"],
                config_id=spec.config_id,
                neural_calls=executor.neural_calls,
                **times,
                **comparison,
                output=str(path.relative_to(output)),
                output_sha256=sha(path),
            )
            rows.append(row)
            if not comparison["exact"]:
                print(
                    f"PRESERVED MISMATCH {spec.config_id} {stage} {frame['flight']} "
                    f"frame{frame['capture_id']}: {comparison['differing_pixels']} pixels",
                    flush=True,
                )
        warm = [r for r in rows if r["stage"] == "warm"]
        wall, cpu = (
            summarize([r["wall_s"] for r in warm]),
            summarize([r["process_s"] for r in warm]),
        )
        profiles[spec.config_id] = dict(
            signature=api["model_signature"](spec),
            warm_samples=len(warm),
            warm_median_s=wall["p50_s"],
            warm_p95_s=wall["p95_s"],
            warm_wall=wall,
            warm_process=cpu,
            constructor=construction,
            first_real_call={k: rows[0][k] for k in ("wall_s", "process_s")},
            declared_latency_s=spec.mission_latency_s,
            checkpoint=plan["checkpoints"][spec.config_id],
            prediction_parity=all(r["exact"] for r in rows),
            host_before=before,
            host_after=host_state(),
            actual_neural_calls=sum(r["neural_calls"] for r in rows),
        )
        raw.extend(rows)
        save(output / "inference_samples.json", raw)
        save(output / "progress.json", dict(configs=len(profiles), expected_configs=len(CONFIGS)))
        print(
            f"{len(profiles)}/9 {spec.config_id}:24 warm samples; "
            f"p95={wall['p95_s']:.6f}s; parity={profiles[spec.config_id]['prediction_parity']}",
            flush=True,
        )
        del executor
        gc.collect()
    tracking = [track_pair(trace, terrain, api) for trace in traces]
    save(output / "tracking_samples.json", tracking)
    warm_tracking = [r for trace in tracking for r in trace["samples"] if not r["instance_cold"]]
    tracking_wall = summarize([r["wall_s"] for r in warm_tracking])
    tracking_cpu = summarize([r["process_s"] for r in warm_tracking])
    eligible = all(p["prediction_parity"] for p in profiles.values())
    profile = dict(
        schema_version="mission_execution_host_profile_v1",
        complete=eligible,
        measurement_complete=True,
        eligible_for_sensitivity=eligible,
        purpose=plan["purpose"],
        plan_sha256=sha(output / "plan.json"),
        input_lock_sha256=sha(output / "input_lock.json"),
        profiles=profiles,
        latency_s={name: p["warm_p95_s"] for name, p in profiles.items()},
        tracking=dict(
            eligible_for_sensitivity=True,
            parity=True,
            warm_median_s=tracking_wall["p50_s"],
            warm_p95_s=tracking_wall["p95_s"],
            warm_wall=tracking_wall,
            warm_process=tracking_cpu,
            traces=len(tracking),
            updates=sum(len(t["samples"]) for t in tracking),
            measured_boundary=plan["tracking_boundary"],
            caveat=plan["tracking_caveat"],
        ),
        unchanged=dict(
            router_latency_s=0.05,
            model_load_latency_s=0.3,
            model_load_energy_j=2,
            declared_active_power_w=15,
            measured_electrical_power=False,
        ),
        interpretation="Development-only nominal warm p95 sensitivity profile on one CPU host; "
        "not independent trials, onboard validation or worst-case bound",
    )
    verify_inputs(output, plan)
    save(output / "profile.json", profile)
    artifacts = {
        str(p.relative_to(output)): sha(p)
        for p in sorted(output.rglob("*"))
        if p.is_file() and p.name not in ("completion.json",)
    }
    save(
        output / "completion.json",
        dict(
            complete=eligible,
            measurement_complete=True,
            artifacts_sha256=artifacts,
            warm_samples=216,
            inference_calls=len(raw),
            actual_neural_calls=sum(r["neural_calls"] for r in raw),
            tracking_updates=profile["tracking"]["updates"],
            host_after=host_state(),
        ),
    )
    if not eligible:
        save(
            output / "REJECTED.json",
            dict(
                reason="Strict sampled prediction parity failed",
                deviations=[r for r in raw if not r["exact"]],
            ),
        )
        raise ValueError(
            "Preserved measurements are ineligible; do not use incompatible predictions"
        )
    verify(output)
    return profile


def verify(output):
    """Strict profile admission: immutable inputs/outputs, mask parity and derived statistics."""
    plan, completion, profile = (
        read(output / n) for n in ("plan.json", "completion.json", "profile.json")
    )
    if (
        not completion["complete"]
        or not profile["complete"]
        or not profile["eligible_for_sensitivity"]
    ):
        raise ValueError("Incomplete or incompatible host profile")
    if profile["plan_sha256"] != sha(output / "plan.json") or profile["input_lock_sha256"] != sha(
        output / "input_lock.json"
    ):
        raise ValueError("Changed profile plan/input lock")
    verify_inputs(output, plan)
    for name, expected in completion["artifacts_sha256"].items():
        if sha(output / name) != expected:
            raise ValueError("Changed measured artifact: " + name)
    if tuple(profile["profiles"]) != tuple(sorted(CONFIGS)) or set(profile["latency_s"]) != set(
        CONFIGS
    ):
        raise ValueError("Complete nine-configuration profile required")
    rows = read(output / "inference_samples.json")
    expected_masks = {
        (r["flight"], r["capture_id"], r["config_id"]): r
        for r in read(output / "expected_masks.json")
    }
    for row in rows:
        import numpy as np

        with np.load(output / row["output"], allow_pickle=False) as data:
            predicted = data["prediction_mask"].copy()
        expected_mask = load_mask(
            expected_masks[row["flight"], row["capture_id"], row["config_id"]]
        )
        checked = compare_masks(predicted, expected_mask)
        if not checked["exact"] or any(row[k] != v for k, v in checked.items()):
            raise ValueError("Saved prediction parity is not reproducible")
    expected_count = len(plan["frames"]) * plan["repetitions"]
    for name in CONFIGS:
        own = [r for r in rows if r["config_id"] == name]
        warm = [r for r in own if r["stage"] == "warm"]
        expected_grid = [
            (rep, f["flight"], f["capture_id"])
            for rep in range(plan["repetitions"])
            for f in plan["frames"]
        ]
        if [(r["repetition"], r["flight"], r["capture_id"]) for r in warm] != expected_grid:
            raise ValueError("Changed warm-sample denominator or selection")
        if len(own) != expected_count + 2 or any(not r["exact"] for r in own):
            raise ValueError("Missing first/warm calls or prediction mismatch")
        p = profile["profiles"][name]
        stats = summarize([r["wall_s"] for r in warm])
        if (
            p["warm_wall"] != stats
            or p["warm_samples"] != expected_count
            or not p["prediction_parity"]
        ):
            raise ValueError("Incorrect measured latency summary")
        if (
            p["warm_p95_s"] != stats["p95_s"]
            or profile["latency_s"][name] != stats["p95_s"]
            or p["warm_median_s"] != stats["p50_s"]
        ):
            raise ValueError("Sensitivity latency differs from measured warm statistics")
        from missionbench.v2.scenario import ExecutorConfigSpec
        from experiments.v4_benchmark.calibrated_router import model_signature

        spec = next(s for s in plan["specs"] if s["config_id"] == name)
        if p["signature"] != model_signature(ExecutorConfigSpec(**spec)):
            raise ValueError("Measured configuration signature differs")
    tracking = read(output / "tracking_samples.json")
    samples = [r for t in tracking for r in t["samples"] if not r["instance_cold"]]
    if len(tracking) != plan["tracking_trace_count"] or any(not t["parity"] for t in tracking):
        raise ValueError("Missing tracking trace/parity")
    if profile["tracking"]["warm_wall"] != summarize([r["wall_s"] for r in samples]):
        raise ValueError("Changed measured tracking statistics")
    if (
        profile["tracking"]["warm_p95_s"] != profile["tracking"]["warm_wall"]["p95_s"]
        or profile["tracking"]["warm_median_s"] != profile["tracking"]["warm_wall"]["p50_s"]
    ):
        raise ValueError("Wrong tracking sensitivity cost")
    if (
        sum(len(t["samples"]) for t in tracking) != plan["tracking_updates"]
        or profile["tracking"]["updates"] != plan["tracking_updates"]
        or not profile["tracking"]["eligible_for_sensitivity"]
        or not profile["tracking"]["parity"]
    ):
        raise ValueError("Incomplete tracking denominator/parity")
    return profile


def setup_runtime():
    for name, value in dict(
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        NUMEXPR_NUM_THREADS="1",
        BLIS_NUM_THREADS="1",
    ).items():
        os.environ[name] = value
    os.environ["TORCH_HOME"] = str(REPO / ".cache/torch")
    original, cores, selected = sorted(os.sched_getaffinity(0)), set(), []
    for cpu in original:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        key = ((topology / "physical_package_id").read_text(), (topology / "core_id").read_text())
        if key not in cores:
            cores.add(key)
            selected.append(cpu)
    if len(selected) < 4:
        raise ValueError("Four physical CPU cores are required for this profile")
    os.sched_setaffinity(0, set(selected[:4]))
    import cv2
    import torch

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    return dict(
        python=sys.version,
        executable=sys.executable,
        platform=platform.platform(),
        torch_threads=torch.get_num_threads(),
        torch_interop_threads=torch.get_num_interop_threads(),
        opencv_threads=cv2.getNumThreads(),
        original_affinity=original,
        affinity=sorted(os.sched_getaffinity(0)),
        host_before=host_state(),
        cpu_model=next(
            s.split(":", 1)[1].strip()
            for s in Path("/proc/cpuinfo").read_text().splitlines()
            if s.startswith("model name")
        ),
        environment={
            k: os.environ.get(k)
            for k in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "BLIS_NUM_THREADS",
                "TORCH_HOME",
            )
        },
        exclusive_host=False,
        electrical_power_measured=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        p = verify(args.output)
        print(
            json.dumps(
                dict(
                    complete=p["complete"],
                    configs=len(p["profiles"]),
                    warm_samples=sum(v["warm_samples"] for v in p["profiles"].values()),
                    tracking_updates=p["tracking"]["updates"],
                )
            )
        )
        return
    runtime = setup_runtime()
    execute(args.output, imports(), runtime)


if __name__ == "__main__":
    main()
