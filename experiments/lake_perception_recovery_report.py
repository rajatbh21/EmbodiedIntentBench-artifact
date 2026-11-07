"""GT-isolated analysis of the completed lake perception development screen."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from missionbench.v2.evaluation import MissionEvaluator
from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.lake_missions.run import freeze
from experiments.lake_missions.scenario import load
from experiments.v4_benchmark.backends import ReplayBackend
from experiments.v4_benchmark.probe_cache import rgb_key
from experiments.v4_benchmark.runner import evaluation_frame


def validate_screen(rows, indices, configurations):
    expected = {(i, c) for i in indices for c in configurations}
    keys = [(r["capture_id"], r["config_id"]) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Screen requires exactly one result for every selected frame/config")
    if len(set(indices)) != len(indices) or len(set(configurations)) != len(configurations):
        raise ValueError("Duplicate protocol selection")


def summarize(rows, targets):
    tp = sum(r["matched_components"] for r in rows)
    pred = sum(r["predicted_components"] for r in rows)
    visible = sum(len(r["visible_target_ids"]) for r in rows)
    times = [r["measured_wall_clock_s"] for r in rows]
    return dict(
        captures=len(rows),
        matched_components=tp,
        false_positive_components=pred - tp,
        precision=tp / pred if pred else None,
        visible_target_observation_recall=tp / visible if visible else None,
        host_latency_median_s=float(np.median(times)),
        host_latency_p95_s=float(np.percentile(times, 95)),
        target_matched_captures={
            target: sum(target in r["matched_target_ids"] for r in rows) for target in targets
        },
    )


def phase_quality(rows, missions, configurations):
    """Conditional sampled quality, not repeated-evidence mission success."""
    result = []
    for mission in missions:
        for phase in mission["phases"]:
            targets = set(phase["evaluator_target_ids"])
            for config in configurations:
                selected = [
                    r
                    for r in rows
                    if r["config_id"] == config
                    and phase["start_s"] <= r["capture_time_s"] < phase["end_s"]
                ]
                visible = sum(len(targets & set(r["visible_target_ids"])) for r in selected)
                matched = sum(len(targets & set(r["matched_target_ids"])) for r in selected)
                result.append(
                    dict(
                        mission_id=mission["mission_id"],
                        phase_id=phase["phase_id"],
                        config_id=config,
                        sampled_captures=len(selected),
                        visible_phase_target_observations=visible,
                        matched_phase_target_observations=matched,
                        phase_target_observation_recall=matched / visible if visible else None,
                        note="Sampled development quality only; not stage success or feasibility",
                    )
                )
    return result


def diagnostic_image(output, environment, frames, masks, targets, configs):
    """GT-selected diagnostic crops only; never consumed by inference."""
    width, height = 320, 250
    canvas = Image.new("RGB", (width * (len(configs) + 1), height * 2 + 55), "#15202d")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (12, 12), f"{environment}: diagnostic crops selected by visible GT area", fill="white"
    )
    draw.text(
        (12, 30),
        "Green: predicted mask. Yellow: evaluator GT box. No GT used by models.",
        fill="white",
    )
    selected = []
    for row, target in enumerate(("west_2", "west_3")):
        label = next(k for k, v in targets.items() if v == target)
        index = max(frames, key=lambda i: np.count_nonzero(frames[i].object_ids == label))
        frame = frames[index]
        yy, xx = np.where(frame.object_ids == label)
        if not len(xx):
            continue
        x, y = int(xx.mean()), int(yy.mean())
        x0, y0 = max(0, min(x - 70, 820)), max(0, min(y - 50, 620))
        box = (x0, y0, x0 + 140, y0 + 100)
        selected.append(dict(target=target, capture_id=index, diagnostic_crop=list(box)))
        for col, config in enumerate([None, *configs]):
            rgb = frame.rgb.copy()
            if config is not None:
                mask = masks[index, config]
                rgb[mask] = (rgb[mask].astype(float) * 0.35 + np.array([0, 255, 90]) * 0.65).astype(
                    np.uint8
                )
            image = Image.fromarray(rgb)
            ImageDraw.Draw(image).rectangle(
                (int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())), outline="yellow"
            )
            tile = image.crop(box).resize((width, 210), Image.Resampling.NEAREST)
            canvas.paste(tile, (col * width, 55 + row * height + 40))
            caption = "RGB + evaluator box" if config is None else config.removeprefix("CFG_")
            draw.text((col * width + 5, 55 + row * height + 5), caption, fill="white")
            draw.text(
                (col * width + 5, 55 + row * height + 22),
                f"{target}, capture {index}",
                fill="white",
            )
    canvas.save(output / environment / "diagnostics.png")
    save(output / environment / "diagnostics.json", dict(selection=selected, inference_input=False))


def analyze(output, prior):
    protocol = read(output / "protocol.json")
    source = Path(protocol["source"])
    freeze(source)
    assert protocol["source_lock_sha256"] == sha(source / "lock.json")
    assert protocol["inference_script_sha256"] == sha(
        Path(__file__).with_name("lake_perception_recovery.py")
    )
    configs = [s["config_id"] for s in protocol["configurations"]]
    suite = read(source / "suite.json")
    summaries = {}
    for environment, recording_meta in protocol["recordings"].items():
        folder = output / environment
        complete = read(folder / "completion.json")
        assert complete["complete"] and complete["protocol_sha256"] == sha(output / "protocol.json")
        assert complete["inference_sha256"] == sha(folder / "inference.json")
        rows = read(folder / "inference.json")
        validate_screen(rows, recording_meta["capture_indices"], configs)
        assert complete["actual_calls"] == len(rows)
        indexed = {(r["capture_id"], r["config_id"]): r for r in rows}
        scenario = load(
            source / "scenario.json", suite["environments"][environment], Path(suite["assets"])
        )
        backend = ReplayBackend(scenario, source / environment / "recording")
        assert backend.provenance["recording_sha256"] == recording_meta["manifest_sha256"]
        old_folder = prior / environment / "predictions"
        old = read(old_folder / "manifest.json")
        old_complete = read(old_folder / "completion.json")
        assert old_complete["complete"] and old_complete["manifest_sha256"] == sha(
            old_folder / "manifest.json"
        )
        assert old["recording_identity"] == scenario.recording_identity
        assert old["recording_sha256"] == recording_meta["manifest_sha256"]
        scorers = {
            c: MissionEvaluator(
                scenario.matching_iou_threshold,
                len(scenario.targets),
                scenario.observation_interval_s,
            )
            for c in configs
        }
        scored, frames, masks = [], {}, {}
        identical = 0
        try:
            for index in recording_meta["capture_indices"]:
                timestamp = sorted(backend.frames)[index]
                backend.advance_to(timestamp)
                frame = backend.capture()
                frames[index] = frame
                t = (timestamp - scenario.schedule.epoch_ns) / 1e9
                truth = evaluation_frame(frame, index, t, scenario)
                for config in configs:
                    row = indexed[index, config]
                    assert row["timestamp_ns"] == timestamp and row["capture_time_s"] == t
                    assert row["rgb_sha256"] == rgb_key(frame.rgb)
                    path = (folder / row["file"]).resolve()
                    assert path.parent == folder.resolve() and sha(path) == row["sha256"]
                    with np.load(path, allow_pickle=False) as data:
                        mask = data["prediction_mask"].copy()
                    assert mask.dtype == np.bool_ and mask.shape == frame.rgb.shape[:2]
                    assert row["execution"]["success"] and row["execution"]["config_id"] == config
                    masks[index, config] = mask
                    if config == "CFG_MASKRCNN_1280":
                        previous = old["models"][config]["predictions"][row["rgb_sha256"]]
                        old_path = (old_folder / previous["file"]).resolve()
                        assert (
                            old_path.parent == old_folder.resolve()
                            and sha(old_path) == previous["sha256"]
                        )
                        with np.load(old_path, allow_pickle=False) as data:
                            identical += int(np.array_equal(mask, data["prediction_mask"]))
                    scored.append(
                        dict(
                            capture_id=index,
                            config_id=config,
                            measured_wall_clock_s=row["execution"]["measured_wall_clock_s"],
                            **scorers[config].update(truth, mask).to_dict(),
                        )
                    )
        finally:
            backend.close()
        save(folder / "outcomes.json", scored)
        save(
            folder / "phase_quality.json",
            phase_quality(
                scored,
                read(source / "missions.json")["missions"],
                configs,
            ),
        )
        summaries[environment] = dict(
            configurations={
                c: summarize([r for r in scored if r["config_id"] == c], scenario.targets.values())
                for c in configs
            },
            control_exact_mask_reproductions=identical,
            control_captures=len(recording_meta["capture_indices"]),
            prior_manifest_sha256=sha(old_folder / "manifest.json"),
        )
        diagnostic_image(output, environment, frames, masks, scenario.targets, configs)
    report = dict(
        scope=protocol["scope"],
        protocol_sha256=sha(output / "protocol.json"),
        analysis_script_sha256=sha(Path(__file__)),
        environments=summaries,
        all_captures_scored=all(
            len(r["capture_indices"]) == r["captures"] for r in protocol["recordings"].values()
        ),
        mission_feasibility_certified=False,
        policy_executed=False,
        energy_measured=False,
        caveat="A sampled screen cannot certify full-recording feasibility or infeasibility.",
    )
    save(output / "summary.json", report)
    save(
        output / "host_profiles.json",
        dict(
            scope="Development host timing summary; not an onboard deployment profile",
            source_summary_sha256=sha(output / "summary.json"),
            host_threads=protocol["host_threads"],
            configurations={
                c: dict(
                    max_environment_p95_executor_s=max(
                        env["configurations"][c]["host_latency_p95_s"] for env in summaries.values()
                    ),
                    measured_energy_j=None,
                    measured_resident_memory_mb=None,
                    switch_latency_s=None,
                    promoted_to_router_pool=False,
                )
                for c in configs
            },
        ),
    )
    write_report(output, report)
    return report


def write_report(output, report):
    lines = [
        "# Lake perception recovery screen",
        "",
        report["scope"],
        "",
        "Fresh CPU inference on identical sampled RGB, with unchanged COCO weights and "
        "thresholds. The 960 x 720 sensor images were upsampled or processed with four "
        "fixed overlapping tiles. No new sensor detail was introduced. Ground truth "
        "was used only after inference.",
        "",
        "| Environment | Configuration | Matches | False components | Precision | "
        "Visible-observation recall | Median / p95 host seconds | west_2 / west_3 matches |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    images = []
    for env, data in report["environments"].items():
        for config, m in data["configurations"].items():
            precision = "n/a" if m["precision"] is None else f"{m['precision']:.3f}"
            hits = m["target_matched_captures"]
            lines.append(
                f"| {env} | {config} | {m['matched_components']} | "
                f"{m['false_positive_components']} | {precision} | "
                f"{m['visible_target_observation_recall']:.3f} | "
                f"{m['host_latency_median_s']:.2f} / {m['host_latency_p95_s']:.2f} | "
                f"{hits['west_2']} / {hits['west_3']} |"
            )
        images.extend(
            [
                "",
                f"{env}: {data['control_exact_mask_reproductions']}/"
                f"{data['control_captures']} freshly computed control masks exactly "
                "reproduce the prior cache.",
                "",
                f"![{env} diagnostic crops]({env}/diagnostics.png)",
                "",
            ]
        )
    lines.extend(images)
    lines.extend(
        [
            "## Interpretation limits",
            "",
            "Every configuration used the same predeclared capture indices. Diagnostic "
            "image crops were selected by GT area only for inspection and were never model "
            "inputs. Green pixels are actual predicted masks; yellow boxes are explicitly "
            "evaluator ground truth.",
            "",
            "Host latency includes preprocessing, inference and postprocessing, excludes "
            "model loading/warmup, and is measured once per sampled capture with four CPU "
            "threads. Configuration order rotates by capture. These correlated observations "
            "are not independent flights; p95 is descriptive. This is not drone energy or "
            "on-board hardware calibration.",
            "",
            "This experiment does not execute a routing policy, change the frozen model "
            "pool, adjust mission thresholds, or establish full-mission feasibility. A dense "
            "table and legal cost-aware executions are required before promotion. The "
            "previous infeasibility certificate still applies only to its original model pool.",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("prior", type=Path)
    args = parser.parse_args()
    analyze(args.output, args.prior)
