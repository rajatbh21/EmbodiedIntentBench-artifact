"""Frozen small real-image check of the existing perception pool, not a routing trial."""

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.mission_coverage_repair.protocol import verify
from experiments.mission_coverage_repair.study import scenario_for
from experiments.mission_routing_v2.models import InferencePool
from experiments.v4_benchmark.calibrated_router import model_signature


def letterbox(image, size=(1280, 960)):
    scale = min(size[0] / image.width, size[1] / image.height)
    w, h = round(image.width * scale), round(image.height * scale)
    left, top = (size[0] - w) // 2, (size[1] - h) // 2
    canvas = Image.new("RGB", size)
    canvas.paste(image.resize((w, h), Image.Resampling.BILINEAR), (left, top))
    return np.asarray(canvas).copy(), (w / image.width, h / image.height, left, top)


def component_boxes(mask, minimum_area=4):
    from scipy import ndimage

    labels, n = ndimage.label(mask)
    counts = np.bincount(labels.ravel(), minlength=n + 1)
    return [
        [s[1].start, s[0].start, s[1].stop, s[0].stop]
        for i, s in enumerate(ndimage.find_objects(labels), 1)
        if s is not None and counts[i] >= minimum_area
    ]


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def match(predictions, truth, threshold=0.5):
    """Maximum-cardinality, one-to-one IoU-threshold matching; deterministic ties."""
    edges = [
        sorted(
            (j for j, box in enumerate(truth) if iou(p, box) >= threshold),
            key=lambda j: (-iou(p, truth[j]), j),
        )
        for p in predictions
    ]
    assigned = {}

    def augment(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in assigned or augment(assigned[j], seen):
                assigned[j] = i
                return True
        return False

    for i in range(len(predictions)):
        augment(i, set())
    return [
        dict(prediction_index=i, truth_index=j, iou=iou(predictions[i], truth[j]))
        for j, i in sorted(assigned.items())
    ]


def prepare(images, study, output):
    p = verify(study)
    manifest = read(images / "manifest.json")
    if not manifest["complete"] or manifest["test_images_downloaded"]:
        raise ValueError("Completed training image sample required")
    selection = read(images / "selection.json")
    if sha(images / "selection.json") != manifest["selection_sha256"]:
        raise ValueError("Changed image selection")
    scenario = scenario_for(study, p, next(iter(p["cases"])), "rapid_survey", 15)
    output.mkdir(parents=True, exist_ok=False)
    save(
        output / "protocol.json",
        dict(
            images=str(images.resolve()),
            study=str(study.resolve()),
            image_manifest_sha256=sha(images / "manifest.json"),
            study_protocol_sha256=sha(study / "protocol.json"),
            annotation_path=str(
                Path(selection["source"]) / "instances_train_objects_in_water.json"
            ),
            annotation_sha256=selection["annotation_sha256"],
            model_signatures={s.config_id: model_signature(s) for s in scenario.executors},
            implementations_sha256={
                str(Path(__file__).resolve()): sha(Path(__file__)),
                str(Path("tests/test_real_image_check.py").resolve()): sha(
                    Path("tests/test_real_image_check.py")
                ),
            },
            scope="Twelve fixed-time real training images; auxiliary perception sanity check",
            source_identity_groups=2,
            images_count=12,
            image_size=[1280, 960],
            preprocessing="Aspect-preserving bilinear resize with centered black letterbox",
            target_categories=[1, 2],
            minimum_component_pixels=4,
            box_iou_threshold=0.5,
            matching="Maximum-cardinality one-to-one matching of connected-component boxes",
            unmatched_component_caveat="Annotations cover swimmers and swimmers with life jackets; "
            "unmatched person components are not independently verified false detections",
            paid_mission_replay=False,
            mission_success_claimed=False,
            training_split_only=True,
            policy_training_changed=False,
            independent_real_world_benchmark_claimed=False,
        ),
    )


def run(output):
    p = read(output / "protocol.json")
    for name, expected in p["implementations_sha256"].items():
        if sha(Path(name)) != expected:
            raise ValueError("Changed real-image evaluator")
    images, study = Path(p["images"]), Path(p["study"])
    if (
        sha(images / "manifest.json") != p["image_manifest_sha256"]
        or sha(study / "protocol.json") != p["study_protocol_sha256"]
        or sha(Path(p["annotation_path"])) != p["annotation_sha256"]
    ):
        raise ValueError("Changed input")
    study_p = verify(study)
    scenario = scenario_for(study, study_p, next(iter(study_p["cases"])), "rapid_survey", 15)
    if {s.config_id: model_signature(s) for s in scenario.executors} != p["model_signatures"]:
        raise ValueError("Changed model pool")
    if (output / "predictions").exists():
        raise FileExistsError("Preserve real-image inference attempts")
    (output / "predictions").mkdir()
    annotations = defaultdict(list)
    for a in read(Path(p["annotation_path"]))["annotations"]:
        if a["category_id"] in p["target_categories"]:
            annotations[a["image_id"]].append(a)
    import torch

    torch.set_num_threads(4)
    pool = InferencePool(scenario.executors)
    rows, frames, actual_calls = [], [], 0
    for r in read(images / "manifest.json")["images"]:
        path = images / r["file"]
        if sha(path) != r["sha256"]:
            raise ValueError("Changed original image")
        with Image.open(path) as im:
            if im.size != (r["source_width"], r["source_height"]):
                raise ValueError("Image/annotation size mismatch")
            rgb, (sx, sy, left, top) = letterbox(im.convert("RGB"))
        truth, ids = [], []
        for a in sorted(annotations[r["source_image_id"]], key=lambda a: a["id"]):
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0 or not np.isfinite([x, y, w, h]).all():
                raise ValueError("Invalid source bounding box")
            truth.append(
                [
                    max(0, x) * sx + left,
                    max(0, y) * sy + top,
                    min(r["source_width"], x + w) * sx + left,
                    min(r["source_height"], y + h) * sy + top,
                ]
            )
            ids.append(a["id"])
        started = time.monotonic()
        masks, timings, calls = pool.infer(rgb)
        actual_calls += sum(calls.values())
        frame = dict(
            **r,
            transformed_boxes=truth,
            annotation_ids=ids,
            transform=dict(scale_x=sx, scale_y=sy, left=left, top=top),
        )
        frames.append(frame)
        for name, mask in masks.items():
            predicted = component_boxes(mask)
            paired = match(predicted, truth)
            filename = f"predictions/{r['source_image_id']}_{name}.npz"
            np.savez_compressed(output / filename, prediction_mask=mask)
            rows.append(
                dict(
                    image_id=r["source_image_id"],
                    group_id=r["group_id"],
                    config_id=name,
                    annotated_swimmers=len(truth),
                    predicted_components=len(predicted),
                    matched_swimmers=len(paired),
                    unmatched_components=len(predicted) - len(paired),
                    boxes=predicted,
                    matches=paired,
                    model_forwards=calls[name],
                    measured_host_inference_s=timings[name],
                    file=filename,
                    prediction_sha256=sha(output / filename),
                )
            )
        print(
            f"Real frame {len(frames)}/12: {len(truth)} swimmer annotations, "
            f"{time.monotonic() - started:.1f}s",
            flush=True,
        )
    save(
        output / "results.json",
        dict(
            complete=True,
            frames=frames,
            cases=rows,
            actual_neural_calls=actual_calls,
            image_config_pairs=len(rows),
            protocol_sha256=sha(output / "protocol.json"),
            manifest_sha256=sha(images / "manifest.json"),
            source_sha256=sha(Path(__file__)),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images", type=Path)
    parser.add_argument("--study", type=Path)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.images, args.study, args.output)
    else:
        run(args.output)
