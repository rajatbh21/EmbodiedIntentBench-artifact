"""Compare actual pretrained boxes at configurable input resolutions on real RGB.

Uniform source-index sampling is fixed before inference, never selected by GT or
model success. All data belongs to one development flight. Native boxes stay boxes.
"""

import argparse
import hashlib
import io
import json
import platform
import statistics
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from experiments.real_image_environment.boxes import match
from experiments.real_image_environment.wisard import checksum, parse_boxes, save, verify


def sample_indices(count, limit):
    if count < 1 or limit < 1:
        raise ValueError("sample sizes must be positive")
    if limit >= count:
        return list(range(count))
    if limit == 1:
        return [0]
    return [i * (count - 1) // (limit - 1) for i in range(limit)]


def run(manifest_path, output, frames=12, sizes=(320, 640, 1280), threads=2):
    if threads < 1 or not sizes or len(set(sizes)) != len(sizes) or min(sizes) < 32:
        raise ValueError("invalid probe settings")
    data = json.loads(manifest_path.read_text())
    archive = Path(data["archive"])
    verify(archive)
    records = [data["records"][i] for i in sample_indices(len(data["records"]), frames)]
    import torch
    import torchvision
    from torchvision.models.detection import (
        MaskRCNN_ResNet50_FPN_V2_Weights,
        maskrcnn_resnet50_fpn_v2,
    )

    torch.set_num_threads(threads)
    # Require the existing local checkpoint: no hidden network access or model download.
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / weights.url.rsplit("/", 1)[1]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"existing checkpoint required: {checkpoint}")
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(
        manifest_sha256=checksum(manifest_path),
        archive_sha256=data["archive_sha256"],
        selection="uniform source-index positions, endpoints included",
        image_members=[r["image_member"] for r in records],
        sizes=list(sizes),
        score_threshold=0.5,
        iou_threshold=0.5,
        threads=threads,
        model=str(weights),
        checkpoint_sha256=checksum(checkpoint),
        torch_version=torch.__version__,
        torchvision_version=torchvision.__version__,
        device="cpu",
        platform=platform.platform(),
        latency_scope=(
            "resize + tensor conversion + model + CPU box export; "
            "excludes image decode, load and warmup"
        ),
        timing_use="host diagnostic only; no mission time or energy inferred",
        split="development_only",
        model_families=1,
    )
    save(output / "protocol.json", protocol)  # Frozen before observing any prediction.
    started = time.perf_counter()
    model = maskrcnn_resnet50_fpn_v2(weights=weights).eval()
    load_s = time.perf_counter() - started
    rows, totals = [], {}
    panels = []
    with zipfile.ZipFile(archive) as z, torch.inference_mode():
        for size in sizes:
            # Explicit aspect-preserving resize; prevent torchvision from upscaling to 800.
            model.transform.min_size = (1,)
            model.transform.max_size = size

            def infer(image, size=size):
                t0 = time.perf_counter()
                w, h = image.size
                scale = min(1.0, size / max(w, h))
                nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
                im = image.resize((nw, nh), Image.Resampling.BILINEAR)
                tensor = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255
                model.transform.min_size = (min(nw, nh),)
                result = model([tensor])[0]
                keep = (result["labels"] == 1) & (result["scores"] >= 0.5)
                boxes, scores = result["boxes"][keep].numpy(), result["scores"][keep].numpy()
                predictions = [
                    dict(
                        box=[
                            float(b[0] * w / nw),
                            float(b[1] * h / nh),
                            float(b[2] * w / nw),
                            float(b[3] * h / nh),
                        ],
                        score=float(s),
                    )
                    for b, s in zip(boxes, scores, strict=True)
                ]
                return predictions, time.perf_counter() - t0

            infer(Image.new("RGB", (size, max(32, size * 9 // 16))))
            for frame_index, record in enumerate(records):
                raw = z.read(record["image_member"])
                labels = z.read(record["label_member"])
                if (
                    hashlib.sha256(raw).hexdigest() != record["image_sha256"]
                    or hashlib.sha256(labels).hexdigest() != record["label_sha256"]
                ):
                    raise ValueError("source member checksum mismatch")
                im = Image.open(io.BytesIO(raw)).convert("RGB")
                if (
                    im.size != (record["width"], record["height"])
                    or parse_boxes(labels.decode(), *im.size) != record["boxes"]
                ):
                    raise ValueError("manifest geometry differs from source image/labels")
                predictions, elapsed = infer(im)
                score = match(predictions, [b["box"] for b in record["boxes"]])
                row = dict(
                    image_member=record["image_member"],
                    long_side_px=size,
                    predictions=predictions,
                    score=score,
                    host_latency_s=elapsed,
                )
                rows.append(row)
                save(output / "predictions.json", rows)
                if frame_index < 4:
                    thumb = im.resize((640, 360))
                    draw = ImageDraw.Draw(thumb)
                    for boxes, color in (
                        ([b["box"] for b in record["boxes"]], "lime"),
                        ([b["box"] for b in predictions], "red"),
                    ):
                        for box in boxes:
                            draw.rectangle(
                                [
                                    box[0] * 640 / im.width,
                                    box[1] * 360 / im.height,
                                    box[2] * 640 / im.width,
                                    box[3] * 360 / im.height,
                                ],
                                outline=color,
                                width=2,
                            )
                    draw.rectangle((0, 0, 640, 23), fill="black")
                    draw.text(
                        (5, 5),
                        f"{Path(record['image_member']).stem[-8:]} | {size}px | "
                        f"TP {score['tp']} FP {score['fp']} FN {score['fn']}",
                        fill="white",
                    )
                    panels.append((size, frame_index, thumb))
                print(
                    json.dumps(
                        dict(
                            size=size,
                            frame=frame_index,
                            tp=score["tp"],
                            fp=score["fp"],
                            fn=score["fn"],
                        )
                    ),
                    flush=True,
                )
            selected = [r for r in rows if r["long_side_px"] == size]
            tp, fp, fn = (sum(r["score"][k] for r in selected) for k in ("tp", "fp", "fn"))
            totals[str(size)] = dict(
                tp=tp,
                fp=fp,
                fn=fn,
                precision=tp / (tp + fp) if tp + fp else None,
                box_recall=tp / (tp + fn) if tp + fn else None,
                median_host_latency_s=statistics.median(r["host_latency_s"] for r in selected),
            )
    canvas = Image.new("RGB", (640 * min(4, len(records)), 360 * len(sizes) + 35), "black")
    for size, index, panel in panels:
        canvas.paste(panel, (index * 640, list(sizes).index(size) * 360))
    ImageDraw.Draw(canvas).text(
        (10, canvas.height - 25),
        "WiSARD (Broyles, Hayner, Leung) | green: source boxes; "
        "red: model | one development flight",
        fill="white",
    )
    canvas.save(output / "comparison.jpg")
    summary = dict(
        frames=len(records),
        configurations=len(sizes),
        totals=totals,
        model_load_s=load_s,
        unique_person_recall=None,
        mission_success=None,
        limitation="one development flight; box observations repeat people; no timestamps or IDs",
    )
    save(output / "summary.json", summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--frames", type=int, default=12)
    p.add_argument("--sizes", type=int, nargs="+", default=[320, 640, 1280])
    p.add_argument("--threads", type=int, default=2)
    a = p.parse_args()
    print(json.dumps(run(a.manifest, a.output, a.frames, a.sizes, a.threads)))


if __name__ == "__main__":
    main()
