"""Reconstruct real-image boxes and cardinality independently, then render scoped results."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha


def run(root):
    from scipy import ndimage
    from scipy.optimize import linear_sum_assignment

    p = read(root / "protocol.json")
    result = read(root / "results.json")
    images = Path(p["images"])
    if not result["complete"] or result["protocol_sha256"] != sha(root / "protocol.json"):
        raise ValueError("Complete real-image execution required")
    for path, expected in p["implementations_sha256"].items():
        if sha(Path(path)) != expected:
            raise ValueError("Changed frozen evaluator")
    if sha(Path(p["annotation_path"])) != p["annotation_sha256"]:
        raise ValueError("Changed source annotations")
    if sha(images / "manifest.json") != p["image_manifest_sha256"]:
        raise ValueError("Changed image manifest")
    manifest = read(images / "manifest.json")
    lookup = {(r["image_id"], r["config_id"]): r for r in result["cases"]}
    expected = {
        (r["source_image_id"], c) for r in manifest["images"] for c in p["model_signatures"]
    }
    if len(lookup) != len(result["cases"]) or set(lookup) != expected or len(lookup) != 108:
        raise ValueError("Incomplete real-image model grid")
    labels = defaultdict(list)
    for a in read(Path(p["annotation_path"]))["annotations"]:
        if a["category_id"] in (1, 2):
            labels[a["image_id"]].append(a)
    sources, display = {}, []
    forwards = 0
    for entry in manifest["images"]:
        iid = entry["source_image_id"]
        path = images / entry["file"]
        if sha(path) != entry["sha256"]:
            raise ValueError("Changed source JPEG")
        sources[str(path)] = sha(path)
        with Image.open(path) as original:
            w, h = original.size
            ratio = min(1280 / w, 960 / h)
            rw, rh = round(w * ratio), round(h * ratio)
            dx, dy = (1280 - rw) // 2, (960 - rh) // 2
            image = Image.new("RGB", (1280, 960))
            image.paste(
                original.convert("RGB").resize((rw, rh), Image.Resampling.BILINEAR), (dx, dy)
            )
        truth = []
        for a in sorted(labels[iid], key=lambda a: a["id"]):
            x, y, bw, bh = a["bbox"]
            truth.append(
                [
                    max(0, x) * rw / w + dx,
                    max(0, y) * rh / h + dy,
                    min(w, x + bw) * rw / w + dx,
                    min(h, y + bh) * rh / h + dy,
                ]
            )
        stored_frame = next(r for r in result["frames"] if r["source_image_id"] == iid)
        if not np.allclose(
            np.asarray(truth), np.asarray(stored_frame["transformed_boxes"]), atol=1e-9, rtol=0
        ):
            raise ValueError("Source box transformation mismatch")
        masks = {}
        for config in p["model_signatures"]:
            row = lookup[iid, config]
            file = root / row["file"]
            if sha(file) != row["prediction_sha256"]:
                raise ValueError("Changed neural prediction")
            sources[str(file)] = sha(file)
            with np.load(file, allow_pickle=False) as data:
                mask = data["prediction_mask"]
            if mask.shape != (960, 1280) or mask.dtype != np.bool_:
                raise ValueError("Invalid person mask")
            masks[config] = mask
            components, count = ndimage.label(mask, ndimage.generate_binary_structure(2, 1))
            boxes = []
            for component in range(1, count + 1):
                y, x = np.nonzero(components == component)
                if len(x) >= 4:
                    boxes.append([int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1])
            if boxes != row["boxes"]:
                raise ValueError("Raw component boxes differ")
            overlaps = np.zeros((len(boxes), len(truth)))
            for i, a in enumerate(boxes):
                for j, b in enumerate(truth):
                    lo = np.maximum(a[:2], b[:2])
                    hi = np.minimum(a[2:], b[2:])
                    intersection = float(np.prod(np.maximum(0, hi - lo)))
                    union = float(
                        np.prod(np.subtract(a[2:], a[:2]))
                        + np.prod(np.subtract(b[2:], b[:2]))
                        - intersection
                    )
                    overlaps[i, j] = intersection / union if union else 0
            if overlaps.size:
                valid = overlaps >= 0.5
                objective = valid * (min(overlaps.shape) + 1 + overlaps)
                ii, jj = linear_sum_assignment(objective, maximize=True)
                matched = int(valid[ii, jj].sum())
            else:
                matched = 0
            if (
                matched != row["matched_swimmers"]
                or len(truth) != row["annotated_swimmers"]
                or len(boxes) != row["predicted_components"]
                or len(boxes) - matched != row["unmatched_components"]
            ):
                raise ValueError("Independent cardinality score mismatch")
            forwards += row["model_forwards"]
        if not np.array_equal(
            masks["CFG_MASKRCNN_1280_ROT0_90"],
            masks["CFG_MASKRCNN_1280"] | masks["CFG_MASKRCNN_1280_ROT90"],
        ):
            raise ValueError("Derived union mask differs")
        if entry["requested_time_s"] in (0, 60):
            draw = ImageDraw.Draw(image)
            for box in truth:
                draw.rectangle(box, outline="#fbbf24", width=3)
            for box in lookup[iid, "CFG_YOLO11S_TILED_640"]["boxes"]:
                draw.rectangle(box, outline="#34d399", width=2)
            display.append((image, entry))
    if forwards != 204 or forwards != result["actual_neural_calls"]:
        raise ValueError("Wrong physical neural call count")
    summary = []
    for config in p["model_signatures"]:
        rows = [r for r in result["cases"] if r["config_id"] == config]
        total = sum(r["annotated_swimmers"] for r in rows)
        matched = sum(r["matched_swimmers"] for r in rows)
        summary.append(
            dict(
                config_id=config,
                images=len(rows),
                swimmer_observations=total,
                matched=matched,
                recall=matched / total if total else None,
                unmatched_components=sum(r["unmatched_components"] for r in rows),
            )
        )
    save(
        root / "validation.json",
        dict(
            complete=True,
            image_config_pairs=len(lookup),
            actual_neural_calls=forwards,
            independent_matching_algorithm="Hungarian assignment with cardinality-first objective",
            original_annotations_and_masks_reconstructed=True,
            source_sha256=sha(Path(__file__)),
            input_files_sha256=sources,
            protocol_sha256=sha(root / "protocol.json"),
            results_sha256=sha(root / "results.json"),
            summary=summary,
            mission_success_claimed=False,
        ),
    )
    montage(display, root)
    lines = [
        "# Auxiliary real-flight image check",
        "",
        "The same nine frozen inference configurations process twelve predetermined SeaDronesSee "
        "training images: one nearest frame at each of 0, 20, 40, 60, 80 and 100 seconds in the "
        "two previously selected continuous telemetry windows. No image or annotation selected "
        "the timestamps. This small two-source sample is neither a held-out dataset benchmark "
        "nor a paid mission-routing experiment.",
        "",
        "Images preserve their aspect ratio in a 1280x960 centered black letterbox. Person-mask "
        "components of at least four pixels become bounding boxes. One-to-one matching uses "
        "box IoU ≥ 0.5 against categories 1 and 2 (swimmers, with or without life jackets). "
        "An independent Hungarian calculation reproduces all 108 image/configuration scores. "
        "The 204 neural forwards include tiled passes; the Mask R-CNN union reuses two "
        "already computed views. These numbers are not official dataset AP or mask IoU.",
        "",
        "| Configuration | Matched swimmer observations | Annotated observations | Recall | "
        "Unmatched person components |",
        "|---|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {r['config_id']} | {r['matched']} | {r['swimmer_observations']} | "
        f"{r['recall']:.3f} | {r['unmatched_components']} |"
        for r in summary
    ]
    lines += [
        "",
        "Repeated appearances are observations, not unique people or independent "
        "trials. An unmatched person component is not an independently confirmed false "
        "detection: the source annotations target people in water and may exclude "
        "other visible people. The figure uses the preselected tiled-YOLO reference "
        "at 0 and 60 seconds, with amber source boxes and green predicted component "
        "boxes. The source JPEGs remain unchanged.",
        "",
        "Source: [official SeaDronesSee dataset]"
        "(https://seadronessee.cs.uni-tuebingen.de/dataset). "
        "The [authors' repository](https://github.com/Ben93kie/SeaDronesSee) states CC0 1.0; "
        "the annotation JSON license field says `Unknown`. Both are retained. No test "
        "images were downloaded. This auxiliary check does not change the frozen "
        "504-case simulation experiment, fit a router, establish real-world mission "
        "success, or validate hardware energy assumptions.",
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines))
    names = ("protocol.json", "results.json", "validation.json", "REPORT.md", "sample_montage.jpg")
    save(
        root / "completion.json",
        dict(
            complete=True,
            source_sha256=sha(Path(__file__)),
            artifacts_sha256={n: sha(root / n) for n in names},
        ),
    )
    print(summary, flush=True)


def montage(display, root):
    canvas = Image.new("RGB", (1280, 1040), "#101923")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    for i, (image, entry) in enumerate(display):
        x, y = (i % 2) * 640, (i // 2) * 520
        canvas.paste(image.resize((640, 480)), (x, y + 40))
        draw.text(
            (x + 10, y + 10),
            f"{entry['source_identity']['drone']} | "
            f"t={entry['requested_time_s']}s | amber: source / green: tiled YOLO",
            fill="#edf3f9",
            font=font,
        )
    canvas.save(root / "sample_montage.jpg", quality=92)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
