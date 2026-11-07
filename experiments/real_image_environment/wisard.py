"""Strict importer for the pinned official WiSARD multi-modal sample.

Only RGB is evaluated. No invented timestamps, masks, tracks or camera poses.
"""

import argparse
import hashlib
import io
import json
import math
import zipfile
from pathlib import Path

from PIL import Image

SHA256 = "fd9bd786d53dc6fb9fcec20f2ae8f38df96db15c7edac348ba5a303c2285fc04"
SIZE = 1018780195
SOURCE = "https://sites.google.com/uw.edu/wisard/"
DOWNLOAD = "https://drive.google.com/file/d/1uSgMXuZGVCrWM_151UcykyHejxxaVHOo/view"
RGB_FOLDER = "210417_MtErie_Enterprise_VIS_0003"


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(path):
    if path.stat().st_size != SIZE or checksum(path) != SHA256:
        raise ValueError("WiSARD sample archive does not match pinned source")


def parse_boxes(text, width, height):
    """YOLO normalized center/extent to native half-open pixel boxes.

    Out-of-frame source extents are clipped, with originals and flags retained.
    Unknown classes/malformed/nonfinite/degenerate labels are rejected.
    """
    if width <= 0 or height <= 0:
        raise ValueError("invalid image dimensions")
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5 or fields[0] != "0":
            raise ValueError("expected class-0 person and four normalized coordinates")
        cx, cy, w, h = map(float, fields[1:])
        if not all(math.isfinite(x) for x in (cx, cy, w, h)) or w <= 0 or h <= 0:
            raise ValueError("invalid box coordinates")
        raw = [
            (cx - w / 2) * width,
            (cy - h / 2) * height,
            (cx + w / 2) * width,
            (cy + h / 2) * height,
        ]
        box = [
            max(0, min(limit, x))
            for x, limit in zip(raw, (width, height, width, height), strict=True)
        ]
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("box outside image or degenerate")
        rows.append(dict(box=box, source_normalized=[cx, cy, w, h], clipped=box != raw))
    return rows


def prepare(archive, output):
    verify(archive)
    records = []
    with zipfile.ZipFile(archive) as z:
        names = sorted(
            n for n in z.namelist() if n.startswith(RGB_FOLDER + "/") and n.endswith(".jpeg")
        )
        for name in names:
            raw = z.read(name)
            label = name.removesuffix(".jpeg") + ".txt"
            label_bytes = z.read(label)  # Missing is an error; empty is an annotated negative.
            with Image.open(io.BytesIO(raw)) as im:
                width, height = im.size
            boxes = parse_boxes(label_bytes.decode(), width, height)
            records.append(
                dict(
                    image_member=name,
                    label_member=label,
                    image_sha256=hashlib.sha256(raw).hexdigest(),
                    label_sha256=hashlib.sha256(label_bytes).hexdigest(),
                    width=width,
                    height=height,
                    boxes=boxes,
                    frame_number=int(Path(name).stem.rsplit("_", 1)[1]),
                    timestamp_s=None,
                    sequence_id=RGB_FOLDER,
                )
            )
    output.mkdir(parents=True, exist_ok=False)
    result = dict(
        schema_version="0.1",
        source_url=SOURCE,
        download_url=DOWNLOAD,
        archive=str(archive.resolve()),
        archive_sha256=SHA256,
        archive_bytes=SIZE,
        split="development_only",
        independent_flights=1,
        target_class="person",
        annotation_format="YOLO class 0; archive count.txt identifies humans",
        timing_status="unknown; frame numbers are not timestamps",
        tracks_available=False,
        camera_pose_available=False,
        license_notice_url=SOURCE,
        archive_contains_license=False,
        records=records,
        summary=dict(
            rgb_frames=len(records),
            person_box_observations=sum(len(r["boxes"]) for r in records),
            negative_frames=sum(not r["boxes"] for r in records),
            clipped_boxes=sum(b["clipped"] for r in records for b in r["boxes"]),
        ),
    )
    save(output / "manifest.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    print(json.dumps(prepare(args.archive, args.output)["summary"]))


if __name__ == "__main__":
    main()
