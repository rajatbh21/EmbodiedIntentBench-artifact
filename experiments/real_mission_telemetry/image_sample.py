"""Fetch fixed-time training images from the official public ZIP using bounded ranges.

This auxiliary external-image check does not alter the frozen routing experiment.
No image or object annotation influences selection of the twelve timestamps.
"""

import argparse
import io
import re
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha

URL = (
    "https://cloud.cs.uni-tuebingen.de/public.php/dav/files/W4ztazMxqfHdYWA/"
    "SeaDronesSee_MOT_jpg_compressed.zip"
)
ARCHIVE_BYTES = 29588897110  # Public DAV listing, retained with the source discovery.


class RangeReader(io.RawIOBase):
    """Read-only bounded seek adapter; refuse whole-archive or inconsistent responses."""

    def __init__(self, size, fetch):
        self.size, self.fetch, self.position = size, fetch, 0
        self.transferred, self.requests = 0, []

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError("Unknown seek origin")
        self.position = (0, self.position, self.size)[whence] + offset
        if not 0 <= self.position <= self.size:
            raise ValueError("Archive seek out of bounds")
        return self.position

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if size == 0:
            return b""
        if size > 32 * 1024**2 or self.transferred + size > 96 * 1024**2:
            raise ValueError("Refuse excessive archive transfer")
        start, end = self.position, self.position + size - 1
        data = self.fetch(start, end)
        if len(data) != size:
            raise ValueError("Incomplete archive range")
        self.requests.append(dict(start=start, end=end, bytes=size))
        self.transferred += size
        self.position += size
        return data


def request_range(start, end):
    request = Request(URL, headers={"Range": f"bytes={start}-{end}"})
    with urlopen(request, timeout=45) as response:
        expected = f"bytes {start}-{end}/{ARCHIVE_BYTES}"
        if response.status != 206 or response.headers.get("Content-Range") != expected:
            raise ValueError("Server did not honor the exact bounded range")
        return response.read(end - start + 2)


def select(imported, source):
    manifest = read(imported / "manifest.json")
    document = read(source / "instances_train_objects_in_water.json")
    by_id = {im["id"]: im for im in document["images"]}
    selected = []
    for window in manifest["candidate_windows"]:
        entry = next(g for g in manifest["groups"] if g["group_id"] == window["group_id"])
        path = imported / entry["file"]
        if sha(path) != entry["sha256"]:
            raise ValueError("Changed normalized telemetry")
        group = read(path)
        allowed = set(window["source_frame_ids"])
        frames = [r for r in group["frames"] if r["source_image_id"] in allowed]
        start = frames[0]["time_s"]
        for t in (0, 20, 40, 60, 80, 100):
            frame = min(frames, key=lambda r: (abs(r["time_s"] - start - t), r["source_image_id"]))
            image = by_id[frame["source_image_id"]]
            stem = Path(image["file_name"]).stem
            if not re.fullmatch(r"[0-9]+", stem):
                raise ValueError("Unexpected dataset image name")
            selected.append(
                dict(
                    source_image_id=image["id"],
                    file_stem=stem,
                    source_identity=group["source_identity"],
                    requested_time_s=t,
                    actual_time_s=frame["time_s"] - start,
                    source_width=image["width"],
                    source_height=image["height"],
                    group_id=window["group_id"],
                    segment=window["segment"],
                )
            )
    if len(selected) != 12 or len({r["source_image_id"] for r in selected}) != 12:
        raise ValueError("Expected twelve distinct deterministic images")
    return selected


def run(imported, source, output):
    output.mkdir(parents=True, exist_ok=False)
    source_manifest = read(source / "manifest.json")
    for name, expected in source_manifest["files_sha256"].items():
        if sha(source / name) != expected:
            raise ValueError("Changed official source")
    images = select(imported, source)
    save(
        output / "selection.json",
        dict(
            images=images,
            selection="Nearest source frame at 0,20,40,60,80,100 seconds "
            "in both previously selected continuous training windows; ties use source image ID",
            labels_used_for_selection=False,
            image_content_used_for_selection=False,
            imported=str(imported.resolve()),
            source=str(source.resolve()),
            imported_manifest_sha256=sha(imported / "manifest.json"),
            annotation_sha256=sha(source / "instances_train_objects_in_water.json"),
            source_sha256=sha(Path(__file__)),
            upstream_repository_license_statement="CC0 1.0",
            annotation_license_field="Unknown",
            official_archive_url=URL,
            archive_size=ARCHIVE_BYTES,
        ),
    )
    remote = RangeReader(ARCHIVE_BYTES, request_range)
    entries = []
    with zipfile.ZipFile(remote) as archive:
        names = archive.namelist()
        save(output / "archive_index.json", dict(entries=len(names), sample_names=names[:15]))
        for r in images:
            suffix = "/train/" + r["file_stem"] + ".jpg"
            matches = [n for n in names if n.endswith(suffix)]
            if len(matches) != 1:
                raise ValueError(f"Expected one train image {suffix}; found {len(matches)}")
            info = archive.getinfo(matches[0])
            if info.file_size > 8 * 1024**2:
                raise ValueError("Unexpectedly large image")
            # Reading one member checks its ZIP CRC; no paths from the archive are extracted.
            data = archive.read(info)
            filename = r["file_stem"] + ".jpg"
            (output / filename).write_bytes(data)
            entries.append(
                dict(
                    **r,
                    file=filename,
                    archive_member=info.filename,
                    crc32=info.CRC,
                    sha256=sha(output / filename),
                )
            )
            print(f"Fetched training image {filename} ({len(data)} bytes)", flush=True)
    save(
        output / "manifest.json",
        dict(
            complete=True,
            images=entries,
            selection_sha256=sha(output / "selection.json"),
            actual_downloaded_bytes=remote.transferred,
            requests=remote.requests,
            test_images_downloaded=False,
            used_for_frozen_routing_study=False,
            source_sha256=sha(Path(__file__)),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imported", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.imported, args.source, args.output)
