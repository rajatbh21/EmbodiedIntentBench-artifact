"""Normalize public training telemetry without accessing object labels or images."""

import argparse
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.v4_benchmark.scenario import digest

MAX_CONTINUOUS_GAP_S = 1.0
EARTH_RADIUS_M = 6371008.8


def number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Finite telemetry required")
    return float(value)


def coordinate(value, reference, positive, negative, limit):
    value = number(value)
    if (
        reference not in (positive, negative)
        or abs(value) > limit
        or (value < 0 and reference == positive)
    ):
        raise ValueError("Inconsistent GPS coordinate/reference")
    return abs(value) * (1 if reference == positive else -1)


def normalize(document):
    """Preserve every frame and discontinuity; never interpolate across missing data."""
    groups = defaultdict(list)
    seen = set()
    known = {v["id"] for v in document["videos"]}
    for image in document["images"]:
        if image["id"] in seen:
            raise ValueError("Duplicate source image identifier")
        seen.add(image["id"])
        s = image["source"]
        key = tuple(s[n] for n in ("drone", "folder_name", "video"))
        if any(not isinstance(v, str) or not v for v in key):
            raise ValueError("Complete source identity required")
        groups[key].append(image)
    output = []
    for key, images in sorted(groups.items()):
        images = sorted(images, key=lambda r: (r["source"]["frame_no"], r["id"]))
        origin_time = datetime.fromisoformat(images[0]["date_time"])
        first = images[0]["meta"]
        lat0 = coordinate(first["gps_latitude"], first["gps_latitude_ref"], "N", "S", 90)
        lon0 = coordinate(first["gps_longitude"], first["gps_longitude_ref"], "E", "W", 180)
        altitude0 = number(first["altitude"])
        rows = []
        segments = []
        current = []
        previous = None
        for im in images:
            meta = im["meta"]
            t = (datetime.fromisoformat(im["date_time"]) - origin_time).total_seconds()
            lat = coordinate(meta["gps_latitude"], meta["gps_latitude_ref"], "N", "S", 90)
            lon = coordinate(meta["gps_longitude"], meta["gps_longitude_ref"], "E", "W", 180)
            alt = number(meta["altitude"])
            delta_lon = (lon - lon0 + 180) % 360 - 180
            point = [
                EARTH_RADIUS_M * math.radians(delta_lon) * math.cos(math.radians(lat0)),
                EARTH_RADIUS_M * math.radians(lat - lat0),
                alt - altitude0,
            ]
            issue = None
            if previous is not None:
                gap = t - previous["time_s"]
                if gap <= 0:
                    issue = "nonincreasing_timestamp"
                elif gap > MAX_CONTINUOUS_GAP_S:
                    issue = "missing_time_interval"
                elif im["source"]["frame_no"] <= previous["source_frame_no"]:
                    issue = "nonincreasing_source_frame"
            if issue:
                segments.append(current)
                current = []
            row = dict(
                source_image_id=im["id"],
                source_video_id=im["video_id"],
                source_frame_no=im["source"]["frame_no"],
                recorded_timestamp=im["date_time"],
                time_s=t,
                latitude_deg=lat,
                longitude_deg=lon,
                recorded_altitude_m=alt,
                local_east_north_altitude_delta_m=point,
                gimbal_pitch_deg=number(meta["gimbal_pitch"]),
                compass_heading_deg=number(meta["compass_heading"]),
                gimbal_heading_deg=number(meta["gimbal_heading"]),
                reported_speed_mps=number(meta["speed"]),
                reported_velocity_xyz_mps=[number(meta[k]) for k in ("xspeed", "yspeed", "zspeed")],
                discontinuity_before=issue,
                source_video_listed=im["video_id"] in known,
            )
            rows.append(row)
            current.append(len(rows) - 1)
            previous = row
        if current:
            segments.append(current)
        group_id = digest(key)[:16]
        output.append(
            dict(
                group_id=group_id,
                source_identity=dict(zip(("drone", "folder_name", "video"), key, strict=True)),
                origin=dict(
                    latitude_deg=lat0,
                    longitude_deg=lon0,
                    recorded_altitude_m=altitude0,
                    recorded_timestamp=images[0]["date_time"],
                ),
                frames=rows,
                segments=[
                    dict(indices=s, duration_s=rows[s[-1]]["time_s"] - rows[s[0]]["time_s"])
                    for s in segments
                ],
            )
        )
    if sum(len(g["frames"]) for g in output) != len(document["images"]):
        raise ValueError("Source frames lost")
    return output


def run(source, output):
    if output.exists():
        raise FileExistsError("Preserve telemetry exports")
    manifest = read(source / "manifest.json")
    if manifest["split"] != "official training only" or manifest["test_split_downloaded"]:
        raise ValueError("Training-only source required")
    for name, expected in manifest["files_sha256"].items():
        if sha(source / name) != expected:
            raise ValueError("Changed source metadata")
    doc = read(source / "instances_train_objects_in_water.json")
    groups = normalize(doc)
    output.mkdir(parents=True)
    inventory = []
    windows = []
    for group in groups:
        path = output / (group["group_id"] + ".json")
        save(path, group)
        rows = group["frames"]
        inventory.append(
            dict(
                group_id=group["group_id"],
                source_identity=group["source_identity"],
                frames=len(rows),
                segments=len(group["segments"]),
                missing_video_metadata_frames=sum(not r["source_video_listed"] for r in rows),
                nonincreasing_timestamp_boundaries=sum(
                    r["discontinuity_before"] == "nonincreasing_timestamp" for r in rows
                ),
                gap_boundaries=sum(
                    r["discontinuity_before"] == "missing_time_interval" for r in rows
                ),
                altitude_range_m=[
                    min(r["recorded_altitude_m"] for r in rows),
                    max(r["recorded_altitude_m"] for r in rows),
                ],
                gimbal_pitch_range_deg=[
                    min(r["gimbal_pitch_deg"] for r in rows),
                    max(r["gimbal_pitch_deg"] for r in rows),
                ],
                reported_speed_range_mps=[
                    min(r["reported_speed_mps"] for r in rows),
                    max(r["reported_speed_mps"] for r in rows),
                ],
                file=path.name,
                sha256=sha(path),
            )
        )
        # One deterministic 120-second candidate per continuous segment. No labels or model outputs.
        for i, segment in enumerate(group["segments"]):
            if segment["duration_s"] < 120:
                continue
            selected = [
                rows[j]
                for j in segment["indices"]
                if rows[j]["time_s"] <= rows[segment["indices"][0]]["time_s"] + 120
            ]
            windows.append(
                dict(
                    group_id=group["group_id"],
                    segment=i,
                    source_frame_ids=[r["source_image_id"] for r in selected],
                    duration_s=selected[-1]["time_s"] - selected[0]["time_s"],
                    selection=(
                        "First 120 seconds of every continuous segment lasting at least 120 seconds"
                    ),
                )
            )
    result = dict(
        schema_version="real_flight_telemetry_v1",
        complete=True,
        source_manifest_sha256=sha(source / "manifest.json"),
        source=str(source.resolve()),
        implementation_sha256=sha(Path(__file__)),
        groups=inventory,
        candidate_windows=windows,
        source_frames=len(doc["images"]),
        source_groups=len(groups),
        listed_videos=len(doc["videos"]),
        source_annotation_license_field=doc["licenses"],
        upstream_repository_license_statement=manifest["dataset_license_as_stated_upstream"],
        object_labels_used=False,
        source_contains_object_annotations=True,
        imagery_accessed=False,
        simulator_replayed=False,
        source_authored_mission_contracts_available=False,
        used_for_frozen_coverage_evaluation=False,
        projection=(
            "Local spherical east/north offsets; "
            "vertical is recorded altitude difference, not geodetic ENU"
        ),
        altitude_datum="Unverified; retain recorded value and do not infer terrain clearance",
        timestamp="Image date_time within each source video; no inferred absolute timezone",
        discontinuities=(
            "All records retained; split on nonincreasing timestamps, "
            "frame order or gaps over one second"
        ),
        candidate_contracts=(
            "None: deadlines, repeated-evidence rules and energy limits are not in this metadata"
        ),
    )
    save(output / "manifest.json", result)
    print(
        {
            k: v
            for k, v in result.items()
            if k in ("source_frames", "source_groups", "listed_videos")
        },
        flush=True,
    )
    print(
        dict(
            candidate_windows=len(windows),
            nonincreasing_boundaries=sum(
                i["nonincreasing_timestamp_boundaries"] for i in inventory
            ),
            gap_boundaries=sum(i["gap_boundaries"] for i in inventory),
            unlisted_frames=sum(i["missing_video_metadata_frames"] for i in inventory),
        ),
        flush=True,
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    run(a.source, a.output)
