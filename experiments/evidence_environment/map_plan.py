"""Create a source-grounded map crop and explicitly approximate V4 scenes.

Road/vegetation/building extents come from SAREnv OSM polygons. Geometry is
clipped axis-aligned bounding boxes, heights/targets/path are assumptions. This
is not a photogrammetric reconstruction or a replay of the source SAR mission.
"""

import argparse
import copy
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from experiments.v4_benchmark.scenario import load_scenario

ROOT = Path(__file__).resolve().parents[2]


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def local_point(coordinate, center):
    lon, lat = coordinate[:2]
    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
        raise ValueError("expected WGS84 longitude/latitude")
    radius = 6371008.8
    return [
        radius * math.radians(lon - center[0]) * math.cos(math.radians(center[1])),
        radius * math.radians(lat - center[1]),
    ]


def flatten(value):
    if (
        isinstance(value, list)
        and len(value) >= 2
        and all(isinstance(v, (int, float)) for v in value)
    ):
        yield value
    elif isinstance(value, list):
        for child in value:
            yield from flatten(child)


def crop_features(data, half_extent=120):
    center = data["center_point"]
    features = []
    for feature in data["features"]:
        geometry = feature.get("geometry")
        if not geometry:
            continue
        points = [local_point(p, center) for p in flatten(geometry.get("coordinates", []))]
        if not points:
            continue
        xs, ys = zip(*points, strict=True)
        x0, y0, x1, y1 = (
            max(-half_extent, min(xs)),
            max(-half_extent, min(ys)),
            min(half_extent, max(xs)),
            min(half_extent, max(ys)),
        )
        if x1 - x0 < 0.2 or y1 - y0 < 0.2:
            continue
        features.append(
            dict(
                source_id=feature.get("id"),
                kind=feature["properties"]["feature_type"],
                bbox_m=[x0, y0, x1, y1],
            )
        )
    return center, features


def place_target(preferred, features):
    """Deterministic collision-free probe placement, before rendering/model runs."""
    solids = [
        f["bbox_m"] for f in features if f["kind"] in ("structure", "woodland", "brush", "scrub")
    ]
    offsets = sorted((x * x + y * y, x, y) for x in range(-60, 61, 5) for y in range(-60, 61, 5))
    for _, dx, dy in offsets:
        x, y = preferred[0] + dx, preferred[1] + dy
        if abs(x) > 110 or abs(y) > 110:
            continue
        if not any(x0 - 1 <= x <= x1 + 1 and y0 - 1 <= y <= y1 + 1 for x0, y0, x1, y1 in solids):
            return [x, y, 0]
    raise ValueError("no free ground for probe target; revise crop explicitly")


def generate(source, output, asset, offset_east_m=0, offset_north_m=0):
    manifest = json.loads((source / "sarenv_manifest.json").read_text())
    geo = source / "sarenv_features_1.geojson"
    if hashlib.sha256(geo.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("geographic source checksum mismatch")
    data = json.loads(geo.read_text())
    original_center = data["center_point"]
    data["center_point"] = [
        original_center[0]
        + math.degrees(offset_east_m / (6371008.8 * math.cos(math.radians(original_center[1])))),
        original_center[1] + math.degrees(offset_north_m / 6371008.8),
    ]
    center, features = crop_features(data)
    output.mkdir(parents=True, exist_ok=False)
    write(
        output / "map_crop.json",
        dict(
            center_lon_lat=center,
            coordinate_frame="local east/north meters; spherical approximation",
            extent_m=[240, 240],
            features=features,
            source=manifest,
        ),
    )
    palette = {
        "structure": "#b97359",
        "woodland": "#406b45",
        "brush": "#789356",
        "scrub": "#9dab66",
        "water": "#7badce",
        "road": "#b2b4b6",
        "field": "#d0cb93",
        "drainage": "#7badce",
        "linear": "#999999",
    }
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 840 920">',
        '<rect width="840" height="920" fill="#f4f2e9"/>',
        '<text x="20" y="30" font-size="22">SAREnv map 1 — coarse footprint crop (240 m)</text>',
        '<g transform="translate(420 440) scale(3 -3)">',
    ]
    for f in sorted(features, key=lambda f: f["kind"] == "structure"):
        x0, y0, x1, y1 = f["bbox_m"]
        svg.append(
            f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" '
            f'fill="{palette.get(f["kind"], "#aaaaaa")}" stroke="#ffffff" stroke-width=".2"/>'
        )
    svg += [
        '<path d="M -70 0 L 70 0" stroke="#eb3154" stroke-width="1" stroke-dasharray="3 2"/>',
        "</g>",
        '<text x="20" y="860" font-size="17">Red: assumed transect. Approximate rectangles.</text>',
        '<text x="20" y="890" font-size="15">SAREnv / OpenStreetMap contributors.</text>',
        "</svg>",
    ]
    (output / "map_preview.svg").write_text("\n".join(svg))
    base = json.loads((ROOT / "experiments/v4_benchmark/person_probe_scenario.json").read_text())
    contract = json.loads((ROOT / "experiments/v4_benchmark/audit_contract.json").read_text())
    contract.update(
        contract_id="EVIDENCE_MAP_DIAGNOSTIC", deadline_s=70.0, min_final_battery_frac=0.1
    )
    write(output / "contract.json", contract)
    # Preserve raw field outcomes separately; these rounded design points are not an exact replay.
    cases = []
    selected = [
        f
        for f in features
        if f["kind"] in ("structure", "woodland", "brush", "scrub", "field", "water", "road")
    ]
    expanded = []
    for feature in selected:
        if feature["kind"] != "woodland":
            expanded.append(feature)
            continue
        x0, y0, x1, y1 = feature["bbox_m"]
        for x in range(math.ceil((x0 + 3) / 30) * 30, math.floor((x1 - 3) / 30) * 30 + 1, 30):
            for y in range(math.ceil((y0 + 3) / 30) * 30, math.floor((y1 - 3) / 30) * 30 + 1, 30):
                expanded.append(
                    dict(
                        source_id=f"{feature['source_id']}_tree_{x}_{y}",
                        kind="woodland",
                        bbox_m=[x - 3, y - 3, x + 3, y + 3],
                    )
                )
    selected = expanded
    patch_id = hashlib.sha256(json.dumps(center).encode()).hexdigest()[:10]
    if len(selected) > 120:
        raise ValueError("map crop exceeds simulator labels; choose a smaller crop explicitly")
    for altitude in (35, 45, 80):
        for pitch in (45, 90):
            scenario = copy.deepcopy(base)
            scenario.update(
                scenario_id=f"EVIDENCE_MAP_{patch_id}_H{altitude}_P{pitch}",
                contract_path="../contract.json",
            )
            scenario["mission"].update(duration_s=60.0, warmup_s=2.0, observation_interval_s=0.5)
            targets = []
            for i, (x, y) in enumerate(((0, -15), (40, 15), (80, 0))):
                person = copy.deepcopy(base["world"]["objects"][0])
                person.update(
                    object_id=f"placed_person_{i}",
                    label=i + 1,
                    position_m=place_target((x, y), selected),
                )
                person["mesh"]["manifest"] = str(asset.resolve())
                targets.append(person)
            objects = list(targets)
            for f in selected:
                x0, y0, x1, y1 = f["bbox_m"]
                height = {"structure": 5.0, "woodland": 8.0, "brush": 1.0, "scrub": 0.5}.get(
                    f["kind"], 0.02
                )
                color = palette[f["kind"]].lstrip("#")
                rgb = [int(color[i : i + 2], 16) / 255 for i in (0, 2, 4)]
                objects.append(
                    dict(
                        object_id="map_" + str(f["source_id"]),
                        label=len(objects) + 1,
                        position_m=[(x0 + x1) / 2, (y0 + y1) / 2, height / 2],
                        size_m=[x1 - x0, y1 - y0, height],
                        color_rgba=[*rgb, 1],
                        is_target=False,
                        shape="box",
                    )
                )
                if f["kind"] == "woodland":
                    objects[-1].update(
                        shape="ellipsoid",
                        position_m=[(x0 + x1) / 2, (y0 + y1) / 2, 6],
                        size_m=[6, 6, 4],
                    )
                    objects.append(
                        dict(
                            object_id="trunk_" + str(f["source_id"]),
                            label=len(objects) + 1,
                            position_m=[(x0 + x1) / 2, (y0 + y1) / 2, 2],
                            size_m=[0.4, 0.4, 4],
                            color_rgba=[0.3, 0.2, 0.1, 1],
                            is_target=False,
                            shape="cylinder",
                        )
                    )
            scenario["world"].update(
                objects=objects,
                initial_position_m=[-70, 0, altitude],
                camera_pitch_rad=math.radians(pitch),
                ground_size_m=[260, 260, 0.1],
                max_speed_mps=2.5,
                waypoints=[dict(start_s=0.0, position_m=[70, 0, altitude])],
            )
            folder = output / f"h{altitude}_p{pitch}"
            folder.mkdir()
            write(folder / "scenario.json", scenario)
            loaded = load_scenario(folder / "scenario.json")
            cases.append(
                dict(
                    case_id=folder.name,
                    scenario=str((folder / "scenario.json").resolve()),
                    recording_identity=loaded.recording_identity,
                    partition="development_diagnostic",
                    altitude_m=altitude,
                    camera_depression_deg=pitch,
                )
            )
    write(
        output / "plan.json",
        dict(
            cases=cases,
            feature_counts=dict(Counter(f["kind"] for f in features)),
            source=manifest,
            evidence_sources=[
                "https://zenodo.org/records/20643214",
                "https://github.com/namurproject/SAREnv",
            ],
            design_assumptions=dict(
                heights="structure 5 m, woodland 8 m, brush 1 m, scrub .5 m",
                targets="3 deterministic collision-free synthetic placements; shared appearance",
                speed_mps=2.5,
                duration_s=60.0,
                power="unchanged configured simulator value, not CMU watts",
                map="clipped bounding boxes; no measured terrain elevation",
                sensor="V4 320x240, not the source drone camera",
            ),
            evaluation_ready=False,
            independent_scene_count=1,
            family_id="sarenv_map_1",
            patch_id=patch_id,
            woodland_assumption="30 m grid of 8 m trees with 6 m canopy; positions not measured",
        ),
    )
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--asset", type=Path, default=Path("results/v4_assets/standing_person/manifest.json")
    )
    parser.add_argument("--offset-east-m", type=float, default=0)
    parser.add_argument("--offset-north-m", type=float, default=0)
    args = parser.parse_args()
    print(
        json.dumps(
            dict(
                cases=len(
                    generate(
                        args.source,
                        args.output,
                        args.asset,
                        args.offset_east_m,
                        args.offset_north_m,
                    )
                )
            )
        )
    )


if __name__ == "__main__":
    main()
