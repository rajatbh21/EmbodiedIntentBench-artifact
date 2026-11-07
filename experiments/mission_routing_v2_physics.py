"""Derive a collision-aligned lake asset bundle without modifying downloaded originals.

The source Fuel heightmap has visual z-size 201.66 m but collision z-size 600 m.
This creates invisible obstacles above the displayed terrain. Both heightfields
must use the same raster, dimensions and world transform before low flight tests.
"""

from __future__ import annotations

import argparse
import copy
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from experiments.lake_causal_router import split_fingerprint
from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.lake_missions.assets import verify as verify_assets
from experiments.mission_routing_v2.flights import load_scenario, split_check, verify


def terrain_geometry(path):
    root = ET.parse(path).getroot()
    model = root.find("model")
    link = model.find("link")
    for parent in (model, link, link.find("visual")):
        if any(float(x) for x in parent.findtext("pose", "0 0 0 0 0 0").split()):
            raise ValueError("Only identity parent transforms are supported")
    visual = link.find("visual/geometry/heightmap")
    collision = link.find("collision/geometry/heightmap")
    pose = list(map(float, link.find("collision").findtext("pose", "0 0 0 0 0 0").split()))
    if any(pose[3:]):
        raise ValueError("Rotated heightfield collision requires separate validation")
    cpos = list(map(float, collision.findtext("pos", "0 0 0").split()))
    return dict(
        visual_size=list(map(float, visual.findtext("size").split())),
        collision_size=list(map(float, collision.findtext("size").split())),
        visual_position=list(map(float, visual.findtext("pos", "0 0 0").split())),
        collision_position=[a + b for a, b in zip(pose[:3], cpos, strict=True)],
        visual_uri=visual.findtext("uri"),
        collision_uri=collision.findtext("uri"),
    )


def validate_alignment(folder):
    verify_assets(folder)
    geometry = terrain_geometry(folder / "harmonic_terrain/model.sdf")
    for key in ("size", "position", "uri"):
        if geometry["visual_" + key] != geometry["collision_" + key]:
            raise ValueError("Visual/collision heightmap mismatch: " + key)
    return geometry


def derive(original, output):
    manifest = verify_assets(original)
    before = terrain_geometry(original / "harmonic_terrain/model.sdf")
    if output.exists():
        raise FileExistsError(output)
    shutil.copytree(original, output)
    path = output / "harmonic_terrain/model.sdf"
    tree = ET.parse(path)
    link = tree.getroot().find("model/link")
    visual = link.find("visual/geometry/heightmap")
    collision = link.find("collision/geometry/heightmap")
    collision.find("size").text = visual.findtext("size")
    collision.find("uri").text = visual.findtext("uri")
    extra = collision.find("pos")
    if extra is not None:
        collision.remove(extra)
    parent = link.find("collision")
    pose = parent.find("pose")
    if pose is None:
        pose = ET.SubElement(parent, "pose")
    pose.text = visual.findtext("pos", "0 0 0") + " 0 0 0"
    ET.indent(tree)
    tree.write(path, encoding="unicode", xml_declaration=True)
    relative = "harmonic_terrain/model.sdf"
    derived = copy.deepcopy(manifest)
    for model in derived["models"]:
        if relative in model["files"]:
            model["files"][relative] = sha(path)
    derived["derivation"] = dict(
        source_manifest_sha256=sha(original / "manifest.json"),
        source_directory=str(original.resolve()),
        source_sdf_sha256=sha(original / relative),
        changed_files=[relative],
        modification=(
            "Align collision heightmap raster, dimensions and transform to unchanged visual terrain"
        ),
        geometry_before=before,
        geometry_after=terrain_geometry(path),
        visual_scenery_is_new=False,
        implementation_sha256=sha(Path(__file__)),
    )
    save(output / "manifest.json", derived)
    with (output / "ATTRIBUTION.txt").open("a") as stream:
        stream.write(
            "\nDerived benchmark bundle: collision heightmap aligned to visual terrain. "
            "Visual geometry, textures, downloaded archives and source attribution preserved.\n"
        )
    validate_alignment(output)
    verify_assets(original)
    print(derived["derivation"], flush=True)


def prepare(original, assets, output):
    p = verify(original)
    geometry = validate_alignment(assets)
    output.mkdir(parents=True, exist_ok=False)
    for name in p["frozen_files"]:
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original / name, destination)
    revised = copy.deepcopy(p)
    revised["assets"] = str(assets.resolve())
    profile = read(Path(p["screen"]) / "calibration.json")
    for flight, case in revised["cases"].items():
        scenario = load_scenario(
            output / flight / "rapid_survey/scenario.json", case["environment"], assets
        )
        case["lineage"] = split_fingerprint(scenario, scenario.scenery_sha256)
        case["split"] = split_check(profile["training_lineage"], case["lineage"])
    revised["physics_alignment"] = dict(
        original_study=str(original.resolve()),
        original_protocol_sha256=sha(original / "protocol.json"),
        derived_assets_manifest_sha256=sha(assets / "manifest.json"),
        original_assets_manifest_sha256=sha(Path(p["assets"]) / "manifest.json"),
        geometry=geometry,
        visual_scene_unchanged=True,
        unseen_map_claim=False,
        policies_changed=False,
        model_pool_changed=False,
        intent_thresholds_changed=False,
        route_and_target_layout_unchanged=True,
        implementation_sha256=sha(Path(__file__)),
    )
    save(output / "protocol.json", revised)
    print(
        "Prepared physics-aligned cohort; require route validation before model inference",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("derive", "prepare", "check"))
    parser.add_argument("--original", type=Path)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "derive":
        derive(args.original, args.output)
    elif args.mode == "prepare":
        prepare(args.original, args.assets, args.output)
    else:
        print(validate_alignment(args.output))
