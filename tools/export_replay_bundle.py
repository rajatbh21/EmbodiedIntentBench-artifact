"""Maintainer utility: export portable replay inputs from a historical study.

Pass the archived study directory explicitly. Existing scientific source files
are read only. The output has new hashes for rewritten metadata, and retains
the exact recorded frame bytes, prediction masks, costs and reference outcomes.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

from replay import read, save, sha


def portable_metadata(value, source_root):
    if isinstance(value, dict):
        return {portable_metadata(k, source_root): portable_metadata(v, source_root) for k, v in value.items()}
    if isinstance(value, list):
        return [portable_metadata(v, source_root) for v in value]
    if isinstance(value, str) and Path(value).is_absolute():
        path = Path(value)
        return "historical/" + (str(path.relative_to(source_root)) if path.is_relative_to(source_root) else path.name)
    return value


def export(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError("Preserve an existing exported bundle")
    destination.mkdir(parents=True)
    old = read(source / "plan.json")
    source_root = source.parents[1]
    keep = ("actions", "policy_costs", "resources", "cases", "grid", "expected_runs", "intent_schedules",
            "report_windows", "scope", "outcome_awareness", "semantics", "profiles", "calibration_path", "battery_design")
    plan = {k: copy.deepcopy(old[k]) for k in keep}
    plan.update(schema_version="portable_perception_replay_v1", assets="terrain", recordings={})
    plan["cases"] = {key: {k: v for k, v in case.items() if k not in (
        "geometry_preflight", "geometry_preflight_sha256", "scenario", "scenario_sha256")}
        for key, case in plan["cases"].items()}
    for relative in [*old["profiles"].values(), old["calibration_path"]]:
        save(destination / relative, portable_metadata(read(source / relative), source_root))
    bindings = read(source / "native_bindings.json")
    for binding in bindings["bindings"]:
        original = Path(binding["recording_path"])
        if sha(original) != binding["recording_sha256"]:
            raise ValueError("Historical recording changed")
        recording = read(original)
        selected = {f["capture_id"] for f in binding["frames"]}
        # Keep all frame metadata and the complete acquisition clock. The final
        # three settle frames are excluded from evaluation by the frozen design.
        recording["world"] = {"camera_pitch_rad": recording["world"]["camera_pitch_rad"],
            "objects": [{k: v for k, v in o.items() if k != "mesh"} for o in recording["world"]["objects"]]}
        recording["provenance"] = {k: v for k, v in recording["provenance"].items() if k != "source"}
        recording["release_derivation"] = dict(original_recording_sha256=sha(original),
            metadata_changes="Removed machine paths, render assets and acquisition source descriptors. Recorded pixels, labels, poses, clocks and IDs are unchanged.",
            included_capture_ids=sorted(selected), excluded_settle_capture_ids=[i for i in range(len(recording["frames"])) if i not in selected])
        relative = f"recordings/{binding['case_id']}/recording.json"
        save(destination / relative, recording)
        for frame in binding["frames"]:
            path = original.parent / frame["source_file"]
            if sha(path) != frame["source_sha256"]:
                raise ValueError("Historical frame changed")
            shutil.copy2(path, (destination / relative).parent / frame["source_file"])
        binding["recording_path"] = relative
        binding["recording_sha256"] = sha(destination / relative)
        plan["recordings"][binding["case_id"]] = relative
        prediction_source = source / "predictions" / binding["case_id"]
        prediction_destination = destination / "predictions" / binding["case_id"]
        prediction_destination.mkdir(parents=True)
        manifest = read(prediction_source / "manifest.json")
        manifest["historical_recording_sha256"] = manifest["recording_sha256"]
        manifest["recording_sha256"] = binding["recording_sha256"]
        for model in manifest["models"].values():
            for prediction in model["predictions"].values():
                path = prediction_source / prediction["file"]
                if sha(path) != prediction["sha256"]:
                    raise ValueError("Historical prediction changed")
                shutil.copy2(path, prediction_destination / prediction["file"])
        save(prediction_destination / "manifest.json", manifest)
    save(destination / "native_bindings.json", bindings)
    terrain = Path(old["assets"]) / "harmonic_terrain"
    sdf = ET.parse(terrain / "model.sdf")
    relative_heightmap = sdf.getroot().findtext("model/link/visual/geometry/heightmap/uri")
    terrain_destination = destination / "terrain/harmonic_terrain"
    (terrain_destination / relative_heightmap).parent.mkdir(parents=True)
    shutil.copy2(terrain / "model.sdf", terrain_destination / "model.sdf")
    shutil.copy2(terrain / relative_heightmap, terrain_destination / relative_heightmap)
    attribution = [{k: v for k, v in model.items() if k != "files"}
        for model in read(Path(old["assets"]) / "manifest.json")["models"]]
    attribution.append(dict(name="Standing person", creator="Marina Kollmitz", owner="OpenRobotics",
        source="https://fuel.gazebosim.org/1.0/OpenRobotics/models/Standing%20person/3",
        license="CC0-1.0", license_url="https://creativecommons.org/publicdomain/zero/1.0/"))
    save(destination / "asset_attribution.json", attribution)
    save(destination / "study.json", plan)
    save(destination / "expected_rows.json", read(source / "rows.json"))
    (destination / "ATTRIBUTION.md").write_text('''# Replay data attribution

The RGB frames are synthetic Gazebo renderings. They depict no real people.
The heightmap and rendered scenery derive from OpenRobotics Fuel assets:

| Asset | Version | License |
| --- | --- | --- |
| [Harmonic Terrain](https://fuel.gazebosim.org/1.0/OpenRobotics/models/Harmonic%20Terrain) | 1 | CC BY 4.0 |
| [Harmonic Terrain Objects](https://fuel.gazebosim.org/1.0/OpenRobotics/models/Harmonic%20Terrain%20Objects) | 1 | CC BY 4.0 |
| [Lake House](https://fuel.gazebosim.org/1.0/OpenRobotics/models/Lake%20House) | 3 | CC BY 4.0 |
| [Coast Waves 2](https://fuel.gazebosim.org/1.0/OpenRobotics/models/Coast%20Waves%202) | 3 | CC BY 4.0 |
| [Standing person](https://fuel.gazebosim.org/1.0/OpenRobotics/models/Standing%20person/3), Marina Kollmitz | 3 | CC0 1.0 |

License links: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).
The replay renderings are distributed under CC BY 4.0 with this attribution.
Changes include object placement, camera paths, terrain collision alignment and
rendering. The included terrain raster is unchanged; the source SDF aligns the
collision vertical scale with the visual scale. This is a replay evaluation
map, not the complete set of assets for launching a new Gazebo world.

The bundle contains 108 evaluated RGB/label frames, original body poses, 324
native model prediction files, frozen model-cost profiles and all 648 reference
result rows. The recorded flight lasts 42 seconds, while evaluation uses the
first 36 seconds. The final three settle captures per recording are not routed
and their image files are omitted. Their timestamps remain in the metadata.

Frame and prediction bytes are unchanged. Machine-specific paths and unused
render-asset descriptors were removed from recording metadata. New metadata
hashes bind this release; the release does not claim that its rewritten metadata
or renamed source files have the historical hashes. The original recording
hash is retained as provenance. Verification checks every bundled file and
every accessed prediction mask. Replays recompute decisions and mission metrics
and compare all scientific fields against the frozen references.

Model weights are not required for deterministic replay. Prediction masks were
generated by Mask R-CNN, YOLO11s and CABiNet. Their model licenses and native
inference setup remain separate from this replay-data license.
''')
    files = {str(p.relative_to(destination)): dict(sha256=sha(p), bytes=p.stat().st_size)
             for p in sorted(destination.rglob("*")) if p.is_file()}
    save(destination / "manifest.json", dict(schema_version="portable_perception_replay_v1", files=files,
        byte_count=sum(v["bytes"] for v in files.values()), original_study_plan_sha256=sha(source / "plan.json"),
        original_reference_rows_sha256=sha(source / "rows.json")))
    print(json.dumps(dict(files=len(files), bytes=sum(v["bytes"] for v in files.values()))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    export(args.source, args.destination)
