"""Immutable Depot originals plus a verified standalone collision derivative."""

import argparse
import copy
import json
import shutil
from pathlib import Path

from experiments.lake_missions.assets import save, sha
from experiments.lake_missions.assets import verify as verify_files
from experiments.mission_environment_transfer.assets import verify as verify_source
from experiments.mission_environment_transfer_v3.collision import export, validate


def prepare(source, output):
    original = verify_source(source)
    if output.exists():
        raise FileExistsError("Preserve all derivative preparation attempts")
    shutil.copytree(source, output)
    shutil.copyfile(source / "manifest.json", output / "source_manifest.json")
    (output / "derived").mkdir()
    mesh = output / "derived/depot_collision.obj"
    derivative = export(output, mesh)
    derivative["validation"] = validate(output, mesh)
    manifest = copy.deepcopy(original)
    manifest.update(
        schema_version="independent_depot_collision_v3",
        collision=derivative,
        source_assets=str(source.resolve()),
        source_manifest_sha256=sha(source / "manifest.json"),
        collision_derivative_relative_path=str(mesh.relative_to(output)),
        source_model_files_modified=False,
    )
    manifest["models"][0]["files"][str(mesh.relative_to(output))] = derivative["sha256"]
    save(output / "manifest.json", manifest)
    with (output / "ATTRIBUTION.txt").open("a") as stream:
        stream.write(
            "\nCollision derivative: source visual triangles transformed to meters/world "
            "coordinates and exported as one standalone OBJ. Original visuals and roof "
            "are unchanged. This avoids Gazebo physics ignoring submesh selection.\n"
        )
    return verify(output)


def verify(folder):
    folder = Path(folder)
    manifest = verify_files(folder)
    if manifest["schema_version"] != "independent_depot_collision_v3":
        raise ValueError("Standalone Depot collision derivative required")
    if sha(folder / "source_manifest.json") != manifest["source_manifest_sha256"]:
        raise ValueError("Changed original source manifest")
    original = json.loads((folder / "source_manifest.json").read_text())
    for model in original["models"]:
        for name, expected in model["files"].items():
            if manifest["models"][0]["files"].get(name) != expected:
                raise ValueError("Original model source hash changed")
    collision = manifest["collision"]
    if sha(folder / manifest["collision_derivative_relative_path"]) != collision["sha256"]:
        raise ValueError("Changed standalone collision mesh")
    for name, key in (
        ("collision.py", "generator_sha256"),
        ("geometry.py", "geometry_parser_sha256"),
    ):
        if sha(Path(__file__).with_name(name)) != collision[key]:
            raise ValueError("Changed collision derivative implementation")
    if not collision["validation"]["complete"]:
        raise ValueError("Unvalidated collision derivative")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    prepare(a.source, a.output)
