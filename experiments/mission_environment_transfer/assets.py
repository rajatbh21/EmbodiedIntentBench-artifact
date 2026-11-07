"""Preserve the versioned CC BY 4.0 OpenRobotics Depot source unchanged."""

import argparse
import json
import shutil
import stat
import zipfile
from pathlib import Path

from experiments.lake_missions.assets import save, sha
from experiments.lake_missions.assets import verify as verify_files

SOURCE = "https://fuel.gazebosim.org/1.0/OpenRobotics/models/Depot"
VERSION = 6
DOWNLOAD = SOURCE + "/6/Depot.zip"
ARCHIVE_BYTES = 81963314


def prepare(source, output):
    metadata = json.loads((source / "metadata.json").read_text())
    if (
        metadata["name"] != "Depot"
        or metadata["owner"] != "OpenRobotics"
        or metadata["version"] != VERSION
        or metadata["license_name"] != "Creative Commons Attribution 4.0 International"
        or (source / "Depot.zip").stat().st_size != ARCHIVE_BYTES
    ):
        raise ValueError("Unexpected source asset/version/license/length")
    output.mkdir(parents=True, exist_ok=False)
    dest = output / "depot"
    dest.mkdir()
    shutil.copyfile(source / "Depot.zip", dest / "source.zip")
    shutil.copyfile(source / "metadata.json", dest / "fuel_metadata.json")
    with zipfile.ZipFile(dest / "source.zip") as archive:
        if sum(i.file_size for i in archive.infolist()) > 300_000_000:
            raise ValueError("Unexpected expanded asset size")
        for item in archive.infolist():
            if not (dest / item.filename).resolve().is_relative_to(dest.resolve()) or stat.S_ISLNK(
                item.external_attr >> 16
            ):
                raise ValueError("Unsafe archive member")
        archive.extractall(dest)
    entry = dict(
        name="Depot",
        owner="OpenRobotics",
        version=VERSION,
        directory="depot",
        source=SOURCE,
        download=DOWNLOAD,
        license=metadata["license_name"],
        license_url=metadata["license_url"],
        files={str(p.relative_to(output)): sha(p) for p in sorted(dest.rglob("*")) if p.is_file()},
    )
    save(
        output / "manifest.json",
        dict(
            schema_version="independent_depot_v1",
            models=[entry],
            map_family="openrobotics_depot",
            original_files_modified=False,
            source_archive_sha256=sha(dest / "source.zip"),
        ),
    )
    (output / "ATTRIBUTION.txt").write_text(
        "Depot v6 by OpenRobotics, Gazebo Fuel. Creative Commons Attribution 4.0.\n"
        + SOURCE
        + "\nhttps://creativecommons.org/licenses/by/4.0/\n"
        "Original archive and files preserved unchanged. Generated worlds freeze fans, "
        "resolve resource paths, and add collision copies of visible meshes. "
        "The roof and every original visual are retained.\n"
    )
    return verify(output)


def verify(folder):
    manifest = verify_files(folder)
    if manifest["schema_version"] != "independent_depot_v1" or len(manifest["models"]) != 1:
        raise ValueError("Independent Depot asset bundle required")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    prepare(a.source, a.output)
