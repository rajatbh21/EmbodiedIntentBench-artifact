"""Prepare pinned simulator assets and capture one fresh 42-second inspection.

Preparation may download public assets; capture never downloads. All paths in
the shipped scenario are portable. Runtime paths and provenance are bound to the
current checkout, without importing historical experiment plans or recordings.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import urllib.parse
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "simulation"
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def local(root, name):
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError("Asset path escapes its declared directory.")
    return path


def checked(path, expected):
    if not Path(path).is_file() or sha(path) != expected:
        raise ValueError(f"Missing or changed asset: {Path(path).name}")


def download(url, destination, maximum_bytes):
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    request = urllib.request.Request(url, headers={"Accept-Encoding": "identity"})
    total = 0
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            total += len(chunk)
            if total > maximum_bytes:
                raise ValueError("Downloaded asset exceeds its declared size limit.")
            output.write(chunk)
    partial.replace(destination)


def prepare(assets, archive_directory=None, person_source=None, offline=False):
    lock = read(DATA / "assets.lock.json")
    checked(DATA / "scenario.json", lock["scenario_sha256"])
    assets.mkdir(parents=True, exist_ok=True)
    models = []
    for entry in lock["models"]:
        folder = assets / "scenery" / entry["directory"]
        folder.mkdir(parents=True, exist_ok=True)
        # A completed preparation is still reverified, including its SDF override.
        expected_files = dict(entry["files"])
        for relative, override in lock["overrides"].items():
            if relative.startswith(entry["directory"] + "/"):
                expected_files[relative.split("/", 1)[1]] = override["sha256"]
        complete = all((folder / name).is_file() and sha(folder / name) == expected
                       for name, expected in expected_files.items())
        if not complete:
            archive = (archive_directory / entry["directory"] / "source.zip"
                       if archive_directory else assets / "archives" / (entry["directory"] + ".zip"))
            if not archive.is_file():
                if offline or archive_directory:
                    raise FileNotFoundError("A pinned scenery archive is absent; use prepare with network access.")
                print("Downloading " + entry["name"], flush=True)
                download(entry["download"], archive, entry["archive_bytes"])
            checked(archive, entry["archive_sha256"])
            with zipfile.ZipFile(archive) as source:
                if sum(item.file_size for item in source.infolist()) > 1024**3:
                    raise ValueError("Scenery archive expands beyond one GiB.")
                for item in source.infolist():
                    local(folder, item.filename)
                    if stat.S_ISLNK(item.external_attr >> 16):
                        raise ValueError("Symlink in scenery archive.")
                for name, expected in entry["files"].items():
                    destination = local(folder, name)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with source.open(name) as stream, destination.open("wb") as output:
                        shutil.copyfileobj(stream, output)
                    checked(destination, expected)
            for relative, override in lock["overrides"].items():
                if relative.startswith(entry["directory"] + "/"):
                    source = DATA / override["file"]
                    checked(source, override["sha256"])
                    shutil.copyfile(source, local(assets / "scenery", relative))
        model = {key: value for key, value in entry.items()
                 if key not in ("archive_sha256", "archive_bytes", "files")}
        model["files"] = {entry["directory"] + "/" + name: expected
                          for name, expected in expected_files.items()}
        models.append(model)
    save(assets / "scenery" / "manifest.json", {
        "schema_version": "1.0", "models": models,
        "derivation": "Collision heightmap aligned to visual terrain; other upstream files unchanged.",
    })

    person = lock["person"]
    folder = assets / "standing_person"
    for name, expected in person["files"].items():
        destination = local(folder, name)
        if destination.is_file() and sha(destination) == expected:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if name == "source_metadata.json":
            shutil.copyfile(DATA / "standing_person_metadata.json", destination)
        elif person_source:
            source = local(person_source, name)
            checked(source, expected)
            shutil.copyfile(source, destination)
        elif offline:
            raise FileNotFoundError("A standing-person asset is absent; prepare it before offline use.")
        else:
            print("Downloading standing-person " + name, flush=True)
            url = person["source_url"] + "/files/" + urllib.parse.quote(name)
            download(url, destination, 64 * 1024 * 1024)
        checked(destination, expected)
    save(folder / "manifest.json", person)
    for name, expected in lock["derived_files"].items():
        source = DATA / "derived" / name
        checked(source, expected)
        destination = assets / "derived" / name
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, destination)
    save(assets / "prepared.json", {"lock_sha256": sha(DATA / "assets.lock.json"),
        "upstream_versions": {entry["name"]: entry["version"] for entry in lock["models"]},
        "person_version": 3, "derived_mesh_files": len(lock["derived_files"]),
        "purpose": "Physical simulator inputs; independent of published replay outcomes."})
    verify(assets)
    return {"prepared": True, "scenery_models": len(models), "derived_mesh_files": len(lock["derived_files"])}


def verify(assets):
    from experiments.lake_missions.assets import verify as verify_scenery
    from experiments.v4_gazebo_pilot.assets import load_asset

    lock = read(DATA / "assets.lock.json")
    checked(DATA / "scenario.json", lock["scenario_sha256"])
    if read(assets / "prepared.json")["lock_sha256"] != sha(DATA / "assets.lock.json"):
        raise ValueError("Prepared assets belong to a different simulation lock.")
    manifest = verify_scenery(assets / "scenery")
    if len(manifest["models"]) != 4:
        raise ValueError("All four scenery models are required.")
    for entry in lock["models"]:
        for name, expected in entry["files"].items():
            relative = entry["directory"] + "/" + name
            if relative in lock["overrides"]:
                expected = lock["overrides"][relative]["sha256"]
            checked(assets / "scenery" / relative, expected)
    if read(assets / "standing_person" / "manifest.json") != lock["person"]:
        raise ValueError("Standing-person manifest differs from the locked version.")
    load_asset(assets / "standing_person" / "manifest.json")
    for name, expected in lock["derived_files"].items():
        checked(assets / "derived" / name, expected)
    return {"verified": True, "asset_lock_sha256": sha(DATA / "assets.lock.json")}


def specification(assets):
    from experiments.v4_gazebo_pilot.assets import load_asset
    from experiments.wide_area_missions.runtime import normalize_specification

    verify(assets)
    doc = copy.deepcopy(read(DATA / "scenario.json"))
    doc["asset_directory"] = str(assets / "scenery")
    descriptor = load_asset(assets / "standing_person" / "manifest.json")
    for obj in doc["world"]["objects"]:
        obj["mesh"]["manifest"] = descriptor["manifest_path"]
        obj["mesh"]["asset"] = descriptor
    scenery = doc["scenery"]
    scenery["source_mesh"] = str(local(assets, scenery["source_mesh"]))
    scenery["source_manifest_sha256"] = sha(assets / "scenery" / "manifest.json")
    scenery["implementation_sha256"] = sha(ROOT / "experiments" / "wide_area_missions" / "scenery.py")
    for asset in scenery["derived_assets"]:
        for visual in asset["visuals"]:
            for key in ("mesh", "albedo", "normal"):
                if visual[key]:
                    visual[key] = str(local(assets, visual[key]))
    return normalize_specification(doc)


def doctor(python):
    if shutil.which("gz") is None:
        raise RuntimeError("Gazebo Harmonic is required; see docs/SIMULATION.md.")
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    probe = subprocess.run([python, "-c",
        "import numpy; from PIL import Image; from gz.transport13 import Node; "
        "from gz.msgs10.clock_pb2 import Clock; print('bindings available')"],
        text=True, capture_output=True, env=environment, timeout=30)
    if probe.returncode:
        raise RuntimeError("Worker Python lacks NumPy, Pillow or Gazebo bindings. " + probe.stderr.strip())
    version = subprocess.check_output(["gz", "sim", "--versions"], text=True, timeout=30).strip()
    if not any(value.startswith("8.") for value in version.split()):
        raise RuntimeError("This acquisition workflow requires Gazebo Sim 8 (Harmonic).")
    return {"gazebo_version": version, "worker_bindings": True,
        "scope": "Dependency probe only; local transport and rendering are checked by capture."}


def capture(assets, output, python):
    import numpy as np
    from PIL import Image
    from experiments.wide_area_missions.runtime import capture as acquire

    doctor(python)
    doc = specification(assets)
    os.chdir(ROOT)
    result = acquire(doc, output, python=python)
    if not result["provenance"]["full_declared_mission"] or len(result["frames"]) != 21:
        raise ValueError("Incomplete 42-second recording.")
    shapes, contrasts = set(), []
    for frame in result["frames"]:
        checked(output / frame["file"], frame["sha256"])
        with np.load(output / frame["file"], allow_pickle=False) as arrays:
            shapes.add(tuple(arrays["rgb"].shape))
            contrasts.append(float(arrays["rgb"].std()))
    if shapes != {(720, 960, 3)} or min(contrasts) <= 1:
        raise ValueError("Rendered frame shape or image variation check failed.")
    with np.load(output / result["frames"][9]["file"], allow_pickle=False) as arrays:
        Image.fromarray(arrays["rgb"]).save(output / "preview.png")
    summary = {"passed": True, "duration_s": 42, "captures": 21,
        "scored_motion_captures": 18, "settle_captures": 3,
        "image_shape": [720, 960, 3], "minimum_rgb_std": min(contrasts),
        "physical_route_validation": read(output / "physical_route_validation.json"),
        "asset_lock_sha256": sha(DATA / "assets.lock.json"),
        "portable_scenario_sha256": sha(DATA / "scenario.json"),
        "scope": "Fresh physical capture only; not a rerun of neural models or routing experiments. "
                 "Published replay inputs are preserved. Pixel equality across hosts is not claimed."}
    save(output / "simulation_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "verify", "doctor", "capture"))
    parser.add_argument("--assets", type=Path, default=ROOT / ".cache" / "simulation")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "simulation_demo")
    parser.add_argument("--python", default="/usr/bin/python3", help="Python with Gazebo's system bindings.")
    parser.add_argument("--archive-directory", type=Path, help="Optional existing pinned archives, each at MODEL/source.zip.")
    parser.add_argument("--person-source", type=Path, help="Optional existing verified standing-person bundle.")
    parser.add_argument("--offline", action="store_true", help="Never download missing assets during preparation.")
    args = parser.parse_args()
    assets, output = args.assets.resolve(), args.output.resolve()
    try:
        if args.mode == "prepare":
            result = prepare(assets, args.archive_directory, args.person_source, args.offline)
        elif args.mode == "verify":
            result = verify(assets)
        elif args.mode == "doctor":
            result = doctor(args.python)
        else:
            result = capture(assets, output, args.python)
    except Exception as error:
        print(json.dumps({"passed": False, "error_type": type(error).__name__, "error": str(error)}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
