"""Download licensed, versioned Gazebo Fuel scenery; retain immutable originals."""

import argparse
import hashlib
import json
import stat
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

NAMES = ("Harmonic Terrain", "Harmonic Terrain Objects", "Lake House", "Coast Waves 2")
SERVER = "https://fuel.gazebosim.org/1.0/OpenRobotics/models/"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def verify(folder):
    folder = Path(folder).resolve()
    manifest = json.loads((folder / "manifest.json").read_text())
    for item in manifest["models"]:
        for relative, expected in item["files"].items():
            path = (folder / relative).resolve()
            if not path.is_relative_to(folder) or sha(path) != expected:
                raise ValueError(f"Scenery checksum mismatch: {relative}")
    return manifest


def fetch(folder):
    folder = Path(folder).resolve()
    if (folder / "manifest.json").exists():
        return verify(folder)
    folder.mkdir(parents=True, exist_ok=True)
    entries = []
    for name in NAMES:
        base = SERVER + urllib.parse.quote(name)
        with urllib.request.urlopen(base, timeout=60) as response:
            metadata = json.load(response)
        if metadata["license_name"] != "Creative Commons Attribution 4.0 International":
            raise ValueError("Review unexpected asset license")
        version = metadata["version"]
        key = name.lower().replace(" ", "_")
        dest = folder / key
        dest.mkdir(exist_ok=True)
        save(dest / "fuel_metadata.json", metadata)
        url = f"{base}/{version}/{urllib.parse.quote(name)}.zip"
        archive = dest / "source.zip"
        if not archive.exists():
            with urllib.request.urlopen(url, timeout=120) as response, archive.open("wb") as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
        with zipfile.ZipFile(archive) as z:
            if sum(i.file_size for i in z.infolist()) > 1024**3:
                raise ValueError("Unexpected archive size")
            for info in z.infolist():
                path = (dest / info.filename).resolve()
                if not path.is_relative_to(dest) or stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError("Unsafe archive entry")
            z.extractall(dest)
        files = {str(p.relative_to(folder)): sha(p) for p in sorted(dest.rglob("*")) if p.is_file()}
        entries.append(
            dict(
                name=name,
                owner="OpenRobotics",
                version=version,
                source=base,
                download=url,
                license=metadata["license_name"],
                license_url=metadata["license_url"],
                directory=key,
                files=files,
            )
        )
        print(f"Downloaded {name} v{version}: {archive.stat().st_size / 1e6:.1f} MB", flush=True)
    manifest = dict(schema_version="1.0", models=entries)
    save(folder / "manifest.json", manifest)
    (folder / "ATTRIBUTION.txt").write_text(
        "Scenery by OpenRobotics, Gazebo Fuel. Creative Commons Attribution 4.0.\n"
        "https://creativecommons.org/licenses/by/4.0/\n"
        + "\n".join(f"{e['name']} v{e['version']}: {e['source']}" for e in entries)
        + "\nOriginal files preserved. Benchmark adaptations documented in world.py.\n"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    fetch(parser.parse_args().output)
