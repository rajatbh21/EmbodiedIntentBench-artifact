"""Fetch a pinned public UAVid CABiNet bundle into a fresh local-only directory.

No training data or optional packages are installed. Source is trusted executable
code; review it before use. Keep upstream code and checkpoint licenses distinct.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

SOURCE_COMMIT = "f48a36294000e7cc186efea07379b915be05418f"
REPOSITORY = "dronefreak/cabinet-mobilenetv3-small-uavid"
REVISION = "25e46666435fd59d2033e0c47cc089cad5cf7da5"
WEIGHTS_SHA256 = "38aeb238e75cfebcf8d475fe865064b9907c3f6cdcb8f404f10d834c4de36790"


def prepare(output: Path):
    output.mkdir(parents=True, exist_ok=False)
    files = {}

    def fetch(url, name):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as response:
            path.write_bytes(response.read())
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()

    for name in (
        "src/__init__.py",
        "src/models/__init__.py",
        "src/models/cab.py",
        "src/models/cabinet.py",
        "src/models/mobilenetv3.py",
        "src/models/constants.py",
        "configs/UAVid_info.json",
        "src/datasets/uavid.py",
        "LICENSE",
    ):
        fetch(f"https://raw.githubusercontent.com/dronefreak/CABiNet/{SOURCE_COMMIT}/{name}", name)
    for name in ("README.md", "config.yaml", "cabinet_best.pth"):
        fetch(f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}", name)
    if files["cabinet_best.pth"] != WEIGHTS_SHA256:
        raise ValueError("published checkpoint checksum mismatch")
    manifest = dict(
        schema_version="0.1",
        repository=REPOSITORY,
        revision=REVISION,
        source_commit=SOURCE_COMMIT,
        source_url="https://github.com/dronefreak/CABiNet",
        code_license="Apache-2.0",
        weights_license="AGPL-3.0",
        files=files,
        weights="cabinet_best.pth",
        class_metadata="configs/UAVid_info.json",
        normalization=dict(mean=[0.480, 0.499, 0.457], std=[0.225, 0.208, 0.228]),
        normalization_source="src/datasets/uavid.py",
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output / "manifest.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(prepare(parser.parse_args().output))


if __name__ == "__main__":
    main()
