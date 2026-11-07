"""Download one version-pinned CC0 Fuel asset locally, then verify its bundle.

Explicit preparation only; benchmark runs never download assets. Stores original
metadata for attribution and licensing review. No asset bytes enter source control.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

from experiments.v4_gazebo_pilot.assets import load_asset, sha256

SOURCE = "https://fuel.gazebosim.org/1.0/OpenRobotics/models/Standing%20person"
VERSION = 3
LOGGER = logging.getLogger(__name__)


def download(url: str) -> bytes:
    if f"/{VERSION}/files/" in url:
        return download_file(url)
    request = Request(url, headers={"Accept-Encoding": "gzip"})
    with urlopen(request, timeout=120) as response:
        payload = response.read()
        return (
            gzip.decompress(payload)
            if response.headers.get("Content-Encoding") == "gzip"
            else payload
        )


def download_file(url: str) -> bytes:
    """Bounded range requests avoid single slow asset transfers; verify every range."""
    request = Request(url, headers={"Range": "bytes=0-0", "Accept-Encoding": "identity"})
    with urlopen(request, timeout=120) as response:
        if response.status != 206:
            if response.status != 200:
                raise ValueError("unexpected asset response status")
            return response.read()
        match = re.fullmatch(r"bytes 0-0/(\d+)", response.headers.get("Content-Range", ""))
        if not match or len(response.read()) != 1:
            raise ValueError("invalid asset size response")
        size = int(match[1])
    if not 0 < size <= 64 * 1024 * 1024:
        raise ValueError("asset file size outside supported range")
    chunk_size = 256 * 1024

    def chunk(start: int) -> bytes:
        end = min(start + chunk_size, size) - 1
        req = Request(
            url,
            headers={
                "Range": f"bytes={start}-{end}",
                "Accept-Encoding": "identity",
            },
        )
        with urlopen(req, timeout=120) as response:
            expected = f"bytes {start}-{end}/{size}"
            if response.status != 206 or response.headers.get("Content-Range") != expected:
                raise ValueError("asset server returned inconsistent range")
            payload = response.read()
            if len(payload) != end - start + 1:
                raise ValueError("truncated asset range")
            return payload

    with ThreadPoolExecutor(max_workers=8) as pool:
        return b"".join(pool.map(chunk, range(0, size, chunk_size)))


def prepare(output: Path) -> Path:
    metadata_bytes = download(SOURCE)
    metadata = json.loads(metadata_bytes)
    if metadata.get("license_name") != "Creative Commons Zero v1.0 Universal":
        raise ValueError("source license changed; review before use")
    if metadata.get("version") != VERSION:
        raise ValueError("source version changed; review version-specific licensing before use")
    output.mkdir(parents=True, exist_ok=False)
    (output / "source_metadata.json").write_bytes(metadata_bytes)
    files = ["source_metadata.json", "model.config", "model.sdf", "meshes/standing.dae"]
    for name in files[1:]:
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(download(f"{SOURCE}/{VERSION}/files/{quote(name)}"))
        LOGGER.info("Downloaded %s (%d bytes)", name, destination.stat().st_size)
    mesh = output / "meshes/standing.dae"
    for image in ET.parse(mesh).getroot().iter():
        if image.tag.rsplit("}", 1)[-1] != "image":
            continue
        ref = next(n.text for n in image.iter() if n.tag.rsplit("}", 1)[-1] == "init_from")
        if not ref or ":" in ref or Path(ref).is_absolute():
            raise ValueError("unsupported texture reference")
        destination = (mesh.parent / ref).resolve()
        name = str(destination.relative_to(output.resolve()))
        if name not in files:
            files.append(name)

    def fetch_texture(name: str) -> None:
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(download(f"{SOURCE}/{VERSION}/files/{quote(name)}"))
        LOGGER.info("Downloaded %s (%d bytes)", name, destination.stat().st_size)

    # Independent textures; each file still validates every byte range.
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(fetch_texture, files[4:]))
    creator = ET.parse(output / "model.config").findtext("author/name")
    manifest = {
        "asset_schema_version": "0.1",
        "asset_id": "fuel_standing_person_v3",
        "source_url": f"{SOURCE}/{VERSION}",
        "license": "CC0-1.0",
        "license_url": metadata["license_url"],
        "creator": creator,
        "source_metadata": "source_metadata.json",
        "mesh": "meshes/standing.dae",
        "files": {name: sha256(output / name) for name in files},
    }
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    load_asset(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(prepare(args.output))


if __name__ == "__main__":
    main()
