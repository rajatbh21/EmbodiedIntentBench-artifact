"""Verify local Collada asset bundles before rendering; never fetch at runtime.

Provenance is an author declaration backed by locally hashed source metadata,
not an automated determination of legal rights. Assets remain local-only.
"""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_file(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).is_absolute() or ":" in name:
        raise ValueError("asset paths must be relative local files")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"asset file missing or outside bundle: {name}")
    return path


def load_asset(path: Path) -> dict:
    """Return a checked render descriptor including every declared file hash."""
    path = path.resolve()
    data = json.loads(path.read_text())
    fields = {
        "asset_schema_version",
        "asset_id",
        "source_url",
        "license",
        "license_url",
        "creator",
        "source_metadata",
        "mesh",
        "files",
    }
    if not isinstance(data, dict) or set(data) != fields:
        raise ValueError("invalid asset manifest fields")
    if data["asset_schema_version"] != "0.1":
        raise ValueError("unsupported asset schema")
    for key in fields - {"files"}:
        if not isinstance(data[key], str) or not data[key].strip():
            raise ValueError(f"asset {key} must be a nonempty string")
    if data["license"].lower() in {"unknown", "pending", "unspecified"}:
        raise ValueError("asset license must be known")
    if not isinstance(data["files"], dict) or not data["files"]:
        raise ValueError("asset files must map relative paths to SHA256")
    checked = {}
    for name, expected in data["files"].items():
        file = local_file(path.parent, name)
        if not isinstance(expected, str) or sha256(file) != expected:
            raise ValueError(f"asset checksum mismatch: {name}")
        checked[file] = expected
    mesh = local_file(path.parent, data["mesh"])
    metadata = local_file(path.parent, data["source_metadata"])
    if mesh not in checked or metadata not in checked or mesh.suffix.lower() != ".dae":
        raise ValueError("hashed Collada mesh and source metadata required")
    root = ET.parse(mesh).getroot()
    # Only local image references are supported. Fail closed on external documents.
    for node in root.iter():
        for attr in ("url", "source"):
            ref = node.get(attr)
            if ref and not ref.startswith("#"):
                raise ValueError("external Collada document references are unsupported")
        if node.tag.rsplit("}", 1)[-1] == "image":
            refs = [n.text for n in node.iter() if n.tag.rsplit("}", 1)[-1] == "init_from"]
            if len(refs) != 1 or not refs[0]:
                raise ValueError("Collada images need one local init_from")
            texture = local_file(path.parent, str((mesh.parent / refs[0]).relative_to(path.parent)))
            if texture not in checked:
                raise ValueError("unhashed Collada texture")
    return {
        "manifest_path": str(path.resolve()),
        "manifest_sha256": sha256(path),
        "mesh_path": str(mesh),
        "provenance": data,
    }


def verify_asset(descriptor: dict) -> None:
    if load_asset(Path(descriptor["manifest_path"])) != descriptor:
        raise ValueError("asset changed after scenario validation")
