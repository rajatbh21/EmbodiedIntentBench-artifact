#!/usr/bin/env python3
"""Fetch or verify the exact three perception models used by the paper replay.

Standard library only. Downloads are HTTPS, bounded by pinned file sizes, checked
with SHA-256, and atomically installed. Downloaded Python/checkpoint files are
never imported or executed here. Existing mismatched files are not overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_CACHE = Path(__file__).resolve().parents[1] / ".cache"
CABINET_SOURCE = "f48a36294000e7cc186efea07379b915be05418f"
CABINET_REPOSITORY = "dronefreak/cabinet-mobilenetv3-small-uavid"
CABINET_REVISION = "25e46666435fd59d2033e0c47cc089cad5cf7da5"
CABINET_DIRECTORY = "models/cabinet_uavid_small"
CABINET_MANIFEST_SHA256 = "defe6058340cf7ae414c08758ab591fe930afc68b1730b6a1d62e8981a978ace"


@dataclass(frozen=True)
class Asset:
    model: str
    path: str
    url: str
    sha256: str
    size_bytes: int


# Order matches the original acquisition manifest, including empty package files.
CABINET_FILES = (
    ("src/__init__.py", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", 0),
    (
        "src/models/__init__.py",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        0,
    ),
    ("src/models/cab.py", "b561d2bb74700358c8fde65920e229a646a4e367c7b5579207227b030798d752", 6559),
    (
        "src/models/cabinet.py",
        "4e08ec92c7d42945c8f338ac4f00aa496241358601daf0c9a1cd085eda6cfe0c",
        12088,
    ),
    (
        "src/models/mobilenetv3.py",
        "e4c492cacbb4a3d6068835e6ad4e8906dcb5a648c6376454123cc7af65faa3cf",
        9932,
    ),
    (
        "src/models/constants.py",
        "42d062632dee68577d98342b22fd825ed10ced2d31878ed8ce7f181c72ae3c31",
        993,
    ),
    (
        "configs/UAVid_info.json",
        "f19d60b306ca5ef89ee0e7255775fd66f756adac8c01847807c76100abbc1768",
        1496,
    ),
    (
        "src/datasets/uavid.py",
        "240ea9eb4d731a5637b89f4998184c792d37e47b20f1984b7495d35d52fe665e",
        12282,
    ),
    ("LICENSE", "568c8f818a927c2c062e54fcb2a5844830ed52f479f02b25941b6eb8958ca860", 11343),
    ("README.md", "9e2a10fa40e679bb2097ac89ea688827d86b8fe023ca9ae55853fa5d28b59c6b", 8384),
    ("config.yaml", "ccaa50f8f022e5915ba5576663c8e511d040db76ce7b40b6b1cd926cbfd019ea", 1721),
    (
        "cabinet_best.pth",
        "38aeb238e75cfebcf8d475fe865064b9907c3f6cdcb8f404f10d834c4de36790",
        24025991,
    ),
)


def assets() -> tuple[Asset, ...]:
    result = [
        Asset(
            "maskrcnn",
            "torch/hub/checkpoints/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth",
            "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth",
            "73cbd0190fcbe3ba339921fbce2c3a0b6bb9126c9a133c85e43a2a8e060a109e",
            185828065,
        ),
        Asset(
            "yolo11s",
            "models/yolo11s-seg.pt",
            "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-seg.pt",
            "1caa81c0195412efa411b632bcfb8c184939dddb6ae41f6a80c41b211ff257c3",
            20669228,
        ),
    ]
    for name, digest, size in CABINET_FILES:
        if name in {"README.md", "config.yaml", "cabinet_best.pth"}:
            url = f"https://huggingface.co/{CABINET_REPOSITORY}/resolve/{CABINET_REVISION}/{name}"
        else:
            url = f"https://raw.githubusercontent.com/dronefreak/CABiNet/{CABINET_SOURCE}/{name}"
        result.append(Asset("cabinet", f"{CABINET_DIRECTORY}/{name}", url, digest, size))
    return tuple(result)


def cabinet_manifest_bytes() -> bytes:
    """Reproduce the original, path-independent manifest byte for byte."""
    manifest = dict(
        schema_version="0.1",
        repository=CABINET_REPOSITORY,
        revision=CABINET_REVISION,
        source_commit=CABINET_SOURCE,
        source_url="https://github.com/dronefreak/CABiNet",
        code_license="Apache-2.0",
        weights_license="AGPL-3.0",
        files={name: digest for name, digest, _ in CABINET_FILES},
        weights="cabinet_best.pth",
        class_metadata="configs/UAVid_info.json",
        normalization=dict(mean=[0.480, 0.499, 0.457], std=[0.225, 0.208, 0.228]),
        normalization_source="src/datasets/uavid.py",
    )
    content = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    if hashlib.sha256(content).hexdigest() != CABINET_MANIFEST_SHA256:
        raise ValueError("The built-in CABiNet manifest has changed")
    return content


def destination(cache: Path, relative: str) -> Path:
    path = cache / relative
    if path.is_symlink() or not path.resolve().is_relative_to(cache):
        raise ValueError(f"Refusing a symlink or destination outside cache: {relative}")
    return path


def verify(path: Path, digest: str, size: int) -> None:
    if not path.is_file():
        raise ValueError(f"Missing file: {path}")
    if path.stat().st_size != size:
        raise ValueError(f"File size mismatch: {path}")
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != digest:
        raise ValueError(f"SHA-256 mismatch: {path}")


class HTTPSOnlyRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme != "https":
            raise ValueError("Refusing a non-HTTPS download redirect")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(asset: Asset, path: Path) -> None:
    if urllib.parse.urlsplit(asset.url).scheme != "https":
        raise ValueError("Download URL must use HTTPS")
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(HTTPSOnlyRedirect())
    request = urllib.request.Request(
        asset.url, headers={"User-Agent": "missionbench-model-fetch/1.0"}
    )
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".download-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            with opener.open(request, timeout=60) as response:
                received = 0
                while chunk := response.read(1024 * 1024):
                    received += len(chunk)
                    if received > asset.size_bytes:
                        raise ValueError(f"Download exceeds pinned size: {asset.path}")
                    stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        verify(temporary, asset.sha256, asset.size_bytes)
        # Link from the same filesystem: atomic creation without overwriting a
        # file another fetch may have installed since our earlier existence check.
        try:
            os.link(temporary, path)
        except FileExistsError:
            verify(path, asset.sha256, asset.size_bytes)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_manifest(cache: Path, *, verify_only: bool) -> None:
    content = cabinet_manifest_bytes()
    path = destination(cache, f"{CABINET_DIRECTORY}/manifest.json")
    if path.exists() or verify_only:
        verify(path, CABINET_MANIFEST_SHA256, len(content))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".manifest-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            verify(path, CABINET_MANIFEST_SHA256, len(content))
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--model", action="append", choices=("maskrcnn", "yolo11s", "cabinet"))
    parser.add_argument(
        "--verify-only", action="store_true", help="Verify without network or writes"
    )
    parser.add_argument(
        "--list", action="store_true", help="Print pinned URLs, hashes, sizes; no writes"
    )
    args = parser.parse_args(argv)
    selected = set(args.model or ("maskrcnn", "yolo11s", "cabinet"))
    manifest = tuple(asset for asset in assets() if asset.model in selected)
    if args.list:
        print(json.dumps([asdict(asset) for asset in manifest], indent=2))
        return 0
    cache = args.cache_root.expanduser().resolve()
    errors = []
    for asset in manifest:
        try:
            path = destination(cache, asset.path)
            if path.exists() or args.verify_only:
                verify(path, asset.sha256, asset.size_bytes)
            else:
                print(f"Downloading {asset.path}", flush=True)
                download(asset, path)
            print(f"Verified {asset.path}", flush=True)
        except (OSError, ValueError, urllib.error.URLError) as exc:
            errors.append(f"{asset.path}: {exc}")
    if "cabinet" in selected and not errors:
        try:
            write_manifest(cache, verify_only=args.verify_only)
            print(f"Verified {CABINET_DIRECTORY}/manifest.json")
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(
            "Existing mismatched files are preserved. Inspect/remove them before retrying.",
            file=sys.stderr,
        )
        return 1
    print(f"Ready: {len(manifest)} pinned upstream files in {cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
