"""Fail-closed asset preparation checks; no simulator or downloads required."""
import importlib.util
import json
from pathlib import Path
import stat
import zipfile

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "simulate.py"
SPEC = importlib.util.spec_from_file_location("portable_simulation", SCRIPT)
SIM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SIM)


def test_local_asset_path_rejects_escape_and_external_symlink(tmp_path):
    root = tmp_path / "assets"
    root.mkdir()
    assert SIM.local(root, "mesh/model.dae") == root / "mesh/model.dae"
    for invalid in ("../outside", str(tmp_path / "outside")):
        with pytest.raises(ValueError, match="escapes"):
            SIM.local(root, invalid)
    (root / "external").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        SIM.local(root, "external/texture.png")


def test_checksum_rejects_modified_and_missing_assets(tmp_path):
    path = tmp_path / "mesh.dae"
    path.write_bytes(b"locked mesh")
    expected = SIM.sha(path)
    SIM.checked(path, expected)
    path.write_bytes(b"modified mesh")
    with pytest.raises(ValueError, match="Missing or changed"):
        SIM.checked(path, expected)
    path.unlink()
    with pytest.raises(ValueError, match="Missing or changed"):
        SIM.checked(path, expected)


def fixture_lock(tmp_path, monkeypatch, archive):
    data = tmp_path / "data"
    data.mkdir()
    (data / "scenario.json").write_text('{}')
    lock = {"scenario_sha256": SIM.sha(data / "scenario.json"), "overrides": {},
        "models": [{"name": "Fixture", "directory": "fixture", "files": {"model.sdf": "unimportant"},
                    "archive_sha256": SIM.sha(archive), "archive_bytes": archive.stat().st_size}]}
    SIM.save(data / "assets.lock.json", lock)
    monkeypatch.setattr(SIM, "DATA", data)
    return data, lock


@pytest.mark.parametrize("defect", ["traversal", "symlink"])
def test_prepare_rejects_unsafe_archive_entries(tmp_path, monkeypatch, defect):
    cache = tmp_path / "archives"
    (cache / "fixture").mkdir(parents=True)
    archive = cache / "fixture" / "source.zip"
    with zipfile.ZipFile(archive, "w") as output:
        if defect == "traversal":
            output.writestr("../escape.txt", "must never be extracted")
        else:
            entry = zipfile.ZipInfo("escape.txt")
            entry.create_system = 3
            entry.external_attr = (stat.S_IFLNK | 0o777) << 16
            output.writestr(entry, "../outside")
    fixture_lock(tmp_path, monkeypatch, archive)
    with pytest.raises(ValueError, match="escapes|Symlink"):
        SIM.prepare(tmp_path / "assets", archive_directory=cache, offline=True)
    assert not (tmp_path / "assets" / "scenery" / "escape.txt").exists()


def test_prepare_rejects_archive_tampering_before_extraction(tmp_path, monkeypatch):
    cache = tmp_path / "archives"
    (cache / "fixture").mkdir(parents=True)
    archive = cache / "fixture" / "source.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("model.sdf", "locked")
    fixture_lock(tmp_path, monkeypatch, archive)
    with archive.open("ab") as output:
        output.write(b"unapproved change")
    with pytest.raises(ValueError, match="Missing or changed"):
        SIM.prepare(tmp_path / "assets", archive_directory=cache, offline=True)
    assert not (tmp_path / "assets" / "scenery" / "fixture" / "model.sdf").exists()


def test_download_size_bound_and_atomic_destination(tmp_path):
    path = tmp_path / "small.bin"
    SIM.download("data:application/octet-stream;base64,YWJj", path, 3)
    assert path.read_bytes() == b"abc"
    rejected = tmp_path / "too_large.bin"
    with pytest.raises(ValueError, match="size limit"):
        SIM.download("data:application/octet-stream;base64,YWJj", rejected, 2)
    assert not rejected.exists()


def test_verify_rejects_changed_preparation_lock(tmp_path, monkeypatch):
    data = tmp_path / "data"
    assets = tmp_path / "assets"
    data.mkdir()
    assets.mkdir()
    (data / "scenario.json").write_text('{}')
    SIM.save(data / "assets.lock.json", {"scenario_sha256": SIM.sha(data / "scenario.json")})
    SIM.save(assets / "prepared.json", {"lock_sha256": "0" * 64})
    monkeypatch.setattr(SIM, "DATA", data)
    with pytest.raises(ValueError, match="different simulation lock"):
        SIM.verify(assets)
