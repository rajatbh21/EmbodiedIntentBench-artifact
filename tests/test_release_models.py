"""Offline integrity checks for optional model acquisition; no weights or ML imports."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "_release_fetch_models", Path(__file__).resolve().parents[1] / "tools/fetch_models.py"
)
fetch = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = fetch
_SPEC.loader.exec_module(fetch)

PAYLOAD = b"trusted checkpoint bytes"


@pytest.fixture
def asset():
    return fetch.Asset(
        "yolo11s",
        "models/example.pt",
        "https://example.invalid/model.pt",
        hashlib.sha256(PAYLOAD).hexdigest(),
        len(PAYLOAD),
    )


def no_network(*args, **kwargs):
    raise AssertionError("A cache-only operation attempted a network request")


def mock_response(monkeypatch, response, *, on_open=None):
    class Opener:
        def open(self, request, timeout):
            assert request.full_url == "https://example.invalid/model.pt"
            assert timeout > 0
            if on_open:
                on_open()
            return response

    monkeypatch.setattr(fetch.urllib.request, "build_opener", lambda *args: Opener())


def test_valid_cache_is_reused_and_verified_without_network(tmp_path, monkeypatch, asset):
    target = tmp_path / asset.path
    target.parent.mkdir(parents=True)
    target.write_bytes(PAYLOAD)
    modified = target.stat().st_mtime_ns
    monkeypatch.setattr(fetch, "assets", lambda: (asset,))
    monkeypatch.setattr(fetch.urllib.request, "build_opener", no_network)
    args = ["--cache-root", str(tmp_path), "--model", "yolo11s"]

    assert fetch.main(args) == 0
    assert fetch.main([*args, "--verify-only"]) == 0
    assert fetch.main(args) == 0
    assert target.read_bytes() == PAYLOAD
    assert target.stat().st_mtime_ns == modified
    assert list(target.parent.iterdir()) == [target]


def test_missing_offline_cache_fails_without_creating_files(tmp_path, monkeypatch, asset):
    cache = tmp_path / "missing"
    monkeypatch.setattr(fetch, "assets", lambda: (asset,))
    monkeypatch.setattr(fetch.urllib.request, "build_opener", no_network)

    assert fetch.main(["--cache-root", str(cache), "--verify-only"]) == 1
    assert not cache.exists()


def test_same_size_corrupt_cache_is_preserved(tmp_path, monkeypatch, asset):
    target = tmp_path / asset.path
    target.parent.mkdir(parents=True)
    corrupt = b"x" * len(PAYLOAD)
    target.write_bytes(corrupt)
    monkeypatch.setattr(fetch, "assets", lambda: (asset,))
    monkeypatch.setattr(fetch.urllib.request, "build_opener", no_network)

    assert fetch.main(["--cache-root", str(tmp_path), "--model", "yolo11s"]) == 1
    assert target.read_bytes() == corrupt


def test_verified_download_becomes_visible_only_when_complete(tmp_path, monkeypatch, asset):
    target = tmp_path / asset.path

    class InspectableStream(io.BytesIO):
        def read(self, size):
            assert not target.exists()
            return super().read(size)

    mock_response(monkeypatch, InspectableStream(PAYLOAD))
    fetch.download(asset, target)

    assert target.read_bytes() == PAYLOAD
    fetch.verify(target, asset.sha256, asset.size_bytes)
    assert list(target.parent.iterdir()) == [target]


@pytest.mark.parametrize("failure", ["truncated", "oversized", "wrong_digest"])
def test_invalid_download_never_installs_checkpoint(tmp_path, monkeypatch, asset, failure):
    payload = {
        "truncated": PAYLOAD[:-1],
        "oversized": PAYLOAD + b"x",
        "wrong_digest": b"x" * len(PAYLOAD),
    }[failure]
    target = tmp_path / asset.path
    mock_response(monkeypatch, io.BytesIO(payload))

    with pytest.raises(ValueError):
        fetch.download(asset, target)
    assert not target.exists()
    assert not list(target.parent.iterdir())


def test_interrupted_download_removes_partial_file(tmp_path, monkeypatch, asset):
    target = tmp_path / asset.path

    class InterruptedStream(io.BytesIO):
        def read(self, size):
            if self.tell():
                raise OSError("simulated connection interruption")
            return super().read(3)

    mock_response(monkeypatch, InterruptedStream(PAYLOAD))
    with pytest.raises(OSError, match="interruption"):
        fetch.download(asset, target)
    assert not target.exists()
    assert not list(target.parent.iterdir())


def test_concurrent_mismatched_destination_is_not_overwritten(tmp_path, monkeypatch, asset):
    target = tmp_path / asset.path
    existing = b"x" * len(PAYLOAD)
    mock_response(monkeypatch, io.BytesIO(PAYLOAD), on_open=lambda: target.write_bytes(existing))

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        fetch.download(asset, target)
    assert target.read_bytes() == existing
    assert list(target.parent.iterdir()) == [target]


def test_symlink_cannot_escape_cache_or_substitute_checkpoint(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (cache / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        fetch.destination(cache, "escape/model.pt")

    (cache / "real.pt").write_bytes(PAYLOAD)
    (cache / "alias.pt").symlink_to(cache / "real.pt")
    with pytest.raises(ValueError):
        fetch.destination(cache, "alias.pt")


def test_plain_http_download_and_redirect_are_rejected(tmp_path, monkeypatch, asset):
    unsafe = fetch.Asset(asset.model, asset.path, "http://example.invalid/a", asset.sha256, 1)
    monkeypatch.setattr(fetch.urllib.request, "build_opener", no_network)
    with pytest.raises(ValueError, match="HTTPS"):
        fetch.download(unsafe, tmp_path / asset.path)
    with pytest.raises(ValueError, match="HTTPS"):
        fetch.HTTPSOnlyRedirect().redirect_request(
            None, None, 302, "", {}, "http://example.invalid/a"
        )
    assert not list(tmp_path.iterdir())


def test_cabinet_manifest_matches_frozen_experiment_and_is_idempotent(tmp_path):
    expected = "defe6058340cf7ae414c08758ab591fe930afc68b1730b6a1d62e8981a978ace"
    fetch.write_manifest(tmp_path, verify_only=False)
    target = tmp_path / fetch.CABINET_DIRECTORY / "manifest.json"
    content = target.read_bytes()
    modified = target.stat().st_mtime_ns
    assert hashlib.sha256(content).hexdigest() == expected
    assert len(json.loads(content)["files"]) == 12

    fetch.write_manifest(tmp_path, verify_only=True)
    fetch.write_manifest(tmp_path, verify_only=False)
    assert target.read_bytes() == content
    assert target.stat().st_mtime_ns == modified

    target.write_bytes(b"x" * len(content))
    with pytest.raises(ValueError):
        fetch.write_manifest(tmp_path, verify_only=False)
    assert target.read_bytes() == b"x" * len(content)


def test_offline_manifest_check_does_not_generate_missing_manifest(tmp_path):
    with pytest.raises(ValueError, match="Missing file"):
        fetch.write_manifest(tmp_path, verify_only=True)
    assert not list(tmp_path.iterdir())
