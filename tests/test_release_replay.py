"""Meaningful release checks: reject tampering and compare every result metric."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("release_replay", ROOT / "tools/replay.py")
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


def test_full_factorial_and_all_input_hashes():
    plan = replay.verify_data(ROOT / "data/replay")
    assert len(plan["grid"]) == 648
    assert len(plan["cases"]) == 6
    assert len({row["policy"] for row in plan["grid"]}) == 9
    assert len({row["intent_schedule"] for row in plan["grid"]}) == 3


def test_comparison_does_not_ignore_nested_or_new_scientific_fields():
    reference = {"run_id": "example", "task_metrics": {"late": 0}, "success": True,
                 "path": "old", "artifacts_sha256": {"protocol": "old"}}
    changed_provenance = dict(reference, path="new", artifacts_sha256={"protocol": "new"})
    assert replay.compare_rows([changed_provenance], [reference]) == []
    changed_result = dict(changed_provenance, task_metrics={"late": 1}, new_metric=3)
    assert replay.compare_rows([changed_result], [reference]) == [
        {"run_id": "example", "fields": ["new_metric", "task_metrics"]}]


def test_paths_cannot_escape_bundle(tmp_path):
    for path in ("../outside", str(tmp_path / "absolute")):
        with pytest.raises(ValueError, match="inside its root"):
            replay.checked_path(tmp_path, path)


def test_tampered_bundle_is_rejected(tmp_path):
    replay.save(tmp_path / "study.json", {"changed": True})
    replay.save(tmp_path / "manifest.json", dict(schema_version="portable_perception_replay_v1", files={
        "study.json": {"bytes": (tmp_path / "study.json").stat().st_size, "sha256": "0" * 64}}))
    with pytest.raises(ValueError, match="Replay input changed"):
        replay.verify_data(tmp_path)


def test_bundle_has_no_absolute_machine_paths():
    import json
    def check(value):
        if isinstance(value, dict):
            for key, val in value.items():
                check(key)
                check(val)
        elif isinstance(value, list):
            for val in value:
                check(val)
        elif isinstance(value, str):
            assert not Path(value).is_absolute(), value
    for path in (ROOT / "data/replay").rglob("*.json"):
        check(json.loads(path.read_text()))
