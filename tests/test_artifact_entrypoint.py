"""Artifact entrypoint protects evidence and detects incomplete reproductions."""
import json
from pathlib import Path

import pytest

from tools import artifact


def test_existing_output_is_preserved(tmp_path):
    evidence = tmp_path / "receipt.json"
    evidence.write_text("existing evidence")
    with pytest.raises(ValueError, match="prior results are preserved"):
        artifact.run("full", tmp_path)
    assert evidence.read_text() == "existing evidence"


def test_comparison_detects_changed_and_missing_tables(tmp_path):
    reference, generated = tmp_path / "reference", tmp_path / "generated"
    reference.mkdir()
    generated.mkdir()
    for folder in (reference, generated):
        (folder / "runs.csv").write_text("success\n1\n")
        (folder / "exp3_timelines.json").write_text("[]")
        (folder / "summary.json").write_text('{"runs": 1}')
    assert artifact.compare_reports(reference, generated) == []
    (generated / "runs.csv").write_text("success\n0\n")
    (generated / "exp3_timelines.json").unlink()
    assert {row["file"] for row in artifact.compare_reports(reference, generated)} == {
        "runs.csv", "exp3_timelines.json",
    }


def test_reference_source_tampering_is_rejected(tmp_path):
    source = tmp_path / "experiments/validation"
    reference = tmp_path / "examples/validation"
    source.mkdir(parents=True)
    reference.mkdir(parents=True)
    (source / "config.json").write_text("{}")
    (source / "runtime.py").write_text("original = True\n")
    (reference / "summary.json").write_text(json.dumps({
        "config_sha256": artifact.digest(source / "config.json"),
        "source_sha256": {"runtime.py": artifact.digest(source / "runtime.py")},
    }))
    (reference / "runs.csv").write_text("success\n1\n")
    (reference / "exp3_timelines.json").write_text("[]")
    assert "runs.csv" in artifact.verify_reference_sources(tmp_path)
    (source / "runtime.py").write_text("original = False\n")
    with pytest.raises(ValueError, match="Validation source differs"):
        artifact.verify_reference_sources(tmp_path)


def test_published_validation_sources_match_receipt():
    repo = Path(__file__).resolve().parents[1]
    assert "runs.csv" in artifact.verify_reference_sources(repo)


def test_only_declared_summary_source_digest_may_change(tmp_path):
    reference, generated = tmp_path / "reference", tmp_path / "generated"
    reference.mkdir()
    generated.mkdir()
    original = {"source_sha256": {"figures.py": "a" * 64, "runtime.py": "c" * 64},
                "counts": {"successes": 218}}
    released = json.loads(json.dumps(original))
    released["source_sha256"]["figures.py"] = "b" * 64
    change = {"figures.py": {"original_sha256": "a" * 64, "released_sha256": "b" * 64,
                              "reason": "Presentation identifier salt"}}
    for folder in (reference, generated):
        (folder / "runs.csv").write_text("success\n1\n")
        (folder / "exp3_timelines.json").write_text("[]")
    (reference / "summary.json").write_text(json.dumps(original))
    (generated / "summary.json").write_text(json.dumps(released))
    assert artifact.compare_reports(reference, generated, change) == []
    assert artifact.compare_reports(reference, generated)
    released["counts"]["successes"] = 219
    (generated / "summary.json").write_text(json.dumps(released))
    assert artifact.compare_reports(reference, generated, change)
    released["counts"]["successes"] = 218
    released["source_sha256"]["runtime.py"] = "d" * 64
    (generated / "summary.json").write_text(json.dumps(released))
    assert artifact.compare_reports(reference, generated, change)
    released["source_sha256"]["runtime.py"] = "c" * 64
    released["source_sha256"]["figures.py"] = "e" * 64
    (generated / "summary.json").write_text(json.dumps(released))
    assert artifact.compare_reports(reference, generated, change)


def test_algorithm_file_cannot_be_exempted_by_release_mapping(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/release_source_changes.json").write_text(json.dumps({
        "schema_version": "release_source_changes_v1",
        "files": {"runtime.py": {"original_sha256": "a" * 64,
                                  "released_sha256": "b" * 64, "reason": "Unexpected change"}},
    }))
    with pytest.raises(ValueError, match="Only the declared figure-presentation source"):
        artifact.declared_source_changes(tmp_path)


def test_report_aggregation_matches_original_order_and_restores_module(monkeypatch, tmp_path):
    import builtins
    from experiments.validation import analysis, execution

    builtin_sum = builtins.sum
    execution_had_sum = "sum" in vars(execution)
    monkeypatch.delattr(analysis, "sum", raising=False)

    def build(raw, report):
        assert analysis.sum([0.1] * 10) == 0.9999999999999999
        assert analysis.sum([1e16, 1.0, -1e16]) == 0.0
        assert analysis.sum([1, 2, 3], 4) == 10
        assert builtins.sum is builtin_sum
        assert ("sum" in vars(execution)) == execution_had_sum

    monkeypatch.setattr(analysis, "build", build)
    artifact.build_reports(tmp_path, tmp_path / "tables")
    assert "sum" not in vars(analysis)
    assert builtins.sum is builtin_sum


def test_report_aggregation_restores_existing_binding_on_failure(monkeypatch, tmp_path):
    from experiments.validation import analysis

    sentinel = object()
    monkeypatch.setattr(analysis, "sum", sentinel, raising=False)

    def fail(raw, report):
        assert analysis.sum is artifact.sequential_sum
        raise RuntimeError("report failure")

    monkeypatch.setattr(analysis, "build", fail)
    with pytest.raises(RuntimeError, match="report failure"):
        artifact.build_reports(tmp_path, tmp_path / "tables")
    assert analysis.sum is sentinel


def test_summary_floating_values_still_require_exact_equality(tmp_path):
    reference, generated = tmp_path / "reference", tmp_path / "generated"
    reference.mkdir()
    generated.mkdir()
    for folder in (reference, generated):
        (folder / "runs.csv").write_text("success\n1\n")
        (folder / "exp3_timelines.json").write_text("[]")
    expected = {"exp4": {"urgency": [{"aware_mean_nonflight_energy_j": 170.98133590143223}]}}
    actual = {"exp4": {"urgency": [{"aware_mean_nonflight_energy_j": 170.9813359014323}]}}
    (reference / "summary.json").write_text(json.dumps(expected))
    (generated / "summary.json").write_text(json.dumps(actual))
    assert artifact.compare_reports(reference, generated) == [
        {"file": "summary.json", "reason": "Generated values differ"},
    ]


def test_recheck_preserves_existing_output(tmp_path):
    with pytest.raises(ValueError, match="prior results are preserved"):
        artifact.recheck(tmp_path / "prior", tmp_path, "a" * 64)
