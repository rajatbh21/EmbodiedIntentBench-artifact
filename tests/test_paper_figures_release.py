"""Verify the portable paper figures against their shipped scientific sources."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "examples/paper_figures"
SPEC = importlib.util.spec_from_file_location("paper_figure_verifier", PACKAGE / "verify_sources.py")
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


@pytest.fixture
def relocated(tmp_path):
    destination = tmp_path / "isolated"
    shutil.copytree(PACKAGE, destination / "paper_figures")
    shutil.copytree(ROOT / "examples/validation", destination / "validation")
    return destination


def test_original_scientific_data_is_preserved():
    assert hashlib.sha256((PACKAGE / "figure_data.json").read_bytes()).hexdigest() == (
        "17f1c44352b382583e195c0467eb340e34f14f951741018a07f67e25060825ec"
    )


def test_release_figure_sources_verify_without_runtime_dependencies():
    result = VERIFY.verify()
    assert result["passed"] and result["checks_passed"] >= 180
    assert result["scope"]["new_policy_execution"] is False
    assert "no raw Depot rerun" in result["scope"]["fig4"]


def test_package_relocates_without_original_checkout(relocated, tmp_path):
    process = subprocess.run(
        [sys.executable, "-S", str(relocated / "paper_figures/verify_sources.py")],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)["passed"]


def test_modified_validation_source_is_rejected(relocated):
    source = relocated / "validation/exp3_metrics.csv"
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="SHA-256"):
        VERIFY.verify(relocated / "paper_figures")


def test_depot_original_rows_are_reduced_not_only_hash_checked(relocated):
    package = relocated / "paper_figures"
    path = package / "sources/depot_rows.json"
    rows = json.loads(path.read_text())
    rows["policy_cases"][0]["full_success"] = not rows["policy_cases"][0]["full_success"]
    path.write_text(json.dumps(rows))
    manifest_path = package / "sources/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["depot_rows_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="original case successes"):
        VERIFY.verify(package)


def test_summary_metadata_change_does_not_hide_scientific_mismatch(relocated):
    path = relocated / "validation/summary.json"
    summary = json.loads(path.read_text())
    for row in summary["exp2"]["class_counts"]:
        if row["contract"] == "confirm":
            row["conditions"] += 1
            break
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="summary partition"):
        VERIFY.verify(relocated / "paper_figures")
