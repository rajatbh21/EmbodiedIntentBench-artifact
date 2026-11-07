"""Portable checks for the release audit, independent of research data."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "check_release.py"
SPEC = importlib.util.spec_from_file_location("release_audit", SCRIPT)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


@pytest.fixture
def release(tmp_path):
    root = tmp_path / "release"
    root.mkdir()
    (root / "missionbench").mkdir()
    (root / "missionbench" / "__init__.py").write_text("")
    (root / "pyproject.toml").write_text(
        '[project]\nname = "missionbench"\n'
        'authors = [{ name = "Example researcher" }]\n'
        '[project.scripts]\nmissionbench = "missionbench.run_benchmark:main"\n'
    )
    (root / "README.md").write_text("# Mission benchmark\n")
    return root


def codes(result):
    return {item["code"] for item in result["issues"]}


def test_portable_clean_tree_and_ignored_outputs(release):
    for folder in ("results", "output", ".cache", ".venv", ".git"):
        (release / folder).mkdir()
        (release / folder / "private.txt").write_text("private-example")
    result = AUDIT.audit_release(release, forbidden=("private-example",))
    assert result["passed"]
    assert result["scanned_files"] == 3


def test_identifiers_case_insensitive_in_contents_and_filenames(release):
    (release / "PRIVATE-EXAMPLE.txt").write_text("public description")
    (release / "note.txt").write_text("A Private-Example author")
    result = AUDIT.audit_release(release, forbidden=("private-example",))
    assert len(result["issues"]) == 2
    assert codes(result) == {"forbidden_identifier"}
    assert "Private-Example" not in json.dumps(result)


def test_machine_paths_detected_without_private_token_list(release):
    path = "/" + "home" + "/" + "researcher" + "/workspace"
    (release / "note.txt").write_text(path)
    assert codes(AUDIT.audit_release(release)) == {"machine_path"}


def test_symlink_escape_and_broken_link(release, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    (release / "escape.txt").symlink_to(outside)
    (release / "broken.txt").symlink_to(release / "absent.txt")
    assert codes(AUDIT.audit_release(release)) == {"escaping_symlink", "broken_symlink"}


def test_directory_symlink_escape(release, tmp_path):
    (release / "external").symlink_to(tmp_path, target_is_directory=True)
    assert "escaping_symlink" in codes(AUDIT.audit_release(release))


def test_oversized_file(release):
    (release / "large.bin").write_bytes(b"0" * 1025)
    assert "oversized_file" in codes(AUDIT.audit_release(release, max_bytes=1024))


def test_local_markdown_links_and_reference_images(release):
    (release / "with space.txt").write_text("ok")
    (release / "README.md").write_text(
        '[file](<with space.txt>)\n[file](with%20space.txt)\n'
        '[anchor](#usage)\n[web](https://example.org/missing)\n'
        '[email](mailto:team@example.org)\n'
        '```md\n[example](not-a-real-file.txt)\n```\n'
        '![missing](missing.png)\n[reference]: another-missing.png\n'
    )
    result = AUDIT.audit_release(release)
    assert len(result["issues"]) == 2
    assert codes(result) == {"broken_link"}


def test_local_links_cannot_depend_on_outside_or_excluded_files(release, tmp_path):
    (tmp_path / "outside.txt").write_text("Present only on the build machine")
    (release / "results").mkdir()
    (release / "results" / "local.txt").write_text("Generated output")
    (release / "README.md").write_text(
        "[outside](../outside.txt)\n[output](results/local.txt)\n"
        "[package](missionbench/)\n[root](./)\n"
    )
    assert codes(AUDIT.audit_release(release)) == {"escaping_link", "unpackaged_link"}


def test_tracked_inventory_rejects_untracked_link_and_scans_tracked_outputs(release):
    subprocess.run(["git", "init", "-q", str(release)], check=True)
    (release / "README.md").write_text("[instructions](untracked.md)\n")
    (release / "results").mkdir()
    (release / "results" / "tracked.txt").write_text("private-example")
    subprocess.run(["git", "-C", str(release), "add", "."], check=True)
    (release / "untracked.md").write_text("Local file absent from the archive")
    result = AUDIT.audit_release(release, tracked_only=True, forbidden=("private-example",))
    assert codes(result) == {"unpackaged_link", "forbidden_identifier"}
    assert result["inventory"] == "git_index"
    assert result["scanned_files"] == 4


def test_tracked_inventory_requires_standalone_checkout(release):
    assert codes(AUDIT.audit_release(release, tracked_only=True)) == {"missing_tracked_checkout"}


def test_tracked_inventory_cannot_audit_cleaned_worktree_over_stale_index(release):
    subprocess.run(["git", "init", "-q", str(release)], check=True)
    (release / "note.txt").write_text("private-example")
    subprocess.run(["git", "-C", str(release), "add", "."], check=True)
    (release / "note.txt").write_text("Sanitized but not staged")
    assert "tracked_worktree_mismatch" in codes(AUDIT.audit_release(release, tracked_only=True))


def test_metadata_requires_author_and_correct_package_name(release):
    path = release / "pyproject.toml"
    path.write_text(path.read_text().replace('name = "missionbench"', 'name = "other"')
                    .replace('authors = [{ name = "Example researcher" }]', 'authors = []'))
    assert codes(AUDIT.audit_release(release)) == {"package_name", "author_metadata"}


def test_attribution_exemption_preserves_credit_but_not_machine_paths(release):
    (release / "UPSTREAM_LICENSE.txt").write_text("Copyright upstream-example")
    result = AUDIT.audit_release(release, forbidden=("upstream-example",),
        attribution_allowlist={"UPSTREAM_LICENSE.txt": "Required upstream copyright attribution."})
    assert result["passed"]
    with (release / "UPSTREAM_LICENSE.txt").open("a") as stream:
        stream.write("\n/" + "Users" + "/" + "researcher" + "/workspace")
    result = AUDIT.audit_release(release, forbidden=("upstream-example",),
        attribution_allowlist={"UPSTREAM_LICENSE.txt": "Required upstream copyright attribution."})
    assert codes(result) == {"machine_path"}


def test_unused_or_empty_exemptions_fail(release):
    result = AUDIT.audit_release(release, attribution_allowlist={"missing.txt": ""})
    assert codes(result) == {"invalid_exemption", "unused_exemption"}


def test_orphan_requires_standalone_committed_checkout(release):
    assert codes(AUDIT.audit_release(release, require_orphan=True)) == {"missing_release_commit"}
    subprocess.run(["git", "init", "-q", str(release)], check=True)
    subprocess.run(["git", "-C", str(release), "add", "."], check=True)
    command = ["git", "-C", str(release), "-c", "user.name=Example researcher", "-c",
               "user.email=person@example.org", "commit", "-q"]
    subprocess.run([*command, "-m", "Release"], check=True)
    assert AUDIT.audit_release(release, require_orphan=True)["passed"]
    subprocess.run([*command, "--allow-empty", "-m", "Second commit"], check=True)
    assert codes(AUDIT.audit_release(release, require_orphan=True)) == {"history_parent"}


def test_orphan_rejects_forbidden_commit_metadata(release):
    subprocess.run(["git", "init", "-q", str(release)], check=True)
    subprocess.run(["git", "-C", str(release), "add", "."], check=True)
    subprocess.run(["git", "-C", str(release), "-c", "user.name=Example researcher",
                    "-c", "user.email=person@example.org", "commit", "-q", "-m",
                    "private-example release"], check=True)
    result = AUDIT.audit_release(release, require_orphan=True, forbidden=("private-example",))
    assert codes(result) == {"forbidden_commit_identifier"}
    assert "Example researcher" not in json.dumps(result)
    assert "private-example" not in json.dumps(result)


def test_orphan_tracked_audit_requires_committed_index(release):
    subprocess.run(["git", "init", "-q", str(release)], check=True)
    subprocess.run(["git", "-C", str(release), "add", "."], check=True)
    subprocess.run(["git", "-C", str(release), "-c", "user.name=Example researcher",
                    "-c", "user.email=person@example.org", "commit", "-q", "-m", "Release"], check=True)
    (release / "note.txt").write_text("Additional staged file")
    subprocess.run(["git", "-C", str(release), "add", "."], check=True)
    assert codes(AUDIT.audit_release(release, require_orphan=True, tracked_only=True)) == {"uncommitted_index"}


def test_cli_failure_and_private_tokens(release, tmp_path):
    (release / "note.txt").write_text("private-example")
    tokens = tmp_path / "private-tokens.json"
    tokens.write_text(json.dumps(["private-example"]))
    process = subprocess.run([sys.executable, str(SCRIPT), "--root", str(release),
        "--forbidden-file", str(tokens)], capture_output=True, text=True)
    assert process.returncode == 1
    result = json.loads(process.stdout)
    assert not result["passed"] and result["issue_count"] == 1
    assert "private-example" not in process.stdout


def test_invalid_private_configuration_fails(release, tmp_path):
    tokens = tmp_path / "private-tokens.json"
    tokens.write_text('{}')
    process = subprocess.run([sys.executable, str(SCRIPT), "--root", str(release),
        "--forbidden-file", str(tokens)], capture_output=True, text=True)
    assert process.returncode == 2
    assert "configuration_error" in json.loads(process.stdout)


def test_private_token_configuration_must_stay_outside_release(release):
    tokens = release / "private-tokens.json"
    tokens.write_text('["private-example"]')
    process = subprocess.run([sys.executable, str(SCRIPT), "--root", str(release),
        "--forbidden-file", str(tokens)], capture_output=True, text=True)
    assert process.returncode == 2
    assert "outside the release root" in json.loads(process.stdout)["configuration_error"]
