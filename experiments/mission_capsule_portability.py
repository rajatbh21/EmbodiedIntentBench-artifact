"""Package and test an endpoint capsule outside the research checkout."""

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(capsule, archive, report):
    capsule, archive, report = (p.resolve() for p in (capsule, archive, report))
    if archive.exists() or report.exists():
        raise FileExistsError("Preserve the prior package and audit; choose new output paths")
    manifest_path = capsule / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    with tarfile.open(archive, "x:gz", compresslevel=6) as bundle:
        bundle.add(capsule, arcname=capsule.name)
    scratch = Path(tempfile.mkdtemp(prefix="mission_endpoint_review_"))
    with tarfile.open(archive, "r:gz") as bundle:
        bundle.extractall(scratch, filter="data")
    extracted = scratch / capsule.name
    command = [sys.executable, "-I", "-S", str(extracted / "review.py"), "check"]

    def check(label):
        proc = subprocess.run(command, cwd=scratch, capture_output=True, text=True, check=False)
        log = report.with_name(report.stem + "_" + label + ".log")
        log.write_text(proc.stdout + proc.stderr)
        return proc, log

    baseline, baseline_log = check("clean")
    if baseline.returncode:
        raise RuntimeError("Relocated check failed: " + str(baseline_log))
    result = json.loads(baseline.stdout)
    if not (
        result["complete"]
        and result["isolated_python"]
        and result["site_packages_disabled"]
        and result["policy_executions"] == 504
        and result["witness_successes"] == 24
        and Path(result["scorer_loaded_from"]).is_relative_to(extracted)
    ):
        raise ValueError("Incomplete or nonisolated endpoint reproduction")

    # Change only a disposable extracted copy; the capsule and experiment stay intact.
    target_name = manifest["rows"][0]["relative"] + "/outcome.json"
    target = extracted / target_name
    original = target.read_bytes()
    try:
        target.write_bytes(original + b"\n")
        tampered, tampered_log = check("tampered")
    finally:
        target.write_bytes(original)
    if tampered.returncode == 0 or "Changed capsule artifact" not in tampered.stderr:
        raise ValueError("Changed-artifact negative control was not rejected")
    if sha(target) != manifest["files_sha256"][target_name]:
        raise ValueError("Temporary copy restoration failed")
    if sha(extracted / "manifest.json") != sha(manifest_path):
        raise ValueError("Archive manifest differs from source")

    audit = {
        "complete": True,
        "schema_version": "mission_endpoint_portability_v1",
        "scope": (
            "Portable canonical scoring from audited logs, not complete experiment reproduction"
        ),
        "archive": str(archive),
        "archive_sha256": sha(archive),
        "archive_bytes": archive.stat().st_size,
        "capsule_manifest_sha256": sha(manifest_path),
        "files_verified": len(manifest["files_sha256"]),
        "relocated_root": str(extracted),
        "command": command,
        "python_version": sys.version,
        "endpoint_result": result,
        "negative_control": {
            "artifact": target_name,
            "modification": "Append a newline to an extracted outcome, retaining valid JSON",
            "rejected": True,
            "exit_code": tampered.returncode,
            "temporary_copy_restored": True,
        },
        "logs_sha256": {str(p): sha(p) for p in (baseline_log, tampered_log)},
        "source_sha256": sha(Path(__file__)),
    }
    report.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        k: audit[k] for k in ("complete", "archive", "archive_bytes", "files_verified")
    }))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    run(args.capsule, args.archive, args.report)
