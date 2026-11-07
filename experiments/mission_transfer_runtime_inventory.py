"""Record existing runtime and checkpoint provenance without importing or running models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_record(path):
    path = path.resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def probe(arguments):
    """Bound read-only metadata queries; unavailable utilities remain explicit."""
    executable = shutil.which(arguments[0])
    if executable is None:
        return {"command": arguments, "available": False}
    try:
        result = subprocess.run(arguments, text=True, capture_output=True, timeout=20, check=False)
        return {
            "command": arguments,
            "executable": executable,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except subprocess.TimeoutExpired:
        return {"command": arguments, "executable": executable, "timed_out": True}


def collect(cohort, study, output):
    cohort, study, output = (Path(p).resolve() for p in (cohort, study, output))
    if output.exists():
        raise FileExistsError("Preserve the original post-hoc runtime inventory")
    repo = Path(__file__).resolve().parents[1]
    protocol = read(cohort / "protocol.json")
    scenario_path = cohort / sorted(protocol["cases"])[0] / "rapid_survey/scenario.json"
    specs = read(scenario_path)["executor_configs"]
    torchvision_source = (
        repo / ".venv/lib/python3.12/site-packages/torchvision/models/detection/mask_rcnn.py"
    )
    checkpoint = repo / ".cache/torch/hub/checkpoints/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth"
    url = "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth"
    if url not in torchvision_source.read_text():
        raise ValueError("Installed Mask R-CNN weight URL differs from this run's enum")
    weights, configuration_weights = {}, {}
    for spec in specs:
        params = spec["parameters"]
        if "checkpoint" in params:
            path = Path(params["checkpoint"])
            entry = file_record(path)
            if entry["sha256"] != params["checkpoint_sha256"]:
                raise ValueError("Frozen YOLO checkpoint digest differs")
            entry["digest_bound_before_inference"] = True
            entry["binding"] = "Frozen scenario checkpoint_sha256"
        elif "bundle_manifest" in params:
            manifest = Path(params["bundle_manifest"])
            if sha(manifest) != params["bundle_sha256"]:
                raise ValueError("Frozen CABiNet bundle manifest differs")
            bundle = read(manifest)
            path = manifest.parent / bundle["weights"]
            entry = file_record(path)
            if entry["sha256"] != bundle["files"][bundle["weights"]]:
                raise ValueError("Frozen CABiNet checkpoint digest differs")
            entry.update(
                digest_bound_before_inference=True,
                binding="Frozen scenario bundle_sha256 and bundle file manifest",
                bundle_manifest=file_record(manifest),
                source_url=bundle["source_url"],
                source_commit=bundle["source_commit"],
                weights_revision=bundle["revision"],
                code_license=bundle["code_license"],
                weights_license=bundle["weights_license"],
            )
        else:
            if (
                params.get("model_id") != "maskrcnn_resnet50_fpn_v2"
                or params.get("weights_id") != "COCO_V1"
            ):
                raise ValueError("Unrecognized frozen model enum")
            path = checkpoint
            entry = file_record(path)
            entry.update(
                digest_bound_before_inference=False,
                binding="COCO_V1 enum was frozen; this exact cached-file digest is post hoc",
                official_url_in_installed_source=url,
                official_filename_hash_prefix_matches=entry["sha256"].startswith("73cbd019"),
                installed_enum_source=file_record(torchvision_source),
            )
        key = path.resolve().as_posix()
        weights[key] = entry
        configuration_weights[spec["config_id"]] = key
    cpu = {}
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if ":" in line:
            key, value = (part.strip() for part in line.split(":", 1))
            if key in {
                "model name",
                "vendor_id",
                "cpu family",
                "model",
                "stepping",
                "cpu cores",
                "siblings",
            }:
                cpu.setdefault(key, value)
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    package_query = probe(
        [
            "dpkg-query",
            "-W",
            "-f=${binary:Package}\t${Version}\t${db:Status-Abbrev}\n",
            "libgz-*",
            "gz-*",
            "python3-gz-*",
            "libdart*",
            "libogre*",
            "*mesa*",
            "libegl*",
            "libglx*",
            "python3-av",
            "ffmpeg",
        ]
    )
    installed = []
    for line in package_query.get("stdout", "").splitlines():
        fields = line.split("\t")
        if len(fields) == 3 and fields[2].startswith("ii"):
            installed.append({"package": fields[0], "version": fields[1]})
    package_query.pop("stdout", None)
    display = probe(["lspci", "-nn"])
    if "stdout" in display:
        display["stdout"] = "\n".join(
            line
            for line in display["stdout"].splitlines()
            if any(
                kind in line
                for kind in ("VGA compatible controller", "3D controller", "Display controller")
            )
        )
        display["output_filter"] = "Display, VGA and 3D controller entries only"
    runtime = cohort / "runtime_inventory.json"
    result = {
        "schema_version": "posthoc_runtime_reproduction_inventory_v1",
        "recorded_utc": datetime.now(UTC).isoformat(),
        "scope": "Read-only inventory of the existing run environment and cached files",
        "posthoc": True,
        "installable_lockfile": False,
        "clean_machine_reproduction_demonstrated": False,
        "original_methods_or_protocols_changed": False,
        "policy_outcomes_read": False,
        "model_execution_performed": False,
        "exact_weight_state_during_past_inference_independently_observed": False,
        "cohort_protocol_sha256": sha(cohort / "protocol.json"),
        "comparison_protocol_sha256": sha(study / "protocol.json"),
        "scenario_sha256": sha(scenario_path),
        "earlier_runtime_inventory": file_record(runtime),
        "configuration_weights": configuration_weights,
        "checkpoint_files": weights,
        "cache_paths": {
            "torch_home_used_by_pipeline": str(repo / ".cache/torch"),
            "yolo_runtime": str(repo / ".cache/lake_yolo_runtime"),
            "yolo_weights": str(repo / ".cache/lake_yolo_weights"),
        },
        "host": {
            "os_release": Path("/etc/os-release").read_text(),
            "platform": platform.platform(),
            "cpu": cpu,
            "logical_cpu_count": os.cpu_count(),
            "memory_total_kib": int(memory["MemTotal"].split()[0]),
            "python": sys.version,
            "python_executable": sys.executable,
            "venv_configuration": (repo / ".venv/pyvenv.cfg").read_text(),
            "system_python": probe(["/usr/bin/python3", "--version"]),
            "compiler": probe(["g++", "--version"]),
        },
        "graphics_and_physics": {
            "gazebo_version": probe(["gz", "sim", "--versions"]),
            "gz_common_graphics_version": probe(
                ["pkg-config", "--modversion", "gz-common5-graphics"]
            ),
            "display_adapters": display,
            "installed_packages": installed,
            "package_query": package_query,
            "render_engine_declared_in_world": "ogre2",
            "observed_renderer_driver_at_acquisition": None,
            "limitation": (
                "Installed Mesa/DART versions are inventory, "
                "not proof of the exact loaded acquisition driver"
            ),
        },
        "source_sha256": sha(Path(__file__)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.cohort, args.study, args.output)
    print({"checkpoint_files": len(result["checkpoint_files"]), "output": str(args.output)})
