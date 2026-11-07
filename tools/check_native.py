#!/usr/bin/env python3
"""Run fresh CPU inference and compare three model families with recorded masks.

Default: capture IDs 0, 9, 17 of shoreline_loop/sector_3, for nine comparisons.
Requires the optional native dependencies and a verified local model cache.
This is a parity check, not a new timing study or a new simulation flight.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import fetch_models  # noqa: E402
from replay import checked_path, read, save, sha, source_hashes, verify_data  # noqa: E402


def run(data: Path, cache: Path, output: Path, case_id: str, capture_ids: list[int]) -> dict:
    data, cache, output = data.resolve(), cache.resolve(), output.resolve()
    if output.exists():
        raise ValueError("Choose a new output file; existing validation receipts are preserved")
    if len(set(capture_ids)) != len(capture_ids) or not capture_ids:
        raise ValueError("Capture IDs must be nonempty and unique")
    verify_data(data)
    if fetch_models.main(["--cache-root", str(cache), "--verify-only"]):
        raise ValueError("Model verification failed; run tools/fetch_models.py first")
    specs_path = REPO / "data/native_models.json"
    configuration = read(specs_path)
    if configuration["schema_version"] != "perception_native_models_v1":
        raise ValueError("Unsupported native model specification")
    bindings = read(data / "native_bindings.json")["bindings"]
    binding = next((item for item in bindings if item["case_id"] == case_id), None)
    if binding is None:
        raise ValueError("Unknown case: " + case_id)
    frames = {item["capture_id"]: item for item in binding["frames"]}
    if not set(capture_ids) <= frames.keys():
        raise ValueError("Requested capture ID is not in the declared inference inputs")
    record_path = checked_path(data, binding["recording_path"])
    expected_names = {"CFG_MASKRCNN_1280", "CFG_YOLO11S_960_ROT0", "CFG_CABINET_UAVID_1280"}
    configs = configuration["executor_configs"]
    if len(configs) != 3 or {s["config_id"] for s in configs} != expected_names:
        raise ValueError("Exactly the three paper model configurations are required")
    for spec in configs:
        for key in ("checkpoint", "bundle_manifest"):
            if key in spec["parameters"]:
                spec["parameters"][key] = str(checked_path(cache, spec["parameters"][key]))
    os.environ["TORCH_HOME"] = str(cache / "torch")
    os.environ["YOLO_CONFIG_DIR"] = str(cache / "ultralytics")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    import numpy as np
    import torch

    from experiments.mission_routing_v2.models import InferencePool
    from missionbench.v2.scenario import ExecutorConfigSpec

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    hashes_before = source_hashes()
    hashes_before.update(
        {
            str(p.relative_to(REPO)): sha(p)
            for p in (Path(__file__), REPO / "tools/fetch_models.py", specs_path)
        }
    )
    pool = InferencePool(tuple(ExecutorConfigSpec(**spec) for spec in configs))
    rows = []
    for capture_id in capture_ids:
        frame = frames[capture_id]
        source = checked_path(record_path.parent, frame["source_file"])
        if sha(source) != frame["source_sha256"]:
            raise ValueError("RGB capture changed after verification")
        # The npz also contains evaluator labels. Only RGB reaches the model pool.
        with np.load(source, allow_pickle=False) as archive:
            rgb = archive["rgb"].copy()
        masks, durations, calls = pool.infer(rgb)
        if set(masks) != expected_names:
            raise ValueError("Native inference did not return the three expected masks")
        for model_id in sorted(masks):
            mask = masks[model_id]
            if mask.shape != rgb.shape[:2] or mask.dtype != np.bool_:
                raise ValueError(f"Invalid native mask: {model_id}, capture {capture_id}")
            relative = f"predictions/{case_id}/prediction_{capture_id:05d}_{model_id}.npz"
            expected_path = checked_path(data, relative)
            with np.load(expected_path, allow_pickle=False) as archive:
                expected = archive["prediction_mask"]
                equal = bool(np.array_equal(mask, expected))
                changed = int(np.count_nonzero(mask != expected))
            rows.append(
                dict(
                    model=model_id,
                    capture_id=capture_id,
                    output_shape=list(mask.shape),
                    foreground_pixels=int(mask.sum()),
                    differing_pixels=changed,
                    cached_mask_exact_match=equal,
                    neural_calls=calls[model_id],
                    host_seconds=durations[model_id],
                    input_sha256=frame["source_sha256"],
                    expected_mask_file_sha256=sha(expected_path),
                )
            )
            print(
                f"{'MATCH' if equal else 'MISMATCH'} {model_id}, capture {capture_id}", flush=True
            )
    # Validate source/data again so a concurrent edit cannot silently pass the check.
    for relative, digest in hashes_before.items():
        if sha(REPO / relative) != digest:
            raise ValueError("Release implementation changed during native validation")
    verify_data(data)
    versions = {}
    for package in (
        "torch",
        "torchvision",
        "ultralytics",
        "numpy",
        "pillow",
        "opencv-python",
        "opencv-python-headless",
    ):
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            versions[package] = importlib.metadata.version(package)
    receipt = dict(
        schema_version="native_mask_parity_v1",
        case_id=case_id,
        capture_ids=capture_ids,
        complete=len(rows) == len(capture_ids) * 3,
        fresh_neural_inference=True,
        new_simulator_flight=False,
        exact_mask_parity=all(row["cached_mask_exact_match"] for row in rows),
        comparisons=len(rows),
        rows=rows,
        python_version=platform.python_version(),
        package_versions=versions,
        torch_threads=4,
        source_hashes=hashes_before,
        model_assets=[
            dict(path=a.path, sha256=a.sha256, bytes=a.size_bytes) for a in fetch_models.assets()
        ],
        replay_manifest_sha256=sha(data / "manifest.json"),
        scope=(
            "Fresh CPU predictions on selected recorded RGB captures. "
            "Host durations are diagnostic; this is not a new timing or accuracy study."
        ),
    )
    save(output, receipt)
    print(json.dumps({key: receipt[key] for key in ("comparisons", "exact_mask_parity")}, indent=2))
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO / "data/replay")
    parser.add_argument("--cache-root", type=Path, default=fetch_models.DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=REPO / "results/native-validation.json")
    parser.add_argument("--case", default="shoreline_loop/sector_3")
    parser.add_argument("--captures", nargs="+", type=int, default=[0, 9, 17])
    args = parser.parse_args()
    receipt = run(args.data, args.cache_root, args.output, args.case, args.captures)
    return 0 if receipt["complete"] and receipt["exact_mask_parity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
