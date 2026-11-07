"""Run independent policy cohorts concurrently, preserving logs and failed outputs."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from experiments.mission_routing_v2.flights import verify
from experiments.mission_routing_v2_checked import require_route


def run(root, jobs=6, wait_for_inference=False):
    p = verify(root)
    if not 1 <= jobs <= 6:
        raise ValueError("Use one to six independent cohort workers")
    if (root / "runs").exists():
        raise FileExistsError("Preserve prior runs; this batch does not overwrite or resume them")
    if wait_for_inference:
        deadline = time.monotonic() + 1800
        while not all((root / f / "predictions/completion.json").exists() for f in p["cases"]):
            if time.monotonic() > deadline:
                raise TimeoutError("Inference did not complete within 30 minutes")
            print("Waiting for complete verified inference caches", flush=True)
            time.sleep(30)
    require_route(root)
    pending = [(f, power) for f in p["cases"] for power in p["active_cpu_power_sensitivity_w"]]
    active = []
    logs = root / "batch_logs"
    logs.mkdir(exist_ok=False)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONPATH=".")
    try:
        while pending or active:
            while pending and len(active) < jobs:
                flight, power = pending.pop(0)
                path = logs / f"{flight}_{power}.log"
                handle = path.open("w")
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "experiments.mission_routing_v2.study",
                        "--output",
                        str(root),
                        "--flight",
                        flight,
                        "--power",
                        str(power),
                    ],
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                )
                active.append((process, handle, path))
                print(f"Started {flight}, {power} W", flush=True)
            for process, handle, path in active[:]:
                status = process.poll()
                if status is not None:
                    handle.close()
                    active.remove((process, handle, path))
                    if status:
                        raise RuntimeError(f"Cohort failed; inspect {path}")
                    print(f"Completed {path.stem}", flush=True)
            if active:
                time.sleep(3)
        print("All six cohorts completed: 306 policy executions", flush=True)
    finally:
        for process, handle, _ in active:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
            handle.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--wait-for-inference", action="store_true")
    args = parser.parse_args()
    run(args.output, args.jobs, args.wait_for_inference)
