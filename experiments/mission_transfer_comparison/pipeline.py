"""Run bounded dense inference or replay/audit ready source recordings."""

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.mission_environment_transfer.cohort import verify as verify_cohort
from experiments.mission_environment_transfer.sensor import verify_gate
from experiments.mission_transfer_comparison.protocol import verify


def command(module, root, flight, log):
    start = time.monotonic()
    args = [sys.executable, "-m", module, "--output", str(root), "--flight", flight]
    with log.open("x") as stream:
        completed = subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT, check=False)
    result = dict(
        flight=flight,
        elapsed_s=time.monotonic() - start,
        returncode=completed.returncode,
        command=args,
        log=str(log),
        log_sha256=sha(log),
    )
    print(result, flush=True)
    return result


def infer(cohort, workers):
    p = verify_cohort(cohort)
    verify_gate(cohort)
    # The source RGB and outputs may only be used under a pre-acquisition method freeze.
    verify(cohort / "study")
    logs = cohort / "logs"
    logs.mkdir(exist_ok=True)
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                command,
                "experiments.mission_environment_transfer.infer",
                cohort,
                flight,
                logs / f"infer_{flight}.log",
            )
            for flight in p["cases"]
        ]
        for future in as_completed(futures):
            rows.append(future.result())
            save(
                logs / "infer_progress.json",
                dict(expected=len(p["cases"]), finished=len(rows), cases=rows),
            )
    if any(r["returncode"] for r in rows):
        raise RuntimeError("Failed inference attempts preserved in their original logs")
    save(
        logs / "infer_completion.json",
        dict(
            complete=True,
            cases=rows,
            protocol_sha256=sha(cohort / "protocol.json"),
            source_sha256=sha(Path(__file__)),
        ),
    )


def replay(cohort, workers):
    root = cohort / "study"
    p = verify(root)
    logs = cohort / "logs"
    logs.mkdir(exist_ok=True)

    def condition(flight):
        tasks = []
        for stage in ("study", "validate"):
            task = command(
                "experiments.mission_transfer_comparison." + stage,
                root,
                flight,
                logs / f"{stage}_{flight}.log",
            )
            tasks.append(task)
            if task["returncode"]:
                raise RuntimeError(
                    "Execution or validation failed; preserve attempt: " + task["log"]
                )
        return dict(flight=flight, stages=tasks)

    submitted, finished = {}, []
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while len(finished) < len(p["cases"]):
            for flight, case in p["cases"].items():
                ready = cohort / case["source_flight"] / "predictions/completion.json"
                if flight not in submitted and ready.exists() and read(ready)["complete"]:
                    submitted[flight] = pool.submit(condition, flight)
            done = [f for f, value in submitted.items() if value is not None and value.done()]
            for flight in done:
                finished.append(submitted[flight].result())
                submitted[flight] = None
                save(
                    logs / "replay_progress.json",
                    dict(expected=len(p["cases"]), finished=len(finished), cases=finished),
                )
            progress = logs / "infer_progress.json"
            if progress.exists() and any(c["returncode"] for c in read(progress)["cases"]):
                raise RuntimeError("Inference failed; pipeline stopped with previous work retained")
            if time.monotonic() - start > 12 * 3600:
                raise TimeoutError("Preserve incomplete pipeline after twelve hours")
            if len(finished) < len(p["cases"]):
                time.sleep(5)
    from experiments.mission_transfer_comparison.audit import run as audit
    from experiments.mission_transfer_comparison.report import run as report

    audit(root)
    report(root)
    save(
        logs / "replay_completion.json",
        dict(complete=True, cases=finished, source_sha256=sha(Path(__file__))),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("infer", "replay"))
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    a = parser.parse_args()
    (infer if a.stage == "infer" else replay)(a.cohort, a.workers)
