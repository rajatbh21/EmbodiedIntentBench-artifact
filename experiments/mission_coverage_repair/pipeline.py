"""Run and independently validate each condition as its dense predictions finish."""

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.mission_coverage_repair.protocol import verify


def run(cohort):
    root = cohort / "study_repair_v1"
    p = verify(root)
    logs = cohort / "logs"
    logs.mkdir(exist_ok=True)

    def condition(flight):
        log = logs / f"repair_pipeline_{flight}.log"
        start = time.monotonic()
        with log.open("x") as stream:
            for module in ("study", "validate"):
                command = [
                    sys.executable,
                    "-m",
                    "experiments.mission_coverage_repair." + module,
                    "--output",
                    str(root),
                    "--flight",
                    flight,
                ]
                completed = subprocess.run(
                    command, stdout=stream, stderr=subprocess.STDOUT, check=False
                )
                if completed.returncode:
                    raise RuntimeError(f"{flight} {module} failed; see {log}")
        result = dict(
            flight=flight, elapsed_s=time.monotonic() - start, log=str(log), log_sha256=sha(log)
        )
        print(result, flush=True)
        return result

    submitted = {}
    completed = []
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as pool:
        while len(completed) < len(p["cases"]):
            for flight, c in p["cases"].items():
                ready = cohort / c["source_flight"] / "predictions/completion.json"
                if flight not in submitted and ready.exists() and read(ready)["complete"]:
                    submitted[flight] = pool.submit(condition, flight)
            done = [f for f, v in submitted.items() if v is not None and v.done()]
            for f in done:
                completed.append(submitted[f].result())
                submitted[f] = None
                save(
                    logs / "repair_pipeline_progress.json",
                    dict(expected=len(p["cases"]), finished=len(completed), cases=completed),
                )
            progress = logs / "infer_progress.json"
            if progress.exists() and any(c["returncode"] for c in read(progress)["cases"]):
                raise RuntimeError("An inference failed")
            if time.monotonic() - start > 12 * 3600:
                raise TimeoutError("Preserve unfinished pipeline after twelve hours")
            if len(completed) < len(p["cases"]):
                time.sleep(5)
    save(
        logs / "repair_pipeline_completion.json",
        dict(complete=True, cases=completed, source_sha256=sha(Path(__file__))),
    )
    from experiments.mission_coverage_repair.audit import run as audit

    audit(root)
    from experiments.mission_coverage_repair.report import run as report

    report(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    run(parser.parse_args().cohort)
