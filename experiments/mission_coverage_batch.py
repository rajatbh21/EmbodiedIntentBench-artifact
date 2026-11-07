"""Bounded local execution of the frozen prospective study; no policy tuning."""

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.mission_coverage_routing.cohort import verify
from experiments.mission_coverage_routing.sensor import verify_gate


def run(cohort, stage, workers):
    if workers not in (1, 2):
        raise ValueError("Use one or two workers on this host")
    p = verify(cohort)
    verify_gate(cohort)
    root = cohort if stage == "infer" else cohort / "study"
    cases = list(p["cases"] if stage == "infer" else read(root / "protocol.json")["cases"])
    modules = dict(
        infer="experiments.mission_coverage_routing.cohort",
        replay="experiments.mission_coverage_routing.study",
        validate="experiments.mission_coverage_routing.validate",
        witness="experiments.mission_coverage_witness",
    )
    logs = cohort / "logs"
    logs.mkdir(exist_ok=True)

    def task(case):
        command = [sys.executable, "-m", modules[stage]] + (["infer"] if stage == "infer" else [])
        command += ["--output", str(root), "--flight", case]
        log = logs / f"{stage}_{case}.log"
        start = time.monotonic()
        with log.open("x") as stream:
            completed = subprocess.run(
                command, stdout=stream, stderr=subprocess.STDOUT, check=False
            )
        row = dict(
            case=case,
            returncode=completed.returncode,
            elapsed_s=time.monotonic() - start,
            command=command,
            log=str(log),
            log_sha256=sha(log),
        )
        print(row, flush=True)
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(task, c) for c in cases]
        for future in as_completed(futures):
            rows.append(future.result())
            save(
                logs / f"{stage}_progress.json",
                dict(stage=stage, expected=len(cases), finished=len(rows), cases=rows),
            )
    if any(r["returncode"] for r in rows):
        raise RuntimeError("Failed tasks retained; inspect their logs")
    save(
        logs / f"{stage}_completion.json",
        dict(
            complete=True,
            cases=rows,
            source_sha256=sha(Path(__file__)),
            protocol_sha256=sha(root / "protocol.json"),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("infer", "replay", "validate", "witness"))
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    a = parser.parse_args()
    run(a.cohort, a.stage, a.workers)
