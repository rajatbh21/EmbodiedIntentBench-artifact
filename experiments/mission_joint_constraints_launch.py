"""Launch four disjoint immutable shards with explicit affinity and thread settings."""

import argparse
import os
import shlex
import subprocess
import time
from pathlib import Path

from experiments.mission_joint_constraints import OUTPUT, REPO, require, save, sha, verify


def launch(root, stage):
    root = Path(root)
    plan = verify(root)
    require(stage in ("execute", "audit"), "Unknown stage")
    require(len(plan["shards"]) == 4, "This bounded launcher requires four declared shards")
    record = root / (stage + "_commands.json")
    require(not record.exists(), "Preserve launch history")
    environment = dict(os.environ)
    explicit = dict.fromkeys(
        (
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "BLIS_NUM_THREADS",
        ),
        "1",
    )
    explicit["PYTHONPATH"] = ".cache/lake_yolo_runtime:."
    environment.update(explicit)
    commands = {}
    for core, shard in zip((0, 2, 4, 6), plan["shards"], strict=True):
        command = ["taskset", "-c", str(core), str(REPO / ".venv/bin/python"), "-u", "-m"]
        command.extend(
            ["experiments.mission_joint_constraints", "execute"]
            if stage == "execute"
            else ["experiments.mission_joint_constraints_audit"]
        )
        command.extend(["--output", str(root), "--shard", shard])
        commands[shard] = dict(argv=command, shell_display=shlex.join(command), cpu=core)
    save(
        record,
        dict(
            stage=stage,
            plan_sha256=sha(root / "plan.json"),
            commands=commands,
            explicit_environment=explicit,
            launcher_sha256=sha(Path(__file__)),
            started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            wall_time_is_diagnostic_only=True,
            core_topology="lscpu: CPUs0,2,4,6 are distinct physical cores0,1,2,3",
        ),
    )
    processes, logs, exits = {}, {}, {}
    try:
        for shard, command in commands.items():
            log = (root / (stage + "_" + shard + ".log")).open("x")
            logs[shard] = log
            processes[shard] = subprocess.Popen(
                command["argv"], cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT
            )
        for shard, process in processes.items():
            exits[shard] = process.wait()
            print(stage, shard, "exit", exits[shard], flush=True)
    finally:
        for log in logs.values():
            log.close()
    save(
        root / (stage + "_process_completion.json"),
        dict(
            complete=len(exits) == 4 and all(code == 0 for code in exits.values()),
            exit_codes=exits,
            commands_sha256=sha(record),
            finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        ),
    )
    require(
        len(exits) == 4 and all(code == 0 for code in exits.values()),
        "At least one shard failed; preserve every output and do not rerun",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("execute", "audit"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    launch(args.output, args.stage)
