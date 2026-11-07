"""Owned Gazebo worker using physical environment configuration."""

import os
import queue
import subprocess
import threading
from pathlib import Path

from experiments.v4_benchmark.backends import LiveBackend as BaseBackend
from experiments.v4_benchmark.backends import save_json


class LiveBackend(BaseBackend):
    def __init__(self, scenario, output: Path, *, python: str = "/usr/bin/python3"):
        self.output = output.resolve()
        self.output.parent.mkdir(parents=True, exist_ok=True)
        spec = self.output.parent / f"{self.output.name}_worker_spec.json"
        save_json(
            spec,
            {
                "world": scenario.world,
                "warmup_s": scenario.warmup_s,
                "environment": vars(scenario.environment),
            },
        )
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        self.log = self.output.with_suffix(".worker.log").open("w")
        self.process = subprocess.Popen(
            [
                python,
                "-m",
                "experiments.routing_mission.worker",
                "--spec",
                str(spec),
                "--output",
                str(self.output),
            ],
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        self.responses: queue.Queue[str] = queue.Queue()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        try:
            ready = self._response()
            self.time_ns = ready["time_ns"]
            self.provenance = {
                "backend": "live_gazebo",
                "gazebo_version": ready["gazebo_version"],
                "recording_identity": scenario.recording_identity,
            }
        except BaseException:
            self.close()
            raise
