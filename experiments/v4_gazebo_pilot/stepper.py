"""Persistent service-only subprocess; keeps blocking requests off sensor callbacks.

One request means one step. A lost acknowledgement is fatal and is never retried.
"""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading

from experiments.v4_gazebo_pilot.world import WORLD


class ServiceStepper:
    def __init__(self):
        self.process = subprocess.Popen(
            [sys.executable, "-m", "experiments.v4_gazebo_pilot.stepper"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.failed = False
        self.responses = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.process.stdout:
            self.responses.put(line)
        self.responses.put("")

    def step(self, count: int, timeout_s: float):
        if self.failed:
            raise RuntimeError("stepper is invalid after an ambiguous request")
        self.failed = True
        self.process.stdin.write(
            json.dumps({"count": count, "timeout_ms": round(timeout_s * 1000)}) + "\n"
        )
        self.process.stdin.flush()
        try:
            reply = self.responses.get(timeout=timeout_s + 2)
        except queue.Empty:
            raise TimeoutError("step acknowledgement missing; not retried") from None
        if not reply or json.loads(reply).get("ok") is not True:
            raise RuntimeError("step service failed; not retried")
        self.failed = False

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)


def main():
    from gz.msgs10.boolean_pb2 import Boolean
    from gz.msgs10.world_control_pb2 import WorldControl
    from gz.transport13 import Node

    node = Node()
    for line in sys.stdin:
        data = json.loads(line)
        request = WorldControl(pause=True, multi_step=data["count"])
        success, response = node.request(
            f"/world/{WORLD}/control", request, WorldControl, Boolean, data["timeout_ms"]
        )
        print(json.dumps({"ok": bool(success and response.data)}), flush=True)
        if not success or not response.data:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
