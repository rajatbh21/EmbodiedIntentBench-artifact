"""Contrast declared mission latency with observed host timings without retuning clocks."""

import argparse
from pathlib import Path

import numpy as np

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.mission_coverage_repair.protocol import verify
from experiments.mission_coverage_repair.study import scenario_for
from experiments.mission_routing_v2.train import checked_manifest


def run(root):
    p = verify(root)
    cohort = Path(p["cohort"])
    spec = scenario_for(root, p, next(iter(p["cases"])), "rapid_survey", 15)
    values = {s.config_id: [] for s in spec.executors}
    inputs = {}
    for flight in sorted({c["source_flight"] for c in p["cases"].values()}):
        folder = cohort / flight / "predictions"
        checked_manifest(folder, cohort / "protocol.json")
        rows = read(folder / "inference.json")
        expected = {(i, config) for i in range(60) for config in values}
        if (
            len(rows) != len(expected)
            or {(r["capture_id"], r["config_id"]) for r in rows} != expected
        ):
            raise ValueError("Incomplete measured inference grid")
        for row in rows:
            elapsed = row["measured_wall_clock_s"]
            if not np.isfinite(elapsed) or elapsed <= 0:
                raise ValueError("Invalid timing")
            values[row["config_id"]].append(elapsed)
        inputs[str(folder / "inference.json")] = sha(folder / "inference.json")
    summary = []
    for s in spec.executors:
        v = values[s.config_id]
        summary.append(
            dict(
                config_id=s.config_id,
                observations=len(v),
                declared_latency_s=s.mission_latency_s,
                host_median_s=float(np.median(v)),
                host_p95_s=float(np.percentile(v, 95)),
                host_max_s=max(v),
                host_above_declared=sum(t > s.mission_latency_s for t in v),
            )
        )
    save(
        root / "host_timing.json",
        dict(
            complete=True,
            summary=summary,
            source_sha256=sha(Path(__file__)),
            inputs_sha256=inputs,
            protocol_sha256=sha(root / "protocol.json"),
            protocol_costs_changed=False,
            controlled_hardware_calibration=False,
            simulator_time_replaced_by_host_time=False,
            scope="Post-hoc concurrent host timings, including cold/warm executions",
            warnings=[
                "Two inference workers with four torch threads each; other host load uncontrolled",
                "Union timings include component calls and must not be added to them again",
                "Does not establish real-time onboard feasibility or measured energy",
            ],
        ),
    )
    lines = [
        "# Declared latency and observed host timing",
        "",
        "The policy experiment uses the frozen earlier CPU latency calibration, not wall-clock "
        "time under concurrent experiment orchestration. The table exposes the difference: "
        "two inference workers used four PyTorch threads each, with other host work uncontrolled. "
        "Warmup and cold frames are retained. These data are not a new controlled deployment "
        "profile, and no mission clock, action or result was changed after observing them.",
        "",
        "| Configuration | Declared (s) | Host median (s) | Host p95 (s) | Above declared /360 |",
        "|---|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {r['config_id']} | {r['declared_latency_s']:.3f} | {r['host_median_s']:.3f} | "
        f"{r['host_p95_s']:.3f} | {r['host_above_declared']} |"
        for r in summary
    ]
    lines += [
        "",
        "A discrepancy is a deployment-validity limitation, not a reason to substitute "
        "a faster observed value into selected successful cells. A future deployment "
        "study must freeze a measured target-device profile before rerunning all paired "
        "policies. Router host timings are reported separately in `analysis.json`; "
        "they are also diagnostic and exclude model execution.",
        "",
    ]
    (root / "HOST_TIMING.md").write_text("\n".join(lines))
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
