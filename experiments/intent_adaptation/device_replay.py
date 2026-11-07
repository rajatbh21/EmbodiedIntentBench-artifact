"""Device measurement of contract-satisfied gating: frame-paced fixed local Mask R-CNN on the Orin.

For each recording and each (contract, gate) schedule, captures are released on the wall clock at their
recorded times. Captures listed in the schedule are processed with real inference (blocking, Eq. 5);
captures after the gate closes are not processed (the router abstains). Non-inference stages are emulated
with their declared durations, as in orin_realtime.py.

Measured with module power = VDD_GPU_SOC + VDD_CPU_CV + VIN_SYS_5V0:
  - inference energy  : module power integrated over each inference call (what replay charges as compute)
  - mission energy    : module power integrated over the whole pass [0, 36 s + last completion]
  - idle baseline     : 6 s before each run

The output retains integrated per-run values and sample counts, not raw power
traces. This runs supplied schedules; it does not implement the online gate.
"""
import sys, os, json, time, argparse
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'experiments' / 'jetson' / 'scripts'))
import orin_profile as op
import numpy as np

PRE_S, POST_S, LOAD_S = 0.07, 0.10, 0.35


def integrate(raw, t0, t1):
    pts = [(t, op.rail_sum(r)) for t, r in raw if t0 <= t <= t1]
    if len(pts) < 2:
        return float("nan"), len(pts)
    ts = np.array([p[0] for p in pts]); ps = np.array([p[1] for p in pts])
    return float(np.trapz(ps, ts)), len(pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True, help="Path to the benchmark checkout")
    ap.add_argument("--schedules", required=True)  # json: {key: {case: [capture ids to process]}}
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    output = Path(a.out)
    if output.exists():
        ap.error("Use a new output file; prior measurements are preserved")
    output.parent.mkdir(parents=True, exist_ok=True)
    rel = Path(a.release).resolve()
    sys.path.insert(0, str(rel)); sys.path.insert(0, str(rel / "tools"))
    os.environ["TORCH_HOME"] = str(rel / ".cache/torch")
    import torch
    torch.set_num_threads(4)
    from dataclasses import replace
    from missionbench.v2.scenario import ExecutorConfigSpec
    from missionbench.v2.instance_models import InstanceBackend, TorchInstanceSegmentationExecutor
    import experiments.mission_execution_profile as mep
    specs = {s["config_id"]: s for s in json.load(open(rel / "data/native_models.json"))["executor_configs"]}
    control = ExecutorConfigSpec(**specs[mep.CONTROL])
    control = replace(control, parameters={**control.parameters, "device": "cuda"})
    pe = mep.PipelineExecutor(control, control, factory=lambda s: TorchInstanceSegmentationExecutor(s, backend=InstanceBackend(s.parameters)))
    bindings = {b["case_id"]: b for b in json.load(open(rel / "data/replay/native_bindings.json"))["bindings"]}
    schedules = json.load(open(a.schedules))
    sp = op.Sampler(); sp.start(); time.sleep(1)
    out = dict(meta=dict(nvpmodel=os.popen("nvpmodel -q 2>/dev/null | grep -A1 'NV Power Mode' | head -1").read().strip(),
                         total_rails=list(op.TOTAL_RAILS)), runs=[])
    for key, per_case in schedules.items():
        for case, todo in per_case.items():
            b = bindings[case]; rec = rel / "data/replay" / Path(b["recording_path"]).parent
            frames = []
            for f in b["frames"]:
                with np.load(rec / f["source_file"], allow_pickle=False) as z:
                    frames.append((f["capture_id"], f["mission_time_s"], z["rgb"].copy()))
            for _ in range(3):
                pe.run(frames[0][2])
            torch.cuda.synchronize()
            i0 = time.perf_counter(); time.sleep(6); i1 = time.perf_counter()
            idle_j, _ = integrate(sp.raw, i0, i1); idle_w = idle_j / (i1 - i0)
            t0 = time.perf_counter() + 0.5
            busy_until, first, calls = 0.0, True, []
            todo_set = set(todo)
            for cid, mt, rgb in frames:
                if cid not in todo_set or mt < busy_until:
                    continue
                while time.perf_counter() < t0 + mt:
                    time.sleep(0.0005)
                time.sleep(PRE_S + (LOAD_S if first else 0.0))
                c0 = time.perf_counter(); pe.run(rgb); c1 = time.perf_counter()
                time.sleep(POST_S)
                calls.append((cid, c0, c1)); busy_until = time.perf_counter() - t0; first = False
            end = max(t0 + 36.0, time.perf_counter())
            while time.perf_counter() < end:
                time.sleep(0.01)
            mission_j, n = integrate(sp.raw, t0, end)
            inf = [integrate(sp.raw, c0, c1)[0] for _, c0, c1 in calls]
            out["runs"].append(dict(key=key, case=case, processed=[c for c, _, _ in calls], n_calls=len(calls),
                                    inference_j=float(np.nansum(inf)), mission_j=mission_j, window_s=end - t0,
                                    idle_w=idle_w, mission_above_idle_j=mission_j - idle_w * (end - t0), samples=n))
            r = out["runs"][-1]
            print(f"{key:24s} {case:26s} calls={r['n_calls']:2d} inference={r['inference_j']:7.1f} J "
                  f"mission={r['mission_j']:7.1f} J above_idle={r['mission_above_idle_j']:6.1f} J idle={idle_w:.2f} W", flush=True)
            output.write_text(json.dumps(out, indent=1))
    sp.stop = True


if __name__ == "__main__":
    main()
